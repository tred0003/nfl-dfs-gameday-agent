"""Two-stage search: an embedding model fetches 50, a ranker re-orders them.

    python embed_tune/rerank.py --first base          # text-embedding-005, then the ranker
    python embed_tune/rerank.py --first gemini_emb    # gemini-embedding-001, then the ranker

An embedding model turns the question and each passage into numbers
SEPARATELY and compares them afterwards (a bi-encoder). Fast -- the passages
are embedded once, ahead of time -- but it never reads the two together.
That is the near-miss problem the tuning runs kept hitting on injury
questions: an older note about the same ankle looks almost the same as
numbers.

A ranker (a cross-encoder) reads the question and one passage together and
scores the pair. Much better at "does this answer it", far too slow to run
over 11,894 passages, so it re-orders the embedding model's top 50.

The ranker is Google's Vertex AI Ranking API (Discovery Engine,
semantic-ranker-default@latest): one call per question, up to 200 passages,
priced per call (~$1 per 1,000). Passages go in as plain text, nothing else
-- no date, no headline field -- so the comparison with the embedding models
stays about reading, not extra metadata.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

from common import CUTOFF, KEY, embed_queries, load_window, name_mask, unit
from eval import K, K_HIT, load_qrels, ndcg, summary

PROJECT = "fantasy-football-ml-503718"
RANKER = "semantic-ranker-default@latest"
URL = (f"https://discoveryengine.googleapis.com/v1/projects/{PROJECT}/locations/global/"
       "rankingConfigs/default_ranking_config:rank")
FIRST_K = 50


def first_stage(name: str, qs, df):
    """(question vectors, passage vectors) for the retriever that fetches the 50."""
    if name == "base":
        return embed_queries([q["question"] for q in qs]), unit(np.vstack(df.embedding.to_numpy()))
    from gemini_embed import CACHE, embed
    from common import genai_client
    return (unit(embed(genai_client(), [q["question"] for q in qs], "RETRIEVAL_QUERY")),
            unit(np.load(CACHE)))


class Ranker:
    def __init__(self):
        import google.auth
        import google.auth.transport.requests
        self.creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        self.req = google.auth.transport.requests.Request()
        self.http = requests.Session()

    def rank(self, question: str, ids: list[str], texts: list[str],
             titles: list[str] | None = None) -> list[str]:
        if not self.creds.valid:
            self.creds.refresh(self.req)
        records = [{"id": str(i), "content": t} for i, t in enumerate(texts)]
        for rec, title in zip(records, titles or []):
            rec["title"] = title
        body = {"model": RANKER, "query": question, "topN": len(ids), "ignoreRecordDetailsInResponse": True,
                "records": records}
        for attempt in range(5):
            try:
                r = self.http.post(URL, json=body, timeout=60, headers={
                    "Authorization": f"Bearer {self.creds.token}", "x-goog-user-project": PROJECT})
            except requests.ConnectionError:      # dropped connections, as in gemini_embed.py
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
                continue
            if r.status_code not in (429, 500, 503):
                break
            time.sleep(2 ** attempt)
        r.raise_for_status()
        return [ids[int(rec["id"])] for rec in r.json()["records"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", choices=["base", "gemini_emb"], default="base")
    ap.add_argument("--dated", action="store_true",
                    help="give the ranker each passage's date (as its title) and today's date")
    args = ap.parse_args()

    qs, qrels = load_qrels(False)
    df = load_window()
    ids, text = df.doc_id.to_numpy(), dict(zip(df.doc_id, df.text))
    # --dated: the ranker reads text and knows no calendar, so "back in Week 4"
    # and "back this week" look alike to it. Found live on 2026-10-27: asked
    # when Puka Nacua would be back, it ranked September notes above Friday's.
    when = dict(zip(df.doc_id, df.published.dt.strftime("Published %a %b %d, %Y")))
    today = f"Today is {CUTOFF:%a %b %d, %Y}. " if args.dated else ""
    qvec, docs = first_stage(args.first, qs, df)

    jobs = []                                   # (qid, mode, question, 50 candidate ids)
    for i, q in enumerate(qs):
        if not any(g == 2 for g in qrels[q["qid"]].values()):
            continue
        sims = docs @ qvec[i]
        modes = [("open", sims)]
        if q["player"]:
            modes.append(("player", np.where(name_mask(df, q["player"]), sims, -np.inf)))
        for mode, s in modes:
            cands = [ids[j] for j in np.argsort(-s)[:FIRST_K] if np.isfinite(s[j])]
            jobs.append((q["qid"], mode, q["question"], cands))

    ranker = Ranker()
    with ThreadPoolExecutor(max_workers=4) as ex:
        ranked = list(ex.map(lambda j: ranker.rank(
            today + j[2], j[3], [text[d] for d in j[3]],
            [when[d] for d in j[3]] if args.dated else None), jobs))

    out = {"open": [], "player": []}
    for (qid, mode, _, _), order in zip(jobs, ranked):
        rel = qrels[qid]
        top = order[:K]
        gains = [rel.get(d, 0) for d in top]
        n2 = sum(g == 2 for g in rel.values())
        out[mode].append({"qid": qid, "hit": float(any(g == 2 for g in gains[:K_HIT])),
                          "recall": sum(g == 2 for g in gains) / min(n2, K),
                          "ndcg": ndcg(gains, list(rel.values())), "top": top})
    name = f"rerank_{args.first}" + ("_dated" if args.dated else "")
    with open(KEY / f"eval_{name}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"{len(jobs)} ranker calls (~${len(jobs) / 1000:.2f})")
    print(f"{args.first} top {FIRST_K} -> {RANKER}")
    print(summary(out))


if __name__ == "__main__":
    main()
