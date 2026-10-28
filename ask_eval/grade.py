"""Grade the Ask answers (run.py) three ways, then compare the two designs.

    python ask_eval/grade.py contract     # free: the citation check, re-run
    python ask_eval/grade.py grounding    # Google's check-grounding API, 1 call per answer
    python ask_eval/grade.py judge        # Vertex AI Gen AI evaluation service (autorater)
    python ask_eval/grade.py report       # the comparison -> ask_eval/results.json

Three graders, because each answers a different question:

  contract    Does the answer keep its own rules? citations.py on the stored
              text: unknown refs, uncited facts, dates no cited passage has,
              and the answer type (answered / partial / abstained /
              unsupported). Deterministic, so it is also a CI-able gate.
              Re-run here, not read from run.py, so a checker fix applies to
              answers already generated.

  grounding   Is every claim SUPPORTED by what was retrieved? The Discovery
              Engine check-grounding API reads the answer and the passages
              and scores each claim 0-1 (entailment, and strict: "on Sunday"
              in a claim needs "Sunday" in a fact). Each passage goes in WITH
              its date as text ("Published Wed Oct 21, 4 days before today"),
              or every correctly dated claim would score unsupported. Its own
              claim->fact citations are compared with the model's [P#] refs:
              citation agreement, an independent check on the citations.

  judge       Is the answer RIGHT? Grounded is not correct: an answer can be
              perfectly supported by an old note and miss this week's. The
              Gen AI evaluation service's autorater compares each answer with
              the answer key's relevant passages (embed_tune/key, the 150
              questions' graded passages, newest first) on a 0-2 rubric that
              puts the newest status first. For the 20 unanswerable
              questions the reference says so, and 2 means a clean refusal.
              The autorater is the service default (this SDK cannot choose
              it); calibrate.py checks it against Claude on a sample.

Each grader writes its own jsonl next to answers.jsonl and is resumable.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "webapp"), str(ROOT / "src"), str(ROOT / "embed_tune")]

HERE = ROOT / "ask_eval"
ANSWERS = HERE / "answers.jsonl"
CONTRACT = HERE / "graded_contract.jsonl"
GROUNDING = HERE / "graded_grounding.jsonl"
JUDGE = HERE / "graded_judge.jsonl"
RESULTS = HERE / "results.json"
PROJECT = "fantasy-football-ml-503718"   # Discovery Engine is enabled here (the ranker)
REF_PASSAGES = 8
REF_CHARS = 700


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")] if path.exists() else []


def key_of(r: dict) -> str:
    return f"{r['qid']}|{r['design']}|{r['rep']}"


def plain(answer: str) -> str:
    """The answer as a reader sees it: refs and bold marks out."""
    import citations
    return re.sub(r"\s+([.!?,])", r"\1", citations.REF.sub("", answer)).replace("**", "").strip()


def items_of(r: dict) -> list[dict]:
    seen, out = set(), []
    for e in r["evidence"]:
        for i in e["items"]:
            if i["ref"] not in seen:
                seen.add(i["ref"])
                out.append(i)
    return out


# ---- contract ---------------------------------------------------------------
def grade_contract() -> None:
    import citations
    rows = []
    for r in load(ANSWERS):
        c = citations.check(r["answer"], items_of(r)) if r["answer_type"] != "empty" else \
            {"answer_type": "empty", "problems": [], "cited_share": None, "claims": []}
        rows.append({"key": key_of(r), "answer_type": c["answer_type"],
                     "cited_share": c["cited_share"], "n_claims": len(c["claims"]),
                     "problems": [p["kind"] for p in c["problems"]]})
    CONTRACT.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
    print(f"contract: {len(rows)} answers -> {CONTRACT.name}")


# ---- grounding --------------------------------------------------------------
class Grounding:
    def __init__(self):
        import google.auth
        import google.auth.transport.requests
        import requests
        self.creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        self.req = google.auth.transport.requests.Request()
        self.http = requests.Session()
        self.url = (f"https://discoveryengine.googleapis.com/v1/projects/{PROJECT}/locations/"
                    "global/groundingConfigs/default_grounding_config:check")

    def check(self, answer: str, facts: list[dict]) -> dict:
        if not self.creds.valid:
            self.creds.refresh(self.req)
        for attempt in range(8):
            r = self.http.post(self.url, timeout=60, json={
                "answerCandidate": answer, "facts": facts,
                "groundingSpec": {"citationThreshold": 0.6, "enableClaimLevelScore": True}},
                headers={"Authorization": f"Bearer {self.creds.token}",
                         "x-goog-user-project": PROJECT})
            if r.status_code in (429, 500, 503) and attempt < 7:   # quota is per minute
                time.sleep(min(2 ** attempt, 60))
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError("unreachable")


def _fact(i: dict) -> dict:
    when = f"Published {i['date']}, {i['days_ago']} days before today"
    return {"factText": f"{when}. Source: {i['source']}. {i['text']}"[:9_900],
            "attributes": {"ref": i["ref"]}}


def _overlap_refs(our_claims: list[dict], api_claim: dict, text: str) -> list[str]:
    """The model's refs on the sentence(s) an API claim falls in."""
    span = text[api_claim.get("startPos", 0):api_claim.get("endPos", 0)]
    refs = []
    for c in our_claims:
        if c["text"] and (span.strip(" .") in c["text"] or c["text"].strip(" .") in span):
            refs += c["refs"]
    return refs


def grade_grounding(workers: int = 2) -> None:
    import citations
    done = {g["key"] for g in load(GROUNDING)}
    todo = [r for r in load(ANSWERS) if key_of(r) not in done]
    g = Grounding()
    print(f"grounding: {len(todo)} answers to check ({len(done)} done)", flush=True)

    def one(r):
        items = items_of(r)
        text = plain(r["answer"])
        out = {"key": key_of(r)}
        if not items or r["answer_type"] in ("empty",) or not text:
            return out | {"skipped": "no evidence or empty answer"}
        resp = g.check(text, [_fact(i) for i in items])
        ours = citations.check(r["answer"], items)["claims"]
        for c in ours:
            c["text"] = plain(c["text"])
        # citationIndices point into resp["citedChunks"], NOT into the facts sent;
        # each cited chunk's `source` is the index of the fact it came from.
        chunks = resp.get("citedChunks", [])
        fact_of = [int(ch["source"]) for ch in chunks if str(ch.get("source", "")).isdigit()]
        claims, agree, checked = [], 0, 0
        for c in resp.get("claims", []):
            if not c.get("groundingCheckRequired", True):
                continue
            api_refs = list(dict.fromkeys(
                items[fact_of[k]]["ref"] for k in c.get("citationIndices", [])
                if k < len(fact_of) and fact_of[k] < len(items)))
            mine = _overlap_refs(ours, c, text)
            if mine and api_refs:
                checked += 1
                agree += bool(set(mine) & set(api_refs))
            claims.append({"text": c.get("claimText", ""), "score": c.get("score"),
                           "api_refs": api_refs, "model_refs": mine})
        return out | {"support": resp.get("supportScore"), "claims": claims,
                      "citation_checked": checked, "citation_agree": agree}

    def safe(r):
        try:
            return one(r)
        except Exception as e:  # noqa: BLE001 -- not written, so the next run retries it
            print(f"  skip {key_of(r)}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            return None

    with ThreadPoolExecutor(max_workers=workers) as ex, GROUNDING.open("a", encoding="utf-8") as f:
        for n, res in enumerate(ex.map(safe, todo), 1):
            if res is not None:
                f.write(json.dumps(res) + "\n")
            if n % 50 == 0:
                print(f"  {n}/{len(todo)}", flush=True)
    print(f"grounding -> {GROUNDING.name}")


# ---- judge ------------------------------------------------------------------
CRITERIA = {
    "correct": (
        "The response answers the user's question correctly according to the reference "
        "passages, which are the archive's relevant news, each dated, newest first, as of "
        "'today' in the prompt. When the references disagree, the NEWEST one is the truth: "
        "a response built on an older note that a newer one superseded is wrong on that point. "
        "If the reference says the archive has NO answer, the correct response says it found "
        "nothing and does not treat the question's premise as true. "
        "IMPORTANT: the reference is a sample of the relevant news, not all of it. The "
        "response may use other archive passages, so a detail that is merely ABSENT from the "
        "reference (a game date, a stat) is not an error; only a detail that CONTRADICTS the "
        "reference is. Whether every detail is supported is checked separately. Judge the "
        "core answer to the question."),
}
RUBRIC = {
    "2": "Correct: the core answer matches the newest reference passages (or, when the "
         "reference says there is no answer, the response says nothing was found).",
    "1": "Partly correct: right direction but misses the newest status or the main point, or "
         "a secondary detail contradicts the reference, or it hedges where the reference is clear.",
    "0": "Wrong: the core answer contradicts the newest references, or it answers a false "
         "premise as if true, or it says nothing was found when the reference has the answer.",
}


def _references(qs: dict, qrels: dict) -> dict:
    """qid -> reference text: the key's grade-2 passages (grade 1 when none), newest first."""
    from common import load_window
    df = load_window().set_index("doc_id")
    refs = {}
    for qid, q in qs.items():
        if q.get("kind") == "unanswerable":
            refs[qid] = (f"NONE. The archive has nothing that answers this question "
                         f"({q.get('why', 'not covered')}). A correct response says no archived "
                         "news was found.")
            continue
        grades = qrels.get(qid, {})
        best = [d for d, gr in grades.items() if gr == 2 and d in df.index] or \
               [d for d, gr in grades.items() if gr == 1 and d in df.index]
        rows = df.loc[best].sort_values("published", ascending=False).head(REF_PASSAGES)
        refs[qid] = "\n\n".join(
            f"[{p:%a %b %d}] {' '.join(str(t).split())[:REF_CHARS]}"
            for p, t in zip(rows.published, rows.text)) or None
    return refs


def grade_judge() -> None:
    import pandas as pd
    import vertexai
    from eval import load_qrels
    from vertexai.evaluation import EvalTask, PointwiseMetric, PointwiseMetricPromptTemplate

    from run import questions
    qs = {q["qid"]: q for q in questions()}
    _, qrels = load_qrels(approved_only=False)
    refs = _references(qs, qrels)
    done = {j["key"] for j in load(JUDGE)}
    rows = [r for r in load(ANSWERS) if key_of(r) not in done and refs.get(r["qid"])]
    print(f"judge: {len(rows)} answers to grade ({len(done)} done)", flush=True)
    if not rows:
        return
    vertexai.init(project=PROJECT, location="us-central1")
    metric = PointwiseMetric(metric="correct", metric_prompt_template=PointwiseMetricPromptTemplate(
        criteria=CRITERIA, rating_rubric=RUBRIC, input_variables=["prompt", "reference"]))
    with JUDGE.open("a", encoding="utf-8") as f:
        for start in range(0, len(rows), 100):          # batches: progress + resumability
            batch = rows[start:start + 100]
            df = pd.DataFrame({
                "prompt": [f"Today is Mon Oct 26, 2026. Question: {r['question']}"
                           + (f" (about {r['player']})" if r["player"] else "") for r in batch],
                "response": [plain(r["answer"]) for r in batch],
                "reference": [refs[r["qid"]] for r in batch]})
            res = EvalTask(dataset=df, metrics=[metric]).evaluate()
            t = res.metrics_table
            for r, score, why in zip(batch, t["correct/score"], t["correct/explanation"]):
                f.write(json.dumps({"key": key_of(r), "score": None if pd.isna(score) else float(score),
                                    "explanation": str(why)[:600]}) + "\n")
            f.flush()
            print(f"  {start + len(batch)}/{len(rows)}", flush=True)
    print(f"judge -> {JUDGE.name}")


# ---- report -----------------------------------------------------------------
def report() -> None:
    import numpy as np
    answers = {key_of(r): r for r in load(ANSWERS)}
    con = {g["key"]: g for g in load(CONTRACT)}
    gro = {g["key"]: g for g in load(GROUNDING)}
    jud = {g["key"]: g for g in load(JUDGE)}

    def pct(xs, q):
        return round(float(np.percentile(xs, q)), 2) if xs else None

    out = {}
    for design in ("agent", "pipeline"):
        rs = [r for k, r in answers.items() if r["design"] == design]
        ans = [r for r in rs if r["kind"] != "unanswerable"]
        una = [r for r in rs if r["kind"] == "unanswerable"]
        types_ = lambda group: {t: sum(con[key_of(r)]["answer_type"] == t for r in group)  # noqa: E731
                                for t in ("answered", "partial", "abstained", "unsupported", "empty")}
        probs = [p for r in rs for p in con[key_of(r)]["problems"]]
        sup = [gro[key_of(r)]["support"] for r in ans
               if gro.get(key_of(r), {}).get("support") is not None]
        ca = sum(gro.get(key_of(r), {}).get("citation_agree", 0) for r in rs)
        cc = sum(gro.get(key_of(r), {}).get("citation_checked", 0) for r in rs)
        js = lambda group: [jud[key_of(r)]["score"] for r in group  # noqa: E731
                            if jud.get(key_of(r), {}).get("score") is not None]
        # Stability: same question, two runs -- same answer type?
        by_q = {}
        for r in rs:
            by_q.setdefault(r["qid"], []).append(con[key_of(r)]["answer_type"])
        pairs = [v for v in by_q.values() if len(v) >= 2]
        lat = [r["elapsed_s"] for r in rs]
        out[design] = {
            "answers": len(rs),
            "correct_mean_0_2": round(float(np.mean(js(ans))), 3) if js(ans) else None,
            "correct_share_2": round(float(np.mean([s == 2 for s in js(ans)])), 3) if js(ans) else None,
            "refusal_mean_0_2": round(float(np.mean(js(una))), 3) if js(una) else None,
            "grounding_support_mean": round(float(np.mean(sup)), 3) if sup else None,
            "grounding_support_below_0.5": round(float(np.mean([s < 0.5 for s in sup])), 3) if sup else None,
            "citation_agreement": round(ca / cc, 3) if cc else None,
            "answer_types_answerable": types_(ans), "answer_types_unanswerable": types_(una),
            "problems": {k: probs.count(k) for k in sorted(set(probs))},
            "type_stable_across_runs": round(float(np.mean([len(set(v)) == 1 for v in pairs])), 3)
            if pairs else None,
            "latency_s": {"p50": pct(lat, 50), "p95": pct(lat, 95), "max": pct(lat, 100)},
            "cost_usd_per_answer": round(float(np.mean([r["cost_usd"] for r in rs])), 5),
            "model_calls_tool_calls_mean": round(float(np.mean([r["tool_calls"] for r in rs])), 2),
            "tokens_in_mean": round(float(np.mean([r["tokens"]["input"] for r in rs]))),
        }
    # Paired comparison on correctness: same question, same rep.
    diffs = []
    for k, r in answers.items():
        if r["design"] != "agent" or r["kind"] == "unanswerable":
            continue
        k2 = f"{r['qid']}|pipeline|{r['rep']}"
        a, b = jud.get(k, {}).get("score"), jud.get(k2, {}).get("score")
        if a is not None and b is not None:
            diffs.append(b - a)
    if diffs:
        rng = np.random.default_rng(0)
        boots = [float(np.mean(rng.choice(diffs, len(diffs)))) for _ in range(5000)]
        out["pipeline_minus_agent_correct"] = {
            "mean": round(float(np.mean(diffs)), 3), "ci95": [round(float(np.percentile(boots, 2.5)), 3),
                                                              round(float(np.percentile(boots, 97.5)), 3)],
            "pairs": len(diffs), "pipeline_better": sum(d > 0 for d in diffs),
            "agent_better": sum(d < 0 for d in diffs)}
    RESULTS.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import logging
    logging.disable(logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["contract", "grounding", "judge", "report"])
    step = ap.parse_args().step
    {"contract": grade_contract, "grounding": grade_grounding, "judge": grade_judge,
     "report": report}[step]()
