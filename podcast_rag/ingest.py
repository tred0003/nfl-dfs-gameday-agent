"""Podcast transcripts -> chunks -> embeddings -> BigQuery, as a LlamaIndex IngestionPipeline.

    python podcast_rag/ingest.py              # build dfs.podcast_chunks + the staging table
    python podcast_rag/ingest.py --dry-run    # chunk and tag; embed and write nothing

Laptop-only, so its packages are not in requirements.txt (the images that
install that file never run it):
    pip install llama-index-core llama-index-embeddings-google-genai

Why LlamaIndex here and not for search: ingestion is new code, and it is the
part LlamaIndex is good at -- documents with metadata, a splitter, custom
transforms, embeddings, and a cache so a re-run only pays for new chunks
(news_rag/index.py does that cache by hand, in embeddings.parquet). The
SEARCH stays in gameday_agent/archive.py: the time filter inside
VECTOR_SEARCH, the pinned ranker, recency-first and the citations are measured
there, and a rewrite could only match them.

Why not LlamaIndex's BigQueryVectorStore for the write: it owns its schema
(node_id, text, metadata JSON, embedding) and dfs.news_chunks has another --
typed columns the archive filters on (published, player_keys, screen). So the
pipeline ends at embedded nodes and the load below writes news_chunks' schema.

The pipeline:
  Document     one per five-minute clip (transcribe.py). The headline -- show
               and episode title -- is metadata that IS embedded with the text
               (news chunks embed "headline. text" the same way); everything
               else is metadata the embedding never sees.
  splitter     ~320 tokens (~1,200 characters, news' CHUNK_CHARS), on sentences
  EntityTagger roster names in the text -> players / player_ids / player_keys;
               team nicknames + the tagged players' teams -> teams. Gemini
               wrote roster spellings during transcription, so exact matching
               is enough.
  OnTopic      drops chunks that name no NFL player or team: banter ("Maine's
               Umbrella Cover Museum"), and the non-NFL half of a mixed chapter
  embedding    text-embedding-005, RETRIEVAL_DOCUMENT -- the model and task
               type the archive's queries are embedded against

Writes:
  dfs.podcast_chunks             podcast rows only, rebuilt each run
  dfs.news_chunks_with_podcasts  news_chunks + podcast_chunks: the STAGING
                                 table the gate runs against (NEWS_TABLE=...).
                                 The live dfs.news_chunks is not touched until
                                 the gate says podcasts do not hurt search.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from llama_index.core import Document
from llama_index.core.ingestion import IngestionCache, IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, TransformComponent

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

# The injection screen's Gemini client reads these; the Cloud Run job sets
# them, a laptop shell may not. Same defaults as gameday_agent/eval.
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "advisor-agent-2026")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")

from gameday_agent.news import TEAMS  # noqa: E402
from news_rag.index import (  # noqa: E402
    BQ_PROJECT, EMBED_MODEL, EMBED_PROJECT, TABLE, player_keys, schema_fields, screen_texts,
)
from podcast_rag.feed import SHOWS, episodes  # noqa: E402
from podcast_rag.transcribe import LOOP_REPEATS, OUT as TRANSCRIPTS, ROSTER, DATA  # noqa: E402

PODCAST_TABLE = f"{BQ_PROJECT}.dfs.podcast_chunks"
STAGING_TABLE = f"{BQ_PROJECT}.dfs.news_chunks_with_podcasts"
CACHE_DIR = DATA / "ingest_cache"
SCREEN_CACHE = DATA / "screen_cache.json"
CHUNK_TOKENS, OVERLAP_TOKENS = 320, 40
# Every metadata key but the headline stays out of the embedded text.
HIDDEN = ["show", "episode_id", "published", "clip_start", "clip_end", "clip_chars", "label", "url",
          "players", "player_ids", "teams", "part"]


def documents() -> list[Document]:
    audio = {ep.episode_id: ep.audio_url
             for show in SHOWS for ep in episodes(show, limit=SHOWS[show]["last"] * 4)}
    docs, cut = [], 0
    for path in sorted(TRANSCRIPTS.glob("*.jsonl")):
        for r in map(json.loads, path.open(encoding="utf-8")):
            text = r["text"]
            if "repetition loop" in r["problems"]:
                text = before_loop(text)
                cut += 1
            if not text:
                continue
            docs.append(Document(
                id_=f"{r['episode_id']}@{r['start']}", text=text,
                metadata={"headline": f"{SHOWS[r['show']]['name']}: {r['title']}",
                          "show": r["show"], "episode_id": r["episode_id"],
                          "published": r["published"], "clip_start": r["start"],
                          "clip_end": r["end"], "clip_chars": len(text),
                          "label": r["label"],
                          "url": audio.get(r["episode_id"])},
                excluded_embed_metadata_keys=HIDDEN, excluded_llm_metadata_keys=HIDDEN,
                # "Headline. text" -- the shape news story chunks are embedded in
                metadata_template="{value}", text_template="{metadata_str}. {content}"))
    print(f"{cut} clips cut at a decoding loop")
    return docs


def before_loop(text: str) -> str:
    """The text before Gemini's decoding loop. 8 of 1,296 backfill clips came
    back as real speech that turns into "I I I I ..." or one sentence repeated
    a thousand times, even after transcribe.py's retry. The start is real;
    everything from the first copy of the looping 8 words on is dropped."""
    words = text.split()
    grams = Counter(" ".join(words[i:i + 8]) for i in range(len(words) - 7))
    gram, n = grams.most_common(1)[0] if grams else ("", 0)
    if n <= LOOP_REPEATS:
        return text
    first = next(i for i in range(len(words) - 7) if " ".join(words[i:i + 8]) == gram)
    return " ".join(words[:first])


def _norm(s: str) -> str:
    """predict_slate.normalize_name's rule, on running text: "Kyle Pitts Sr."
    and "Kyle Pitts" both become "kyle pitts"."""
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", s.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", s)).strip()


class EntityTagger(TransformComponent):
    """Players and teams named in each chunk, plus the chunk's index within
    its clip (part), assigned here so it survives OnTopic dropping chunks."""

    def __call__(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        roster = json.loads(ROSTER.read_text(encoding="utf-8"))
        by_key: dict[str, list[dict]] = defaultdict(list)
        for p in roster:
            by_key[_norm(p["name"])].append(p)
        name_re = re.compile(r"\b(" + "|".join(sorted(map(re.escape, by_key), key=len, reverse=True)) + r")\b")
        team_re = re.compile(r"\b(" + "|".join(t[1] for t in TEAMS.values()) + r")\b")
        abbr = {t[1]: a for a, t in TEAMS.items()}
        parts: dict[str, int] = defaultdict(int)
        for n in nodes:
            text = _norm(n.get_content())
            found = {k: by_key[k][0] for k in name_re.findall(text)}
            teams = {p["team"] for p in found.values()} | {
                abbr[m] for m in team_re.findall(n.get_content())}
            n.metadata.update(players=sorted(p["name"] for p in found.values()),
                              player_ids=sorted(str(p["espn_id"]) for p in found.values()),
                              teams=sorted(teams), part=parts[n.ref_doc_id])
            parts[n.ref_doc_id] += 1
            n.excluded_embed_metadata_keys = HIDDEN
            n.excluded_llm_metadata_keys = HIDDEN
        return nodes


class OnTopic(TransformComponent):
    dropped: int = 0

    def __call__(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        keep = [n for n in nodes if n.metadata["players"] or n.metadata["teams"]]
        self.dropped += len(nodes) - len(keep)
        return keep


def pipeline(embed: bool = True) -> IngestionPipeline:
    steps = [SentenceSplitter(chunk_size=CHUNK_TOKENS, chunk_overlap=OVERLAP_TOKENS),
             EntityTagger(), OnTopic()]
    if embed:
        from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
        from llama_index.embeddings.google_genai.base import VertexAIConfig
        steps.append(GoogleGenAIEmbedding(
            model_name=EMBED_MODEL, embed_batch_size=50,
            vertexai_config=VertexAIConfig(project=EMBED_PROJECT, location="us-central1")))
    cache = IngestionCache.from_persist_path(str(CACHE_DIR / "cache.json")) \
        if (CACHE_DIR / "cache.json").exists() else IngestionCache()
    return IngestionPipeline(transformations=steps, cache=cache)


def rows(nodes: list[BaseNode]) -> pd.DataFrame:
    out = []
    for n in nodes:
        m = n.metadata
        # Where in the episode this chunk starts: the clip start plus its share
        # of the clip by character position. Close enough to cue the audio.
        at = m["clip_start"] + int((m["clip_end"] - m["clip_start"])
                                   * (n.start_char_idx or 0) / m["clip_chars"])
        text = n.get_content(metadata_mode="embed")        # "Headline. text", as embedded
        out.append({
            "chunk_id": f"{m['episode_id']}@{m['clip_start']}#{m['part']}",
            "article_id": m["episode_id"], "chunk_index": m["part"], "kind": "podcast",
            "source": SHOWS[m["show"]]["name"], "published": m["published"],
            "headline": m["headline"], "text": text,
            "players": m["players"], "player_ids": m["player_ids"], "teams": m["teams"],
            "url": f"{m['url']}#t={at}" if m["url"] else None,
            "text_hash": hashlib.sha1(text.encode()).hexdigest(),
            "embedding": [float(x) for x in n.embedding] if n.embedding else None,
        })
    df = pd.DataFrame(out)
    df["published"] = pd.to_datetime(df["published"], utc=True, format="ISO8601")
    df["player_keys"] = df["players"].map(player_keys)
    return df


def screen(df: pd.DataFrame) -> list[str]:
    """The archive's injection screen, cached on text hash: only "clean" rows
    are ever retrieved, and a re-run should not pay to re-screen."""
    cache = json.loads(SCREEN_CACHE.read_text()) if SCREEN_CACHE.exists() else {}
    todo = df.loc[~df["text_hash"].isin(cache)]
    if len(todo):
        cache.update(zip(todo["text_hash"], screen_texts(todo["text"].tolist())))
        # "unchecked" means the screen call failed: retried on the next run
        SCREEN_CACHE.write_text(json.dumps({h: v for h, v in cache.items() if v != "unchecked"}))
    return df["text_hash"].map(cache).tolist()


def load(df: pd.DataFrame) -> None:
    from google.cloud import bigquery
    bq = bigquery.Client(project=BQ_PROJECT)
    schema = schema_fields()
    cfg = bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_TRUNCATE",
                                 time_partitioning=bigquery.TimePartitioning(field="published"))
    bq.load_table_from_dataframe(df[[f.name for f in schema]], PODCAST_TABLE, job_config=cfg).result()
    cols = ", ".join(f.name for f in schema)
    bq.query_and_wait(f"""
        CREATE OR REPLACE TABLE `{STAGING_TABLE}` PARTITION BY DATE(published) AS
        SELECT {cols} FROM `{TABLE}` UNION ALL SELECT {cols} FROM `{PODCAST_TABLE}`""")
    print(f"loaded {len(df)} chunks into {PODCAST_TABLE}; staging table {STAGING_TABLE} rebuilt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    docs = documents()
    pipe = pipeline(embed=not args.dry_run)
    nodes = pipe.run(documents=docs, show_progress=False)
    if not args.dry_run:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        pipe.cache.persist(str(CACHE_DIR / "cache.json"))
    df = rows(nodes)
    eps = df["article_id"].nunique()
    dropped = sum(t.dropped for t in pipe.transformations if isinstance(t, OnTopic))
    print(f"{len(docs)} clips from {eps} episodes -> {len(df)} on-topic chunks "
          f"({dropped} off-topic dropped this run; cached runs skip the count); "
          f"{(df['players'].str.len() > 0).mean():.0%} name a player, "
          f"median {df['text'].str.len().median():.0f} chars")
    print(df.groupby("source").size().to_string())
    if args.dry_run:
        return
    df["screen"] = screen(df)
    print(df["screen"].value_counts().to_string())
    load(df)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
