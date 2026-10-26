"""Shared pieces for the embedding-tuning experiment: the frozen snapshot, the
search window, the player filter and the query embedder.

Standalone on purpose: news_rag.index imports the agent package (and with it
the ADK), which nothing here needs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from predict_slate import normalize_name  # noqa: E402

DATA = ROOT / "data" / "embed_tune"         # big, rebuildable: the snapshot (snapshot.py)
KEY = ROOT / "embed_tune" / "key"           # small, committed: the answer key and scores
SNAPSHOT = DATA / "snapshot.parquet"
QUESTIONS = ROOT / "embed_tune" / "questions.jsonl"

CUTOFF = pd.Timestamp("2026-10-26", tz="UTC")
WINDOW_DAYS = 42                      # gameday_agent/archive.py SEARCH_DAYS
EMBED_PROJECT = "advisor-agent-2026"
BASE_MODEL = "text-embedding-005"     # what the archive is embedded with today


def load_window() -> pd.DataFrame:
    """The snapshot inside the production search window, one row per passage.

    A republished article keeps its chunk_ids (blurb-40699880#0 appears on
    10/21 and again on 10/23, 174 of them with changed text), so a passage is
    keyed by doc_id = chunk_id@published. Exact-text repeats keep only the
    newest copy, as the archive search does."""
    df = pd.read_parquet(SNAPSHOT)
    df = df[df.published >= CUTOFF - pd.Timedelta(days=WINDOW_DAYS)]
    df = df.sort_values("published", ascending=False).drop_duplicates("text")
    df["doc_id"] = df.chunk_id + "@" + df.published.dt.strftime("%Y-%m-%dT%H:%M")
    return df.reset_index(drop=True)


def unit(m) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def name_mask(df: pd.DataFrame, player: str) -> np.ndarray:
    """The production player filter (gameday_agent/archive.py PLAYER_FILTER):
    the normalized name is tagged on the chunk, or the full name is in the text."""
    key = normalize_name(pd.Series([player], dtype=object)).iloc[0]
    tagged = df.player_keys.map(lambda ks: ks is not None and key in list(ks))
    named = df.text.str.lower().str.contains(player.lower(), regex=False)
    return (tagged | named).to_numpy()


def genai_client():
    from google import genai
    from google.genai import types
    retry = types.HttpRetryOptions(attempts=6, initial_delay=2.0, max_delay=60.0,
                                   exp_base=2.0, http_status_codes=[429, 500, 503, 504])
    return genai.Client(vertexai=True, project=EMBED_PROJECT, location="us-central1",
                        http_options=types.HttpOptions(retry_options=retry))


def embed_queries(texts: list[str], model: str = BASE_MODEL, client=None) -> np.ndarray:
    """Unit-length query embeddings, RETRIEVAL_QUERY task type -- the same call
    the archive search makes for a question."""
    from google.genai import types
    client = client or genai_client()
    out = []
    for i in range(0, len(texts), 100):
        resp = client.models.embed_content(
            model=model, contents=texts[i:i + 100],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"))
        out += [e.values for e in resp.embeddings]
    return unit(out)
