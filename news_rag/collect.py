"""Collect the 2026 NFL news archive the RAG layer retrieves from.

    python news_rag/collect.py                 # since 2026-07-15 (training camp)
    python news_rag/collect.py --since 2026-08-01

Two sources, both ESPN, both checked by hand on 2026-10-14 for how far back
they really go -- the feeds the gameday agent reads live (team-site RSS, PFT,
Yahoo, RotoWire RSS) hold days to a few weeks, so none of them can backfill:

  player blurbs   site.api.espn.com/apis/fantasy/v2/games/ffl/news/players
                  RotoWire-written fantasy notes, one feed per player, back to
                  the start of training camp. Short, dense, the DFS signal.
  league stories  now.core.api.espn.com/v1/sports/news?leagues=nfl
                  Full ESPN articles, paged by offset back past May. The
                  `leagues=nfl` filter leaks -- college football, the AFL's
                  Brisbane Lions -- so only items tagged league "NFL" are kept.

Writes data/news/articles.jsonl, one record per article, de-duplicated on
ESPN's id. data/ is gitignored: ESPN's terms make this personal research use.

`published` is the field everything downstream depends on. Retrieval for a
past week only sees articles published before that week's kickoff -- without
that, a "what would we have known" test reads the game recap and predicts
what already happened.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gameday_agent.news import TEAMS  # noqa: E402

OUT = ROOT / "data" / "news" / "articles.jsonl"
SKILL = {"QB", "RB", "WR", "TE", "FB"}
ROSTER = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{id}/roster"
PLAYER_NEWS = "https://site.api.espn.com/apis/fantasy/v2/games/ffl/news/players?playerId={id}&limit=200"
LEAGUE_NEWS = "https://now.core.api.espn.com/v1/sports/news?leagues=nfl&limit=50&offset={offset}"
PAGE = 50

# No User-Agent header, on purpose: site.api.espn.com 403s a hand-set one
# and serves requests' default (see the dfs-data-source-limits notes).
_session = requests.Session()


def _get(url: str) -> dict:
    r = _session.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def _plain(s: str | None) -> str:
    s = re.sub(r"<[^>]+>", " ", s or "")
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


def _when(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def skill_players() -> list[dict]:
    """Every QB/RB/WR/TE/FB on a 2026 roster, including IR and practice squad."""
    def one(abbr: str) -> list[dict]:
        espn_id = TEAMS[abbr][0]
        out = []
        for group in _get(ROSTER.format(id=espn_id))["athletes"]:
            for a in group["items"]:
                pos = a.get("position", {}).get("abbreviation")
                if pos in SKILL:
                    out.append({"espn_id": a["id"], "name": a["fullName"],
                                "position": pos, "team": abbr,
                                "roster_group": group.get("position")})
        return out

    with ThreadPoolExecutor(max_workers=8) as ex:
        return [p for team in ex.map(one, TEAMS) for p in team]


def player_blurbs(players: list[dict], since: datetime) -> tuple[list[dict], list[str]]:
    def one(p: dict) -> list[dict]:
        out = []
        for f in _get(PLAYER_NEWS.format(id=p["espn_id"])).get("feed", []):
            if not f.get("published") or _when(f["published"]) < since:
                continue
            url = (f.get("links") or {}).get("web", {}).get("href")
            out.append({
                "id": f"blurb-{f['id']}",
                "source": "ESPN fantasy (RotoWire)",
                # The feed also carries ESPN's multi-player fantasy columns,
                # up to 20K characters. They link to a /story/ page; the
                # RotoWire notes have no link. A column is a story: chunked,
                # and left out of get_player_history's per-player notes.
                "kind": "story" if "/story/" in (url or "") else "player_blurb",
                "published": f["published"],
                "headline": _plain(f.get("headline")),
                "text": _plain(f.get("story")) or _plain(f.get("description")),
                "players": [{"espn_id": p["espn_id"], "name": p["name"],
                             "position": p["position"]}],
                "teams": [p["team"]],
                "url": url,
            })
        return out

    rows, errors = [], []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(one, p): p for p in players}
        for fut, p in futs.items():
            try:
                rows.extend(fut.result())
            except Exception as e:  # noqa: BLE001 -- one bad player must not sink the run
                errors.append(f"{p['name']} ({p['espn_id']}): {type(e).__name__}")
    return rows, errors


def league_stories(since: datetime) -> tuple[list[dict], int]:
    """Page back through ESPN's NFL feed until it is clearly past `since`.

    Pages are not strictly ordered -- one page on 10/14 spanned 8/28 to 10/2 --
    so the walk stops after three consecutive pages with nothing new enough,
    not at the first old item.
    """
    abbr_by_espn = {v[0]: k for k, v in TEAMS.items()}
    rows, pages, stale, offset = [], 0, 0, 0
    while stale < 3:
        batch = [offset + i * PAGE for i in range(8)]
        with ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(lambda o: _get(LEAGUE_NEWS.format(offset=o)).get("headlines", []),
                                  batch))
        offset += 8 * PAGE
        for items in results:
            pages += 1
            fresh = [x for x in items if x.get("published") and _when(x["published"]) >= since]
            stale = 0 if fresh else stale + 1
            for x in fresh:
                cats = x.get("categories") or []
                if not any(c.get("type") == "league" and c.get("description") == "NFL" for c in cats):
                    continue
                text = _plain(x.get("story"))
                if not text:
                    continue
                rows.append({
                    "id": f"story-{x['id']}",
                    "source": "ESPN",
                    "kind": (x.get("type") or "Story").lower(),
                    "published": x["published"],
                    "headline": _plain(x.get("headline")),
                    "text": text,
                    "players": [{"espn_id": str(c.get("athleteId")), "name": c.get("description")}
                                for c in cats if c.get("type") == "athlete" and c.get("athleteId")],
                    "teams": sorted({abbr_by_espn[str(c["teamId"])] for c in cats
                                     if c.get("type") == "team" and str(c.get("teamId")) in abbr_by_espn}),
                    "url": (x.get("links") or {}).get("web", {}).get("href"),
                })
            if not items:
                stale = 3
    return rows, pages


class CollectError(RuntimeError):
    """Too much of the collection failed to call the result an archive update."""


# More player feeds failing than this is ESPN refusing us, not a few bad ids.
MAX_FEED_FAILURE = 0.25


def collect(since: datetime, log=print) -> list[dict]:
    """Every article published at or after `since`, de-duplicated on ESPN's id.

    Shared by the full rebuild below and the daily refresh (refresh.py).
    Raises CollectError when most of the per-player feeds fail: a refresh that
    appends 40 of the usual 200 chunks would look like a quiet news day.
    """
    players = skill_players()
    log(f"rosters: {len(players)} skill players on 32 teams")
    blurbs, errors = player_blurbs(players, since)
    log(f"player blurbs: {len(blurbs)} ({len(errors)} player feeds failed)")
    for e in errors[:10]:
        log(f"  failed: {e}")
    if players and len(errors) / len(players) > MAX_FEED_FAILURE:
        raise CollectError(f"{len(errors)} of {len(players)} player feeds failed")
    stories, pages = league_stories(since)
    log(f"league stories: {len(stories)} NFL stories from {pages} pages")

    # A blurb about two players (a QB and his receiver) sits in both players'
    # feeds under one id. Merge their player and team lists rather than keep
    # whichever feed came first, or a search for the second player misses it.
    by_id: dict[str, dict] = {}
    for r in blurbs + stories:
        if (kept := by_id.get(r["id"])) is None:
            by_id[r["id"]] = r
            continue
        ids = {p["espn_id"] for p in kept["players"]}
        kept["players"] += [p for p in r["players"] if p["espn_id"] not in ids]
        kept["teams"] = sorted(set(kept["teams"]) | set(r["teams"]))
    return sorted(by_id.values(), key=lambda r: r["published"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--since", default="2026-07-15", help="earliest publish date kept (UTC)")
    a = p.parse_args()
    since = datetime.fromisoformat(a.since).replace(tzinfo=timezone.utc)

    rows = collect(since)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} articles to {OUT.relative_to(ROOT)}"
          f"  ({rows[0]['published'][:10]} .. {rows[-1]['published'][:10]})" if rows else "nothing written")


if __name__ == "__main__":
    main()
