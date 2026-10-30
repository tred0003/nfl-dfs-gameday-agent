"""The nfl-dfs web app: sign in, edit your lineup, read your latest report.

Multi-user step 2. One small Flask service on Cloud Run in front of the same
Firestore the 12:30 job writes to (src/user_store.py):

    browser ──Firebase Auth (email/password)──► ID token
    browser ──Authorization: Bearer <token>──► this app ──► users/{uid}

    GET  /             the page (static/index.html)
    GET  /api/me         your name, lineup, and latest report
    PUT  /api/lineup     replace your lineup  {"lineup": ["name", ...]}
    POST /api/lineup/import  a DK screenshot (raw image body) -> the slate players in it;
                            saves nothing (webapp/lineup_import.py, Document AI OCR)
    GET  /api/slate      this week's player pool, to click players from
    GET  /api/suggested  the optimizer's top lineups for this week
    POST /api/news/history  four weeks of archived notes on named players
    POST /api/news/search   a question, answered by the retriever alone (ranked passages);
                            "source": "podcasts" searches the podcast transcripts instead
    POST /api/news/ask      a question, answered from the archive with cited passages
    POST /api/news/feedback thumbs up/down and "wrong citation" on one of YOUR answers
    GET  /news              the News desk page (static/news.html)

All news routes go through the news-mcp service: history and search are
direct MCP calls (webapp/news.py), ask is a Gemini agent whose tools come
from news-mcp (webapp/ask.py) -- the same tools Claude Code connects to.
Search and ask run the same retriever, so a bad answer can be traced: did
search miss the passage, or did the model misread it?

The last two are the same for every user and cached per projection run
(webapp/slate.py) -- they cost one query and one solve per run, not per user.

WHO YOU ARE COMES FROM THE TOKEN, NEVER FROM THE REQUEST. current_uid()
verifies the Firebase ID token's signature, expiry and audience with the
Admin SDK and takes the uid out of it. No route reads a uid from the URL, the
query string or the body -- a request that sends {"uid": "u1"} is just
ignored. That is the whole isolation story: user2 has no way to even NAME
user1's data, so there is nothing to forge.

The browser never talks to Firestore directly. The database's security rules
deny all client access (deploy/firestore.rules); every read and write goes
through this server, as its own service account, after the token check.

Signed in but not provisioned (a Firebase account with no users/{uid}
document) is a 403, not an empty lineup -- same "no data is not all clear"
rule as the report job.
"""

from __future__ import annotations

import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ask import ArchiveAgent, AskError  # noqa: E402
from news import McpNews, NewsError  # noqa: E402
from lineup_import import MAX_BYTES as MAX_UPLOAD  # noqa: E402
from lineup_import import DocAiReader, LineupImportError, check_upload, match_lineup  # noqa: E402
from slate import BigQuerySlate, SlateError  # noqa: E402
from user_store import FirestoreStore  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"

# DraftKings classic NFL is 9 slots. A little slack for a half-edited lineup,
# a hard cap so nobody stores a novel in their lineup field.
MAX_PLAYERS = 12
MAX_NAME = 60
MAX_QUESTION = 300
# Questions per user per day. Each is a few Gemini calls (~half a cent), so
# this is a cost guard, not a product limit. Counted per instance in memory:
# gameday-web runs at most 2, so the real ceiling is at most twice this.
ASK_LIMIT = 20
# Searches per user per day. One embedding call and two BigQuery queries, no
# model -- far cheaper than a question, so a looser guard of the same kind.
SEARCH_LIMIT = 100
# Screenshot imports per user per day. One Document AI OCR page each (about
# $0.0015), so this guards against a loop in the page, not against cost.
IMPORT_LIMIT = 20


def verify_firebase_token(token: str) -> str:
    """The uid inside a valid Firebase ID token. Raises on anything invalid.

    Lazy import: the unit tests swap this function out and run without the
    Admin SDK installed. In production firebase_admin checks the signature
    against Google's public keys, the expiry, and that the token was issued
    for THIS project -- a token from some other Firebase project is rejected.
    """
    import firebase_admin
    from firebase_admin import auth

    if not firebase_admin._apps:  # noqa: SLF001 -- initialise once per process
        firebase_admin.initialize_app(options={
            "projectId": os.environ.get("GOOGLE_CLOUD_PROJECT", "advisor-agent-2026")})
    return auth.verify_id_token(token)["uid"]


MAX_NOTE = 500


def ask_record(question: str, player: str, out: dict) -> dict:
    """What is kept of one answer: enough to audit it and to re-grade it later
    (the passage ids, not their text -- the archive has the text)."""
    cited = sorted({r for c in out.get("claims", []) for r in c.get("refs", [])},
                   key=lambda r: int(r[1:]))
    return {"question": question, "player": player, "answer": out.get("answer", ""),
            "answer_type": out.get("answer_type"), "cited_share": out.get("cited_share"),
            "problems": [p.get("kind") for p in out.get("problems", [])],
            "cited_refs": cited,
            "passages": [{"ref": i.get("ref"), "id": i.get("id"), "date": i.get("date")}
                         for e in out.get("evidence", []) for i in e.get("items", [])],
            "versions": out.get("versions", {}), "cost_usd": out.get("cost_usd"),
            "elapsed_s": out.get("elapsed_s")}


def create_app(store=None, verify=verify_firebase_token, slate=None, news=None,
               agent=None, reader=None) -> Flask:
    """Build the app. Tests pass a MemoryStore, a fake verifier and fakes for the rest."""
    app = Flask(__name__, static_folder=None)
    app.config["store"] = store
    slate = slate if slate is not None else BigQuerySlate()
    news = news if news is not None else McpNews()
    agent = agent if agent is not None else ArchiveAgent(slate=slate)
    reader = reader if reader is not None else DocAiReader()
    def spend(uid: str, route: str, limit: int, what: str) -> None:
        """Count one call against a user's daily limit for this route, or 429.
        In Firestore (users/{uid}/limits/{day}), in a transaction: the count
        holds across instances and restarts, which an in-memory one did not."""
        today = datetime.now(timezone.utc).date().isoformat()
        if not get_store().spend(uid, route, today, limit):
            abort(429, f"{limit} {what} a day -- the limit resets at midnight UTC")

    def read_question() -> tuple[str, str]:
        body = request.get_json(silent=True) or {}
        question = body.get("question", "")
        player = body.get("player", "")
        if not isinstance(question, str) or not isinstance(player, str):
            abort(400, "question and player must be text")
        question, player = question.strip(), player.strip()
        if not question:
            abort(400, "ask a question")
        if len(question) > MAX_QUESTION or len(player) > MAX_NAME:
            abort(400, f"questions up to {MAX_QUESTION} characters, names up to {MAX_NAME}")
        return question, player

    def get_store():
        # Created on first use, not at import: a missing credential should be
        # a 500 on the first request with a clear log line, not a crash loop.
        if app.config["store"] is None:
            app.config["store"] = FirestoreStore()
        return app.config["store"]

    def current_uid() -> str:
        header = request.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            abort(401, "sign in first")
        try:
            return verify(header.removeprefix("Bearer ").strip())
        except Exception:  # noqa: BLE001 -- bad, expired, wrong project: all 401
            abort(401, "sign-in expired or invalid -- sign in again")

    def current_user():
        uid = current_uid()
        user = get_store().get_user(uid)
        if user is None:
            abort(403, "this account has no lineup set up")
        return user

    @app.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.get("/news")
    def news_page():
        return send_from_directory(STATIC, "news.html")

    @app.get("/static/<path:name>")
    def static_files(name):
        return send_from_directory(STATIC, name)

    @app.get("/api/config")
    def config():
        # The Firebase WEB config. Not secrets: the apiKey only identifies the
        # project to Firebase Auth, and every browser that loads the page sees
        # it. Served from env so the page has no project baked into it.
        return jsonify(apiKey=os.environ.get("FIREBASE_API_KEY", ""),
                       authDomain=os.environ.get("FIREBASE_AUTH_DOMAIN", ""),
                       projectId=os.environ.get("GOOGLE_CLOUD_PROJECT",
                                                "advisor-agent-2026"))

    @app.get("/api/me")
    def me():
        user = current_user()
        latest = get_store().latest_report(user.uid)
        report = None
        if latest:
            rid, fields = latest
            report = {"date": rid, "text": fields.get("text", ""),
                      "source": fields.get("source", ""),
                      "problems": fields.get("problems", [])}
        return jsonify(uid=user.uid, name=user.name, email=user.email,
                       lineup=user.lineup, report=report)

    @app.put("/api/lineup")
    def put_lineup():
        user = current_user()
        body = request.get_json(silent=True) or {}
        lineup = body.get("lineup")
        if not isinstance(lineup, list) or not all(isinstance(n, str) for n in lineup):
            abort(400, "lineup must be a list of player names")
        names = [n.strip() for n in lineup if n.strip()]
        if not names:
            # Same rule as the job: an empty lineup reads as "all clear".
            abort(400, "a lineup needs at least one player")
        if len(names) > MAX_PLAYERS or any(len(n) > MAX_NAME for n in names):
            abort(400, f"at most {MAX_PLAYERS} players, {MAX_NAME} characters each")
        get_store().set_lineup(user.uid, names)
        return jsonify(uid=user.uid, lineup=names)

    # The body is the image itself, Content-Type its media type. Checked
    # (type, size, magic bytes) before it is sent anywhere, and the result is
    # only a suggestion: the page fills the slots, the user saves.
    @app.post("/api/lineup/import")
    def import_lineup():
        user = current_user()
        if (request.content_length or 0) > MAX_UPLOAD:
            abort(413, f"the file is over {MAX_UPLOAD // (1024 * 1024)} MB")
        data = request.get_data(cache=False)
        try:
            mime = check_upload(data, request.content_type)
        except LineupImportError as e:
            abort(400, str(e))
        spend(user.uid, "import", IMPORT_LIMIT, "screenshot imports")
        try:
            pool = slate.pool()
        except SlateError as e:
            abort(503, f"this week's players could not be loaded ({e})")
        try:
            lines = reader.lines(data, mime)
        except LineupImportError as e:
            abort(503, f"the screenshot could not be read ({e})")
        out = match_lineup(lines, pool["players"])
        print(json.dumps({"event": "lineup_import", "uid": user.uid, "lines": len(lines),
                          "matched": len(out["players"]),
                          "not_recognised": len(out["not_recognised"]),
                          "warnings": len(out["warnings"])}), flush=True)
        return jsonify(week=pool.get("week"), **out)

    # Signed-in only: the pool is not secret, but there is no reason to run
    # BigQuery for anonymous traffic. current_user(), not current_uid(), so an
    # unprovisioned account gets the same 403 here as everywhere else.
    @app.get("/api/slate")
    def get_slate():
        current_user()
        try:
            return jsonify(slate.pool())
        except SlateError as e:
            abort(503, f"this week's players could not be loaded ({e})")

    @app.get("/api/suggested")
    def get_suggested():
        current_user()
        try:
            return jsonify(slate.suggested())
        except SlateError as e:
            abort(503, f"suggested lineups could not be built ({e})")

    # Signed-in only, for the same reason as the slate: no anonymous traffic
    # reaches BigQuery or the embedding model through this app.
    @app.post("/api/news/history")
    def news_history():
        user = current_user()
        body = request.get_json(silent=True) or {}
        players = body.get("players", user.lineup)
        if not isinstance(players, list) or not all(isinstance(n, str) for n in players):
            abort(400, "players must be a list of names")
        names = [n.strip() for n in players if n.strip()]
        if not names:
            abort(400, "name at least one player")
        if len(names) > MAX_PLAYERS or any(len(n) > MAX_NAME for n in names):
            abort(400, f"at most {MAX_PLAYERS} players, {MAX_NAME} characters each")
        try:
            return jsonify(news.history(names))
        except NewsError as e:
            abort(503, f"the news archive could not be read ({e})")

    @app.post("/api/news/search")
    def news_search():
        user = current_user()
        question, player = read_question()
        source = (request.get_json(silent=True) or {}).get("source", "news")
        if source not in ("news", "podcasts"):
            abort(400, 'source is "news" or "podcasts"')
        spend(user.uid, "search", SEARCH_LIMIT, "searches")
        try:
            return jsonify(news.search(question, player, source))
        except NewsError as e:
            abort(503, f"the news archive could not be searched ({e})")

    @app.post("/api/news/ask")
    def news_ask():
        user = current_user()
        question, player = read_question()
        spend(user.uid, "ask", ASK_LIMIT, "questions")
        try:
            out = agent.ask(question, player, user.lineup, uid=user.uid)
        except AskError as e:
            abort(503, f"the question could not be answered ({e})")
        # Kept under the user: the audit trail, and what feedback attaches to.
        # A failed write costs the record, never the answer.
        out["ask_id"] = uuid.uuid4().hex[:16]
        try:
            get_store().save_ask(user.uid, out["ask_id"], ask_record(question, player, out))
        except Exception as e:  # noqa: BLE001
            print(json.dumps({"event": "ask_save_failed", "uid": user.uid,
                              "error": f"{type(e).__name__}: {str(e)[:150]}"}), flush=True)
            out["ask_id"] = None
        return jsonify(out)

    @app.post("/api/news/feedback")
    def news_feedback():
        user = current_user()
        body = request.get_json(silent=True) or {}
        ask_id, rating = body.get("ask_id"), body.get("rating")
        wrong = body.get("wrong_refs", [])
        note = body.get("note", "")
        if not isinstance(ask_id, str) or not re.fullmatch(r"[0-9a-f]{16}", ask_id):
            abort(400, "which answer? (ask_id)")
        if rating not in ("up", "down"):
            abort(400, "rating is up or down")
        if (not isinstance(wrong, list) or len(wrong) > 20
                or not all(isinstance(r, str) and re.fullmatch(r"P\d{1,3}", r) for r in wrong)):
            abort(400, "wrong_refs is a list like [\"P2\"]")
        if not isinstance(note, str) or len(note) > MAX_NOTE:
            abort(400, f"notes up to {MAX_NOTE} characters")
        # Looked up under the token's uid: another user's answer is a 404,
        # the same as one that never existed.
        record = get_store().get_ask(user.uid, ask_id)
        if record is None:
            abort(404, "no such answer")
        known = {c for c in record.get("cited_refs", [])}
        get_store().set_ask_feedback(user.uid, ask_id, {
            # A thumbs-up with a "wrong" citation contradicts itself: drop them.
            "rating": rating, "wrong_refs": [r for r in wrong if r in known] if rating == "down" else [],
            "note": note.strip()})
        return jsonify(ok=True)

    @app.errorhandler(400)
    @app.errorhandler(401)
    @app.errorhandler(403)
    @app.errorhandler(404)
    @app.errorhandler(413)
    @app.errorhandler(429)
    @app.errorhandler(503)
    def json_error(e):
        return jsonify(error=e.description), e.code

    return app


app = create_app()

if __name__ == "__main__":
    # Local: python webapp/main.py, then http://localhost:8080
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)
