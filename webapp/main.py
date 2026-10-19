"""The nfl-dfs web app: sign in, edit your lineup, read your latest report.

Multi-user step 2. One small Flask service on Cloud Run in front of the same
Firestore the 12:30 job writes to (src/user_store.py):

    browser ──Firebase Auth (email/password)──► ID token
    browser ──Authorization: Bearer <token>──► this app ──► users/{uid}

    GET  /             the page (static/index.html)
    GET  /api/me         your name, lineup, and latest report
    PUT  /api/lineup     replace your lineup  {"lineup": ["name", ...]}
    GET  /api/slate      this week's player pool, to click players from
    GET  /api/suggested  the optimizer's top lineups for this week

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

import os
import sys
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from slate import BigQuerySlate, SlateError  # noqa: E402
from user_store import FirestoreStore  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"

# DraftKings classic NFL is 9 slots. A little slack for a half-edited lineup,
# a hard cap so nobody stores a novel in their lineup field.
MAX_PLAYERS = 12
MAX_NAME = 60


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


def create_app(store=None, verify=verify_firebase_token, slate=None) -> Flask:
    """Build the app. Tests pass a MemoryStore, a fake verifier and a fake slate."""
    app = Flask(__name__, static_folder=None)
    app.config["store"] = store
    slate = slate if slate is not None else BigQuerySlate()

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

    @app.errorhandler(400)
    @app.errorhandler(401)
    @app.errorhandler(403)
    @app.errorhandler(404)
    @app.errorhandler(503)
    def json_error(e):
        return jsonify(error=e.description), e.code

    return app


app = create_app()

if __name__ == "__main__":
    # Local: python webapp/main.py, then http://localhost:8080
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)
