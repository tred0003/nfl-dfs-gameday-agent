"""Read the user's DFS lineup from a Google Sheet.

Why a sheet: the lineup used to be MY_LINEUP in gameday_agent/.env, which
`adk deploy` bakes into the deployment. Changing it meant a redeploy from a
laptop, so it did not change -- the deployed agent was still checking the 9/20
roster on 10/9. The user edits this sheet from a phone; the scheduled client
reads it on every run and hands it to the agent with the request.

Layout (Sheet1): header row `Slot | Player`, then one row per roster slot.
Only column B is used. Slots are there for the human editing it.

Auth is Application Default Credentials, so the same code works as the Cloud
Run job's service account (shared on the sheet as Viewer, nothing more) and as
a developer's own login. Plain REST rather than the Sheets client library: one
GET does not justify another dependency in the image.
"""

from __future__ import annotations

SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
VALUES_URL = "https://sheets.googleapis.com/v4/spreadsheets/{sheet_id}/values/A:B"

# DraftKings classic NFL: QB, 2 RB, 3 WR, TE, FLEX, DST.
EXPECTED_SLOTS = 9


class LineupError(RuntimeError):
    """The sheet could not be read, or it held no players."""


def parse_rows(rows: list[list[str]]) -> list[str]:
    """Player names from the sheet's rows. Header and blank rows are skipped."""
    names = []
    for row in rows:
        if len(row) < 2:
            continue
        slot, player = str(row[0]).strip(), str(row[1]).strip()
        if slot.lower() == "slot" or not player:
            continue
        names.append(player)
    return names


def read_lineup(sheet_id: str, credentials=None) -> list[str]:
    """Return the rostered player names, or raise LineupError.

    `credentials` exists for testing under an impersonated identity; in
    production it is None and ADC supplies the job's service account.
    """
    import google.auth
    from google.auth.transport.requests import AuthorizedSession

    if not sheet_id:
        raise LineupError("LINEUP_SHEET_ID is not set")
    try:
        creds = credentials or google.auth.default(scopes=SCOPES)[0]
        r = AuthorizedSession(creds).get(VALUES_URL.format(sheet_id=sheet_id), timeout=20)
    except Exception as e:  # noqa: BLE001 -- auth and transport fail many ways
        raise LineupError(f"could not reach the lineup sheet: {type(e).__name__}: {e}") from e
    if r.status_code != 200:
        # 403 here almost always means the sheet is not shared with the
        # identity running this. Say which, rather than just "forbidden".
        who = getattr(creds, "service_account_email", None) or "the current identity"
        raise LineupError(f"lineup sheet returned HTTP {r.status_code} for {who}: "
                          f"{r.text[:200]}")

    names = parse_rows(r.json().get("values", []))
    if not names:
        raise LineupError("the lineup sheet has no players in column B")
    return names
