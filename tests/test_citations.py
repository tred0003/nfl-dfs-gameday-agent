"""Tests for the Ask answer's output contract (webapp/citations.py):
python -m unittest discover -s tests

No model and no network, so these run in CI. What they pin down:
  - every passage gets a ref, and the same passage keeps one ref across calls
  - refs are read off each sentence, before or after its period
  - a ref that was never retrieved is removed from the answer and flagged
  - an uncited fact is flagged; a sentence about the archive's limits is not
  - a date no cited passage carries is flagged; the publish date, a date
    written in the passage, or "N days ago" (within one) all count
  - "No archived news found" is an abstention
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "webapp"))

import citations  # noqa: E402

ITEMS = [
    {"ref": "P1", "date": "Wed Oct 21", "days_ago": 7, "text": "Limited in practice with an ankle injury."},
    {"ref": "P2", "date": "Sat Oct 24", "days_ago": 4, "text": "Listed questionable; hurt it Oct 12 at Detroit."},
]


class NumberingTest(unittest.TestCase):
    def test_same_passage_keeps_its_ref_across_calls(self):
        refs = {}
        first = {"players": [{"notes": [{"id": "a#0", "text": "x"}, {"id": "b#0", "text": "y"}]}]}
        second = {"passages": [{"id": "b#0", "text": "y"}, {"id": "c#1", "text": "z"}]}
        citations.number_items(first, refs)
        citations.number_items(second, refs)
        self.assertEqual([n["ref"] for n in first["players"][0]["notes"]], ["P1", "P2"])
        self.assertEqual([p["ref"] for p in second["passages"]], ["P2", "P3"])

    def test_text_is_the_identity_when_no_id_came_back(self):
        refs = {}
        citations.number_items({"passages": [{"text": "same"}, {"text": "same"}]}, refs)
        self.assertEqual(refs, {"same": "P1"})


class CheckTest(unittest.TestCase):
    def test_refs_before_or_after_the_period_belong_to_that_sentence(self):
        out = citations.check("He was limited Wednesday [P1]. He is questionable. [P2]", ITEMS)
        self.assertEqual([c["refs"] for c in out["claims"]], [["P1"], ["P2"]])
        self.assertEqual(out["problems"], [])
        self.assertEqual(out["cited_share"], 1.0)

    def test_two_refs_in_one_bracket_or_two(self):
        for text in ("Ankle, still limited [P1, P2].", "Ankle, still limited [P1][P2]."):
            self.assertEqual(citations.check(text, ITEMS)["claims"][0]["refs"], ["P1", "P2"])

    def test_unknown_ref_is_removed_and_flagged(self):
        out = citations.check("He is out for the season [P9]. Limited Wednesday [P1][P7].", ITEMS)
        self.assertEqual(out["answer"], "He is out for the season. Limited Wednesday [P1].")
        self.assertEqual(sorted(p["ref"] for p in out["problems"] if p["kind"] == "unknown_ref"),
                         ["P7", "P9"])
        self.assertIn("uncited", [p["kind"] for p in out["problems"]])

    def test_uncited_fact_flagged_but_archive_caveat_is_not(self):
        out = citations.check("He will play Sunday. The archive is a day behind live news.", ITEMS)
        self.assertEqual([p["kind"] for p in out["problems"]], ["uncited"])
        self.assertEqual(out["cited_share"], 0.0)

    def test_date_must_belong_to_a_cited_passage(self):
        ok = citations.check("Limited on Wed Oct 21 [P1]. Hurt Oct 12 at Detroit [P2]. "
                             "On Wednesday, October 21, he sat [P1].", ITEMS)   # full month name
        self.assertEqual(ok["problems"], [])
        for wrong in ("Limited on Oct 19 [P1].", "Limited on Oct 2 [P1]."):   # Oct 2 is not Oct 21
            bad = citations.check(wrong, ITEMS)
            self.assertEqual([p["kind"] for p in bad["problems"]], ["date_mismatch"])

    def test_days_ago_within_one(self):
        self.assertEqual(citations.check("Limited 6 days ago [P1].", ITEMS)["problems"], [])
        self.assertEqual([p["kind"] for p in citations.check("Limited 2 days ago [P1].", ITEMS)["problems"]],
                         ["date_mismatch"])

    def test_four_answer_types(self):
        cases = {
            "No archived news found about a knee injury for him.": "abstained",
            # live 2026-10-28: a refusal in other words is still an abstention
            "This archive contains NFL news only, so nothing on the Yankees.": "abstained",
            "No archived news found on his knee. He is limited with the ankle [P1].": "partial",
            "**Limited** Wednesday [P1].": "answered",
            "He will play Sunday.": "unsupported",
        }
        for text, kind in cases.items():
            out = citations.check(text, ITEMS)
            self.assertEqual(out["answer_type"], kind, text)
            self.assertEqual(out["abstained"], kind == "abstained", text)

    def test_initials_suffixes_and_times_do_not_split_a_sentence(self):
        for text in ("Odell Beckham Jr. was limited [P1].", "A.J. Brown was limited [P1].",
                     "C.J. Stroud and T.J. Hockenson were limited [P1].",
                     "He practiced at 3 p.m. Sunday [P1]."):
            out = citations.check(text, ITEMS)
            self.assertEqual(len(out["claims"]), 1, text)
            self.assertEqual(out["problems"], [], text)

    def test_abstention_then_advice_in_any_words(self):
        out = citations.check("No archived news found on his ankle. Check ESPN for updates.", ITEMS)
        self.assertEqual(out["answer_type"], "abstained")

    def test_header_line_is_not_an_uncited_fact(self):
        out = citations.check("**Puka Nacua:**\nLimited Wednesday [P1].", ITEMS)
        self.assertEqual(out["problems"], [])

    def test_a_count_after_a_month_is_not_a_date(self):
        self.assertEqual(citations.check("He has missed time since May 2 games ago [P1].", ITEMS)["problems"], [])

    def test_missing_days_ago_flags_instead_of_crashing(self):
        out = citations.check("Limited 10 days ago [P1].", [{"ref": "P1", "date": "Wed Oct 21", "days_ago": None}])
        self.assertEqual([p["kind"] for p in out["problems"]], ["date_mismatch"])

    def test_bold_answer_line_keeps_its_ref(self):
        out = citations.check("**He is trending better** [P2]. Limited earlier [P1].", ITEMS)
        self.assertEqual([c["refs"] for c in out["claims"]], [["P2"], ["P1"]])


if __name__ == "__main__":
    unittest.main()
