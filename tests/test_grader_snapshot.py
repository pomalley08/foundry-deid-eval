"""Pins every grader metric on the cached target outputs, so grader refactors cannot change scores.

Regenerate after a live run changes data/cache:  python -m tests.test_grader_snapshot --update
"""

import hashlib
import json
import sys
import unittest
from pathlib import Path

from deid_eval.config import CACHE_DIR, NOTES_PATH, read_jsonl
from deid_eval.phi_grader import grade, score_note

SNAPSHOT_PATH = Path(__file__).parent / "fixtures" / "grader_snapshot.json"
_KEYS = ("tp", "fn", "fp", "recall", "precision", "residual_phi", "clinical_preservation", "by_category")


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def build_snapshot() -> dict:
    notes = {note["id"]: note for note in read_jsonl(NOTES_PATH)}
    snapshot = {}
    for path in sorted(CACHE_DIR.glob("*.jsonl")):
        for row in read_jsonl(path):
            note = notes[row["id"]]
            item = {**note, "deid_text": row.get("deid_text") or ""}
            result = score_note(item)
            snapshot["%s/%s" % (row["target"], row["id"])] = {
                "deid_sha1": _digest(item["deid_text"]),
                "grade": grade({}, item),
                **{key: result[key] for key in _KEYS},
            }
    return snapshot


class GraderSnapshotTests(unittest.TestCase):
    def test_scores_match_snapshot(self):
        if not SNAPSHOT_PATH.exists():
            self.skipTest("no snapshot; run: python -m tests.test_grader_snapshot --update")
        expected = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
        current = build_snapshot()
        if {k: v["deid_sha1"] for k, v in expected.items()} != {k: v["deid_sha1"] for k, v in current.items()}:
            self.skipTest("data/cache changed since the snapshot; run: python -m tests.test_grader_snapshot --update")
        for key, values in expected.items():
            self.assertEqual(json.loads(json.dumps(current[key])), values, key)


if __name__ == "__main__":
    if "--update" in sys.argv:
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT_PATH.write_text(json.dumps(build_snapshot(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print("wrote", SNAPSHOT_PATH)
    else:
        unittest.main()
