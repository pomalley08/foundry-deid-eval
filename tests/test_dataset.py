import json
import tempfile
import unittest
from pathlib import Path

from deid_eval import dataset
from deid_eval.config import NOTES_PATH, read_jsonl
from deid_eval.phi_grader import CATEGORIES, occurrences


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = dataset.generate(seed=42, size=30)

    def test_same_seed_writes_identical_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp, "a.jsonl"), Path(tmp, "b.jsonl")
            dataset.write_dataset(first, dataset.generate(42, 30))
            dataset.write_dataset(second, dataset.generate(42, 30))
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(len(read_jsonl(first)), 30)

    def test_different_seed_changes_text(self):
        other = dataset.generate(seed=7, size=30)
        self.assertNotEqual([r["source_text"] for r in self.records], [r["source_text"] for r in other])

    def test_committed_dataset_matches_generator(self):
        self.assertEqual(read_jsonl(NOTES_PATH), json.loads(json.dumps(self.records)))

    def test_record_fields(self):
        for record in self.records:
            self.assertEqual(
                set(record), {"id", "note_type", "source_text", "phi_spans", "protected_values"}
            )

    def test_span_offsets_match_text(self):
        for record in self.records:
            for span in record["phi_spans"]:
                self.assertIn(span["category"], CATEGORIES)
                self.assertEqual(record["source_text"][span["start"]:span["end"]], span["text"])

    def test_no_unlabeled_phi_repeats(self):
        for record in self.records:
            self.assertEqual(dataset.check_record(record), [], record["id"])

    def test_category_coverage(self):
        seen = {span["category"] for record in self.records for span in record["phi_spans"]}
        self.assertEqual(seen, set(CATEGORIES))

    def test_protected_values_present_and_outside_phi(self):
        for record in self.records:
            self.assertTrue(record["protected_values"], record["id"])
            ranges = [(s["start"], s["end"]) for s in record["phi_spans"]]
            for value in record["protected_values"]:
                found = occurrences(value, record["source_text"])
                self.assertTrue(found, "%s: %r" % (record["id"], value))
                for start, end in found:
                    self.assertFalse(any(start < b and a < end for a, b in ranges), value)

    def test_hard_negative_ratio(self):
        def hard(record):
            return any(
                dataset.ACUITY_RE.match(v) or v in dataset.EPONYMS for v in record["protected_values"]
            )

        ratio = sum(1 for r in self.records if hard(r)) / len(self.records)
        self.assertGreaterEqual(ratio, 0.30)

    def test_note_type_mix(self):
        for note_type in dataset.NOTE_TYPES:
            share = sum(1 for r in self.records if r["note_type"] == note_type) / len(self.records)
            self.assertGreaterEqual(share, 0.25, note_type)
        self.assertTrue(all(r["note_type"] in dataset.NOTE_TYPES for r in self.records))

    def test_type_specific_content(self):
        for record in self.records:
            categories = {s["category"] for s in record["phi_spans"]}
            if record["note_type"] == "imaging_report":
                self.assertIn("DEVICE_ID", categories, record["id"])
            elif record["note_type"] == "referral_letter":
                self.assertIn("ADDRESS", categories, record["id"])
            else:
                self.assertTrue(
                    any(dataset.ACUITY_RE.match(v) for v in record["protected_values"]), record["id"]
                )


if __name__ == "__main__":
    unittest.main()
