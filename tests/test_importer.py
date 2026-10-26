import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from deid_eval import dataset, importer, targets
from deid_eval.config import data_paths, read_jsonl, write_jsonl
from deid_eval.phi_grader import score_note


def oracle_spans(note):
    return [{"start": s["start"], "end": s["end"], "category": s["category"]} for s in note["phi_spans"]]


class ImporterTests(unittest.TestCase):
    def setUp(self):
        self.notes = dataset.generate(seed=42, size=6)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name, "outputs.jsonl")

    def write(self, rows):
        write_jsonl(self.path, rows)
        return self.path

    def test_text_format(self):
        rows = [{"id": n["id"], "deid_text": n["source_text"], "model": "pipeline-v3", "latency_ms": 12.5} for n in self.notes]
        results = importer.import_outputs(self.notes, self.write(rows), "my-pipeline")
        self.assertEqual([r["id"] for r in results], [n["id"] for n in self.notes])
        self.assertEqual(results[0]["model"], "pipeline-v3")
        self.assertEqual(results[0]["target"], "my-pipeline")
        self.assertEqual(results[0]["source_sha1"], targets.source_hash(self.notes[0]["source_text"]))
        self.assertEqual(score_note({**self.notes[0], "deid_text": results[0]["deid_text"]})["recall"], 0.0)

    def test_spans_format_scores_like_oracle(self):
        rows = [{"id": n["id"], "spans": oracle_spans(n)} for n in self.notes]
        for note, result in zip(self.notes, importer.import_outputs(self.notes, self.write(rows), "spans-target", fmt="spans")):
            score = score_note({**note, "deid_text": result["deid_text"]})
            self.assertEqual((score["recall"], score["precision"], score["clinical_preservation"]), (1.0, 1.0, 1.0))

    def test_overlapping_spans_are_merged(self):
        self.assertEqual(importer.apply_spans("Ana Ruiz seen", [
            {"start": 0, "end": 3, "category": "first"}, {"start": 2, "end": 8, "category": "name"}]), "[FIRST] seen")
        with self.assertRaises(ValueError):
            importer.apply_spans("short", [{"start": 2, "end": 99, "category": "x"}])

    def test_ahds_tag_format_with_split_address(self):
        source = "Pt: Ana Ruiz lives at 12 Birch Lane, Fairview, OR 97035."
        address = "12 Birch Lane, Fairview, OR 97035"
        note = {"id": "n1", "note_type": "exam_note", "source_text": source, "protected_values": [],
                "phi_spans": [
                    {"start": 4, "end": 12, "category": "PATIENT_NAME", "text": "Ana Ruiz"},
                    {"start": source.index(address), "end": source.index(address) + len(address), "category": "ADDRESS", "text": address},
                ]}

        def entity(text, category):
            start = source.index(text)
            return {"category": category, "text": text, "offset": {"utf8": start, "utf16": start, "codePoint": start},
                    "length": {"utf8": len(text), "utf16": len(text), "codePoint": len(text)}}

        # AHDS tags address parts separately, leaving ", " between them.
        row = {"id": "n1", "taggerResult": {"entities": [
            entity("Ana Ruiz", "Patient"), entity("12 Birch Lane", "Street"), entity("Fairview", "City"),
            entity("OR", "State"), entity("97035", "Zip")]}}
        result = importer.import_outputs([note], self.write([row]), "ahds", fmt="ahds-tag")[0]
        self.assertEqual(result["deid_text"], "Pt: [PATIENT_NAME] lives at [ADDRESS], [ADDRESS], [ADDRESS] [ADDRESS].")
        score = score_note({**note, "deid_text": result["deid_text"]})
        self.assertEqual(score["recall"], 1.0)
        self.assertEqual(score["fp"], 0)

    def test_validation_errors(self):
        rows = [{"id": n["id"], "deid_text": n["source_text"]} for n in self.notes]
        with self.assertRaisesRegex(ValueError, "not in the dataset"):
            importer.import_outputs(self.notes, self.write(rows + [{"id": "nope", "deid_text": "x"}]), "t")
        with self.assertRaisesRegex(ValueError, "more than once"):
            importer.import_outputs(self.notes, self.write(rows + rows[:1]), "t")
        with self.assertRaisesRegex(ValueError, "have no row"):
            importer.import_outputs(self.notes, self.write(rows[1:]), "t")
        with self.assertRaisesRegex(ValueError, "no deid_text"):
            importer.import_outputs(self.notes, self.write([{"id": n["id"]} for n in self.notes]), "t")

    def test_missing_and_error_rows_are_recorded_as_errors(self):
        rows = [{"id": n["id"], "deid_text": n["source_text"]} for n in self.notes[1:]]
        rows[0] = {"id": rows[0]["id"], "error": "timeout"}
        results = importer.import_outputs(self.notes, self.write(rows), "t", allow_missing=True)
        self.assertEqual(results[0]["error"], "missing from import file")
        self.assertEqual(results[1]["error"], "timeout")
        self.assertTrue(score_note({**self.notes[0], "deid_text": results[0]["deid_text"]})["failed"])


class ImportCommandTests(unittest.TestCase):
    def test_import_into_custom_data_dir_then_report(self):
        from deid_eval import __main__ as cli

        originals = (cli.NOTES_PATH, cli.CACHE_DIR)
        self.addCleanup(lambda: (setattr(cli, "NOTES_PATH", originals[0]), setattr(cli, "CACHE_DIR", originals[1])))
        with tempfile.TemporaryDirectory() as tmp:
            data_dir, out_dir = Path(tmp, "private"), Path(tmp, "outputs")
            notes = dataset.generate(seed=42, size=4)
            write_jsonl(data_dir / "notes.jsonl", notes)
            outputs = Path(tmp, "pipeline.jsonl")
            write_jsonl(outputs, [{"id": n["id"], "spans": oracle_spans(n)} for n in notes])
            buffer = io.StringIO()
            with mock.patch.dict(os.environ, {"DEID_DATA_DIR": str(data_dir)}), \
                 mock.patch.object(cli, "OUTPUT_DIR", out_dir), redirect_stdout(buffer):
                self.assertEqual(data_paths(), (data_dir / "notes.jsonl", data_dir / "cache"))
                self.assertEqual(cli.main(["import", "--target", "my-pipeline", "--file", str(outputs), "--format", "spans"]), 0)
                self.assertEqual(cli.main(["report"]), 0)
            cached = read_jsonl(data_dir / "cache" / "my-pipeline.jsonl")
            self.assertEqual(len(cached), 4)
            self.assertIn("my-pipeline", (out_dir / "report.md").read_text(encoding="utf-8"))
            self.assertIn("Imported 4 notes", buffer.getvalue())

    def test_relative_data_dir_resolves_from_repo_root(self):
        from deid_eval.config import ROOT

        with mock.patch.dict(os.environ, {"DEID_DATA_DIR": "data/private"}):
            self.assertEqual(data_paths()[0], ROOT / "data" / "private" / "notes.jsonl")


if __name__ == "__main__":
    unittest.main()
