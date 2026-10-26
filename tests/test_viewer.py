import io
import json
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from deid_eval import dataset, targets, viewer
from deid_eval.config import write_jsonl
from deid_eval.phi_grader import score_note


def make_note(source, spans, protected=(), note_id="note-x"):
    phi = []
    for text, category in spans:
        start = source.index(text)
        phi.append({"start": start, "end": start + len(text), "category": category, "text": text})
    return {"id": note_id, "note_type": "exam_note", "source_text": source, "phi_spans": phi, "protected_values": list(protected)}


def row(note, deid, target="t", error=None):
    return targets.make_result(note, target, target, deid_text=deid, latency_ms=10.0, input_tokens=5, output_tokens=4, error=error)


def oracle(note):
    text = note["source_text"]
    for span in sorted(note["phi_spans"], key=lambda s: -s["start"]):
        text = text[:span["start"]] + "[%s]" % span["category"] + text[span["end"]:]
    return text


def segments_with(segments, cls):
    return [s for s in segments if len(s) > 1 and cls in s[1].split()]


class ViewerDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notes = dataset.generate(seed=42, size=9)
        cls.results = {
            "baseline": targets.run_target(targets.BaselineTarget(), cls.notes),
            "oracle": [row(n, oracle(n), "oracle") for n in cls.notes],
        }
        cls.data = viewer.build_data(cls.notes, cls.results, {})

    def test_metrics_and_mark_counts_match_grader(self):
        notes = {n["id"]: n for n in self.notes}
        for record in self.data["notes"]:
            for target, view in record["results"].items():
                deid = next(r for r in self.results[target] if r["id"] == record["id"])["deid_text"]
                score = score_note({**notes[record["id"]], "deid_text": deid})
                self.assertEqual(view["recall"], score["recall"])
                self.assertEqual((view["fn"], view["fp"]), (score["fn"], score["fp"]))
                self.assertEqual(view["preservation"], score["clinical_preservation"])
                kinds = [m["kind"] for m in view["marks"]]
                self.assertEqual(
                    (kinds.count("caught"), kinds.count("leaked"), kinds.count("over")),
                    (score["tp"], score["fn"], score["fp"]),
                )

    def test_segments_reassemble_both_texts(self):
        notes = {n["id"]: n for n in self.notes}
        for record in self.data["notes"]:
            for target, view in record["results"].items():
                deid = next(r for r in self.results[target] if r["id"] == record["id"])["deid_text"]
                self.assertEqual("".join(s[0] for s in view["src"]), notes[record["id"]]["source_text"])
                self.assertEqual("".join(s[0] for s in view["out"]), deid)

    def test_struggle_first_order(self):
        records = self.data["notes"]
        self.assertEqual([r["rank"] for r in records], list(range(len(records))))
        leaks = [r["leaks"] for r in records]
        self.assertEqual(leaks, sorted(leaks, reverse=True))
        self.assertGreater(records[0]["leaks"], 0)
        self.assertEqual(records[-1]["results"]["oracle"]["fn"], 0)

    def test_targets_follow_report_ranking(self):
        self.assertEqual([t["name"] for t in self.data["targets"]], ["oracle", "baseline"])


class ViewerHighlightTests(unittest.TestCase):
    def test_partial_redaction_in_both_panes(self):
        note = make_note("Lives at 12 Birch Lane today.", [("12 Birch Lane", "ADDRESS")])
        view = viewer.note_view(note, row(note, "Lives at 12 [ADDRESS] today."))
        leak = segments_with(view["src"], "leak")
        self.assertEqual("".join(s[0] for s in leak), "12 Birch Lane")
        self.assertIn("partly replaced", leak[0][2])
        self.assertEqual([s[0] for s in segments_with(view["out"], "leak")], ["12"])
        self.assertEqual([m["reason"] for m in view["marks"] if m["kind"] == "leaked"], ["partial"])

    def test_acuity_over_redaction_in_both_panes(self):
        note = make_note("Seen 03/14/2026. VA OD 20/40.", [("03/14/2026", "DATE")], ["20/40", "OD"])
        view = viewer.note_view(note, row(note, "Seen [DATE]. VA OD [DATE]."))
        over = segments_with(view["src"], "over")
        self.assertEqual([s[0] for s in over], ["20/40"])
        self.assertIn("lost", over[0][1].split())
        self.assertIn("with [DATE]", over[0][2])
        out_over = segments_with(view["out"], "over")
        self.assertEqual([s[0] for s in out_over], ["[DATE]"])
        self.assertEqual(view["out"][1][0], "[DATE]")
        self.assertEqual("ph", view["out"][1][1])
        self.assertEqual(sorted(m["kind"] for m in view["marks"]), ["caught", "lost", "over"])

    def test_added_text_and_repeat_leak(self):
        note = make_note("Ms. Ruiz called. Ms. Ruiz agrees.", [])
        second = note["source_text"].rindex("Ruiz")
        note["phi_spans"] = [
            {"start": 4, "end": 8, "category": "PATIENT_NAME", "text": "Ruiz"},
            {"start": second, "end": second + 4, "category": "PATIENT_NAME", "text": "Ruiz"},
        ]
        view = viewer.note_view(note, row(note, "Note: Ms. [PATIENT_NAME] called. Ms. Ruiz agrees."))
        self.assertEqual(sorted(m["reason"] for m in view["marks"] if m["kind"] == "leaked"), ["missed", "repeat"])
        self.assertIn("Note:", "".join(s[0] for s in segments_with(view["out"], "added")))
        self.assertEqual([s[0] for s in segments_with(view["out"], "leak")], ["Ruiz"])

    def test_errored_note(self):
        note = make_note("Seen by Dr. Alvarez", [("Alvarez", "PROVIDER_NAME")])
        view = viewer.note_view(note, row(note, "", error="RuntimeError: boom"))
        self.assertEqual(view["error"], "RuntimeError: boom")
        self.assertEqual(view["out"], [])
        self.assertEqual(view["marks"][0]["reason"], "failed")
        missing = viewer.note_view(note, None)
        self.assertIn("no result", missing["error"])


class ViewerFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.notes = dataset.generate(seed=42, size=3)
        hostile = row(self.notes[0], '</script><script>alert("x")</script> & <b>', "baseline")
        self.results = {"baseline": [hostile] + targets.run_target(targets.BaselineTarget(), self.notes[1:])}
        self.path = viewer.write_viewer(Path(self.tmp.name, "viewer.html"), self.notes, self.results, {})
        self.html = self.path.read_text(encoding="utf-8")

    def _data_block(self):
        match = re.search(r'<script type="application/json" id="viewer-data">(.*?)</script>', self.html, re.S)
        self.assertIsNotNone(match)
        return match.group(1)

    def test_single_file_without_external_resources(self):
        self.assertTrue(self.path.exists())
        # Links to Foundry runs are allowed (they load nothing); loaded resources are not.
        self.assertNotRegex(self.html, r'src\s*=\s*["\']?https?:')
        self.assertNotRegex(self.html, r'<link[^>]+https?:|@import|url\(\s*[\'"]?https?:|<i?frame')

    def test_foundry_links_present_and_safe(self):
        runs = {
            "baseline": {"report_url": "https://ai.azure.com/nextgen/r/x/run/1", "target": "baseline"},
            "other": {"report_url": "https://ai.azure.com/other"},
        }
        data = viewer.build_data(self.notes, self.results, {}, runs)
        self.assertEqual(data["targets"][0]["foundry_url"], "https://ai.azure.com/nextgen/r/x/run/1")
        unsafe = viewer.build_data(self.notes, self.results, {}, {"baseline": {"report_url": "javascript:alert(1)"}})
        self.assertIsNone(unsafe["targets"][0]["foundry_url"])
        self.assertIn('rel: "noopener noreferrer"', self.html)
        self.assertIn('target: "_blank"', self.html)

    def test_no_foundry_links_without_record(self):
        data = viewer.build_data(self.notes, self.results, {})
        self.assertTrue(all(t["foundry_url"] is None for t in data["targets"]))

    def test_disclaimer_present(self):
        data = json.loads(self._data_block())
        self.assertIn("not a compliance determination", data["disclaimer"].lower())
        self.assertIn('id="disclaimer"', self.html)

    def test_embedded_json_round_trips_and_cannot_close_script(self):
        block = self._data_block()
        self.assertNotIn("<", block)
        data = json.loads(block)
        out_text = "".join(s[0] for s in data["notes"][[n["id"] for n in data["notes"]].index(self.notes[0]["id"])]["results"]["baseline"]["out"])
        self.assertTrue(out_text.startswith('</script><script>alert("x")</script>'))
        self.assertEqual(self.html.count("</script>"), 2)

    def test_untrusted_text_rendered_with_text_content(self):
        self.assertNotIn("innerHTML", self.html)


class ViewCommandTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        self.notes = dataset.generate(seed=42, size=3)
        self.notes_path = self.tmp / "notes.jsonl"
        write_jsonl(self.notes_path, self.notes)
        self.cache_dir = self.tmp / "data" / "cache"
        self.cache_dir.mkdir(parents=True)
        self.out_dir = self.tmp / "outputs"

    def _run(self, *argv):
        from deid_eval import __main__ as cli

        buffer = io.StringIO()
        with mock.patch.object(cli, "NOTES_PATH", self.notes_path), \
             mock.patch.object(cli, "CACHE_DIR", self.cache_dir), \
             mock.patch.object(cli, "OUTPUT_DIR", self.out_dir), \
             mock.patch("webbrowser.open") as opened, redirect_stdout(buffer):
            code = cli.main(list(argv))
        return code, buffer.getvalue(), opened

    def test_no_cache_prints_hint_and_writes_nothing(self):
        code, output, opened = self._run("view", "--no-open")
        self.assertEqual(code, 1)
        self.assertIn("python -m deid_eval run", output)
        self.assertFalse((self.out_dir / "viewer.html").exists())
        opened.assert_not_called()

    def test_view_from_cache_without_opening(self):
        targets.save_cache(targets.run_target(targets.BaselineTarget(), self.notes), "baseline", self.cache_dir)
        code, output, opened = self._run("view", "--no-open")
        self.assertEqual(code, 0)
        self.assertTrue((self.out_dir / "viewer.html").exists())
        self.assertIn("viewer.html", output)
        opened.assert_not_called()

    def test_report_also_writes_viewer(self):
        targets.save_cache(targets.run_target(targets.BaselineTarget(), self.notes), "baseline", self.cache_dir)
        code, _, _ = self._run("report")
        self.assertEqual(code, 0)
        for name in ("report.md", "report.csv", "viewer.html"):
            self.assertTrue((self.out_dir / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
