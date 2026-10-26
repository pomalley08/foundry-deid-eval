import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from deid_eval import dataset, foundry, gate, targets

sys.path.insert(0, str(Path(__file__).parent))
import fake_foundry  # noqa: E402
from test_foundry import oracle_rows  # noqa: E402


class GateTests(unittest.TestCase):
    def setUp(self):
        self.notes = dataset.generate(seed=42, size=6)
        perfect = oracle_rows(self.notes, "model-a")
        # model-b leaks one provider name in one note: a single, real recall regression.
        self.leaky_note = next(n for n in self.notes if any(s["category"] == "PROVIDER_NAME" for s in n["phi_spans"]))
        span = next(s for s in self.leaky_note["phi_spans"] if s["category"] == "PROVIDER_NAME")
        leaky = []
        for row in oracle_rows(self.notes, "model-b"):
            if row["id"] == self.leaky_note["id"]:
                text = row["deid_text"].replace("[PROVIDER_NAME]", span["text"], 1)
                row = dict(row, deid_text=text)
            leaky.append(row)
        self.results = {"baseline": targets.run_target(targets.BaselineTarget(), self.notes), "model-a": perfect, "model-b": leaky}
        self.project, self.client = fake_foundry.FakeProject(), fake_foundry.FakeClient()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out_dir = Path(self.tmp.name, "gate")
        for patcher in (mock.patch.dict(os.environ, {"FOUNDRY_EVAL_DEPLOYMENT": "gpt-test"}), mock.patch("deid_eval.foundry.time.sleep")):
            patcher.start()
            self.addCleanup(patcher.stop)

    def submit(self, results=None, wait=True):
        foundry.run_foundry(self.notes, results or self.results, wait=wait, log=lambda *_: None, project=self.project,
                            client=self.client, record_path=Path(self.tmp.name, "runs.json"), commit="abc1234")

    def gate(self, baseline, candidate, tolerance=0.0, allow_new_leaks=False):
        messages = []
        code = gate.run_gate(baseline, candidate, self.notes, tolerance_pp=tolerance, project=self.project,
                             client=self.client, out_dir=self.out_dir, log=messages.append, allow_new_leaks=allow_new_leaks)
        records = sorted(self.out_dir.glob("*.json"), key=lambda p: p.stat().st_mtime_ns)
        return code, json.loads(records[-1].read_text(encoding="utf-8")), messages

    def test_regression_exits_1_and_lists_notes(self):
        self.submit()
        code, record, messages = self.gate("model-a", "model-b")
        self.assertEqual(code, gate.REGRESSION)
        recall = next(row for row in record["criteria"] if row["criterion"] == "phi_recall")
        self.assertTrue(recall["regressed"])
        self.assertEqual(recall["regressed_notes"], [self.leaky_note["id"]])
        self.assertLess(recall["delta_pp"], 0)
        others = [row for row in record["criteria"] if row["criterion"] != "phi_recall"]
        self.assertFalse(any(row["regressed"] for row in others))
        self.assertIn("REGRESSION", messages[0])

    def test_no_regression_exits_0(self):
        self.submit()
        code, record, _ = self.gate("baseline", "model-a")
        self.assertEqual(code, gate.PASS)
        self.assertEqual(record["decision"], "pass")
        self.assertTrue(all(row["candidate_pass_rate"] == 1.0 for row in record["criteria"]))

    def test_tolerance(self):
        self.submit()
        # The leak in model-b is also a new leak, so rate-only judging is needed for tolerance to apply.
        self.assertEqual(self.gate("model-a", "model-b", tolerance=50)[0], gate.REGRESSION)
        self.assertEqual(self.gate("model-a", "model-b", tolerance=50, allow_new_leaks=True)[0], gate.PASS)

    def test_new_leak_fails_even_at_equal_pass_rate(self):
        other = next(n for n in self.notes if n["id"] != self.leaky_note["id"]
                     and any(s["category"] == "PROVIDER_NAME" for s in n["phi_spans"]))
        span = next(s for s in other["phi_spans"] if s["category"] == "PROVIDER_NAME")
        swapped = [dict(r, deid_text=r["deid_text"].replace("[PROVIDER_NAME]", span["text"], 1)) if r["id"] == other["id"] else r
                   for r in oracle_rows(self.notes, "model-c")]
        self.submit(results={**self.results, "model-c": swapped})
        code, record, _ = self.gate("model-b", "model-c")
        recall = next(row for row in record["criteria"] if row["criterion"] == "phi_recall")
        self.assertEqual(recall["delta_pp"], 0)
        self.assertEqual(recall["regressed_notes"], [other["id"]])
        self.assertEqual(recall["improved_notes"], [self.leaky_note["id"]])
        self.assertEqual(recall["regressed_by"], ["new leaks"])
        self.assertEqual(code, gate.REGRESSION)
        self.assertIn("1 notes newly leak PHI", record["reason"])
        self.assertEqual(self.gate("model-b", "model-c", allow_new_leaks=True)[0], gate.PASS)

    def test_missing_run_exits_2(self):
        self.submit()
        code, record, _ = self.gate("model-a", "not-a-target")
        self.assertEqual(code, gate.INCONCLUSIVE)
        self.assertIn("no completed run", record["reason"])

    def test_incomplete_run_exits_2(self):
        self.submit(results={"model-a": self.results["model-a"]})
        self.submit(results={"model-b": self.results["model-b"]}, wait=False)
        code, record, _ = self.gate("model-a", "model-b")
        self.assertEqual(code, gate.INCONCLUSIVE)
        self.assertIn("no completed run for candidate", record["reason"])

    def test_errored_items_exit_2(self):
        self.client.evals.runs.errored_notes = {self.notes[0]["id"]}
        self.submit()
        code, record, _ = self.gate("model-a", "model-b")
        self.assertEqual(code, gate.INCONCLUSIVE)
        self.assertIn("errored", record["reason"])

    def test_criterion_errored_with_clean_run_counts_exit_2(self):
        self.client.evals.runs.errored_criteria = {"phi_recall"}
        self.submit()
        run = foundry.latest_runs(self.client, self.client.evals.items[0].id)["model-b"]
        self.assertEqual(run.result_counts.errored, 0)
        self.assertEqual(foundry.errored_count(run), len(self.notes))
        code, record, _ = self.gate("model-a", "model-b")
        self.assertEqual(code, gate.INCONCLUSIVE)
        self.assertIn("errored in the grading sandbox", record["reason"])

    def test_no_evaluators_or_not_configured_exit_2_with_record(self):
        code, record, _ = self.gate("model-a", "model-b")
        self.assertEqual(code, gate.INCONCLUSIVE)
        self.assertIn("run `python -m deid_eval foundry` first", record["reason"])
        with mock.patch.dict(os.environ, {"FOUNDRY_PROJECT_ENDPOINT": ""}):
            code = gate.run_gate("model-a", "model-b", self.notes, out_dir=self.out_dir, log=lambda *_: None)
        self.assertEqual(code, gate.INCONCLUSIVE)
        self.assertEqual(len(list(self.out_dir.glob("*.md"))), 2)

    def test_decision_record_contents(self):
        self.submit()
        self.gate("model-a", "model-b")
        record = json.loads(sorted(self.out_dir.glob("*.json"))[-1].read_text(encoding="utf-8"))
        markdown = sorted(self.out_dir.glob("*.md"))[-1].read_text(encoding="utf-8")
        self.assertIn("not a compliance determination", record["disclaimer"].lower())
        self.assertIn("not a compliance determination", markdown.lower())
        for role in ("baseline", "candidate"):
            self.assertTrue(record[role]["run_id"].startswith("evalrun_"))
            self.assertTrue(record[role]["report_url"].startswith("https://"))
            self.assertEqual(record[role]["lineage"]["git_commit"], "abc1234")
            self.assertEqual(record[role]["created_by"], "tester@example.com")
        self.assertEqual(record["eval"]["name"], foundry.standing_eval_name())
        self.assertIn("**Decision: REGRESSION** (exit code 1)", markdown)
        self.assertIn("../viewer.html#%s/model-b" % self.leaky_note["id"], markdown)
        self.assertIn("| prompt_sha |", markdown)


if __name__ == "__main__":
    unittest.main()
