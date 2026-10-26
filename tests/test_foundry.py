import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from deid_eval import dataset, foundry, report, targets
from deid_eval.phi_grader import grade, score_note

sys.path.insert(0, str(Path(__file__).parent))
import fake_foundry  # noqa: E402


def oracle_rows(notes, target):
    rows = []
    for note in notes:
        text = note["source_text"]
        for span in sorted(note["phi_spans"], key=lambda s: -s["start"]):
            text = text[:span["start"]] + "[%s]" % span["category"] + text[span["end"]:]
        rows.append(targets.make_result(note, target, target + "-2026", deid_text=text, latency_ms=5.0,
                                        input_tokens=10, output_tokens=8, prompt_sha=targets.PROMPT_SHA))
    return rows


class StandingEvalTests(unittest.TestCase):
    def setUp(self):
        self.notes = dataset.generate(seed=42, size=6)
        self.results = {
            "baseline": targets.run_target(targets.BaselineTarget(), self.notes),
            "model-a": oracle_rows(self.notes, "model-a"),
        }
        self.project, self.client = fake_foundry.FakeProject(), fake_foundry.FakeClient()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.record = Path(self.tmp.name, "foundry_runs.json")
        patcher = mock.patch.dict(os.environ, {"FOUNDRY_EVAL_DEPLOYMENT": "gpt-test"})
        patcher.start()
        self.addCleanup(patcher.stop)
        sleeper = mock.patch("deid_eval.foundry.time.sleep")
        sleeper.start()
        self.addCleanup(sleeper.stop)

    def run_step(self, notes=None, results=None, **kwargs):
        messages = []
        code = foundry.run_foundry(notes or self.notes, results or self.results, log=messages.append,
                                   project=self.project, client=self.client, record_path=self.record,
                                   commit="abc1234", **kwargs)
        self.assertEqual(code, 0)
        return messages

    def runs(self, eval_obj=None):
        eval_obj = eval_obj or self.client.evals.items[-1]
        return self.client.evals.runs.list(eval_obj.id)

    def test_eval_name_is_configurable(self):
        with mock.patch.dict(os.environ, {"FOUNDRY_EVAL_NAME": "pipeline-v2-gate"}):
            self.run_step()
        self.assertEqual(self.client.evals.items[0].name, "pipeline-v2-gate")
        self.run_step()
        self.assertEqual(self.client.evals.items[-1].name, foundry.DEFAULT_EVAL_NAME)
        self.assertEqual(len(self.client.evals.items), 2)

    def test_first_run_registers_evaluators_and_creates_standing_eval(self):
        self.run_step()
        self.assertEqual(self.project.evaluators.created, 3)
        self.assertEqual(len(self.client.evals.items), 1)
        ev = self.client.evals.items[0]
        self.assertEqual(ev.name, foundry.standing_eval_name())
        self.assertEqual(ev.metadata["evaluator_versions"], "phi_recall=1,over_redaction=1,clinical_preservation=1")
        self.assertEqual([c["name"] for c in ev.testing_criteria], list(foundry.CRITERIA))
        self.assertTrue(all(c["evaluator_version"] == "1" and c["initialization_parameters"]["pass_threshold"] == 1.0
                            for c in ev.testing_criteria))
        self.assertEqual(sorted(run.name for run in self.runs()), ["baseline", "model-a"])

    def test_lineage_metadata_on_runs(self):
        self.run_step()
        model_run = next(run for run in self.runs() if run.name == "model-a")
        meta = model_run.metadata
        self.assertEqual(meta["target"], "model-a")
        self.assertEqual(meta["model"], "model-a-2026")
        self.assertEqual(meta["prompt_sha"], targets.PROMPT_SHA)
        self.assertEqual(meta["git_commit"], "abc1234")
        for key in ("outputs_sha", "dataset_sha", "evaluator_versions"):
            self.assertTrue(meta[key] and meta[key] != "unknown", key)
        self.assertEqual(next(run for run in self.runs() if run.name == "baseline").metadata["prompt_sha"], "n/a")

    def test_rerun_reuses_eval_and_versions_and_skips_unchanged_outputs(self):
        self.run_step()
        messages = self.run_step()
        self.assertEqual(self.project.evaluators.created, 3)
        self.assertEqual(len(self.client.evals.items), 1)
        self.assertEqual(len(self.runs()), 2)
        self.assertTrue(any("unchanged outputs" in m for m in messages))
        self.run_step(force=True)
        self.assertEqual(len(self.runs()), 4)

    def test_run_with_grading_errors_is_resubmitted_without_force(self):
        self.client.evals.runs.errored_criteria = {"phi_recall"}
        self.run_step(wait=True)
        self.client.evals.runs.errored_criteria = set()
        messages = self.run_step(wait=True)
        self.assertTrue(any("had grading errors; resubmitting" in m for m in messages))
        self.assertEqual(len(self.runs()), 4)
        latest = foundry.latest_runs(self.client, self.client.evals.items[0].id)
        self.assertTrue(all(foundry.errored_count(run) == 0 for run in latest.values()))
        self.run_step()
        self.assertEqual(len(self.runs()), 4)

    def test_changed_outputs_add_a_run_to_the_same_eval(self):
        self.run_step()
        changed = dict(self.results)
        changed["model-a"] = [dict(r, deid_text=r["deid_text"] + " ") for r in self.results["model-a"]]
        self.run_step(results=changed)
        self.assertEqual(len(self.client.evals.items), 1)
        self.assertEqual(sorted(run.name for run in self.runs()), ["baseline", "model-a", "model-a"])

    def test_dataset_change_creates_new_eval(self):
        self.run_step()
        first = self.client.evals.items[0]
        other = dataset.generate(seed=7, size=6)
        self.run_step(notes=other, results={"baseline": targets.run_target(targets.BaselineTarget(), other)})
        self.assertEqual(len(self.client.evals.items), 2)
        self.assertEqual(len(self.runs(first)), 2)

    def test_grader_change_registers_new_versions_and_new_eval(self):
        self.run_step()
        original = foundry.evaluator_code
        with mock.patch.object(foundry, "evaluator_code", lambda entry: original(entry) + "\n# changed\n"):
            self.run_step()
        self.assertEqual(self.project.evaluators.created, 6)
        self.assertEqual(len(self.client.evals.items), 2)
        self.assertEqual(self.client.evals.items[1].metadata["evaluator_versions"],
                         "phi_recall=2,over_redaction=2,clinical_preservation=2")

    def test_run_record_and_report_links_after_wait(self):
        self.run_step(wait=True)
        recorded = foundry.load_run_record(self.record)
        self.assertEqual(sorted(entry["target"] for entry in recorded), ["baseline", "model-a"])
        latest = foundry.latest_recorded_runs(self.record)
        self.assertTrue(latest["model-a"]["report_url"].startswith("https://"))
        self.assertEqual(latest["model-a"]["metadata"]["git_commit"], "abc1234")

    def test_read_only_lookups(self):
        self.assertIsNone(foundry.find_evaluator_versions(self.project))
        self.run_step()
        versions = foundry.find_evaluator_versions(self.project)
        self.assertEqual(versions, {"phi_recall": "1", "over_redaction": "1", "clinical_preservation": "1"})
        from deid_eval.lineage import dataset_sha

        self.assertIsNotNone(foundry.find_standing_eval(self.client, versions, dataset_sha(self.notes)))
        self.assertEqual(self.project.evaluators.created, 3)


class FoundryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notes = dataset.generate(seed=42, size=6)
        cls.results = {"baseline": targets.run_target(targets.BaselineTarget(), cls.notes)}

    def test_skips_without_project_endpoint_and_report_still_written(self):
        messages = []
        with mock.patch.dict(os.environ, {"FOUNDRY_PROJECT_ENDPOINT": ""}):
            self.assertEqual(foundry.run_foundry(self.notes, self.results, log=messages.append), 0)
        self.assertIn("skipped", messages[0])
        with tempfile.TemporaryDirectory() as tmp:
            report.build_report(self.notes, self.results, {}, Path(tmp))
            self.assertTrue(Path(tmp, "report.md").exists())

    def test_foundry_items_score_like_local(self):
        items = foundry.build_items(self.notes, self.results["baseline"])
        self.assertEqual(len(items), len(self.notes))
        for item, note, row in zip(items, self.notes, self.results["baseline"]):
            self.assertTrue(all(isinstance(value, str) for value in item.values()))
            local = score_note({**note, "deid_text": row["deid_text"]})["recall"]
            self.assertEqual(grade({}, item), local)

    def test_registered_code_is_the_grader_file(self):
        self.assertIn("def grade(sample", foundry.grader_source())

    def test_each_criterion_variant_is_sandbox_safe_and_matches_local(self):
        import ast

        from deid_eval import phi_grader

        forbidden = {"compile", "exec", "eval", "open", "__import__"}
        items = foundry.build_items(self.notes, self.results["baseline"])
        for criterion, (_name, entry, _display, _desc) in foundry.CRITERIA.items():
            code = foundry.evaluator_code(entry)
            self.assertLess(len(code.encode("utf-8")), 256 * 1024)
            for node in ast.walk(ast.parse(code)):
                if isinstance(node, ast.Call):
                    func = node.func
                    name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
                    self.assertNotIn(name, forbidden, criterion)
            namespace = {}
            exec(compile(code, criterion, "exec"), namespace)
            local = getattr(phi_grader, entry)
            for item in items:
                self.assertEqual(namespace["grade"]({}, item), local({}, item), (criterion, item["id"]))


if __name__ == "__main__":
    unittest.main()
