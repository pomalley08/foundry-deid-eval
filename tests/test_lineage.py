import re
import tempfile
import unittest
from pathlib import Path

from deid_eval import dataset, lineage, targets


class LineageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notes = dataset.generate(seed=42, size=3)
        cls.baseline = targets.run_target(targets.BaselineTarget(), cls.notes)

    def test_dataset_sha_is_canonical(self):
        reordered = [dict(reversed(list(note.items()))) for note in self.notes]
        self.assertEqual(lineage.dataset_sha(self.notes), lineage.dataset_sha(reordered))
        changed = [dict(self.notes[0], source_text=self.notes[0]["source_text"] + " ")] + self.notes[1:]
        self.assertNotEqual(lineage.dataset_sha(self.notes), lineage.dataset_sha(changed))
        self.assertNotEqual(lineage.dataset_sha(self.notes), lineage.dataset_sha(dataset.generate(seed=7, size=3)))

    def test_outputs_sha_tracks_any_output_change(self):
        original = lineage.outputs_sha(self.notes, self.baseline)
        self.assertEqual(original, lineage.outputs_sha(self.notes, list(reversed(self.baseline))))
        edited = [dict(self.baseline[0], deid_text=self.baseline[0]["deid_text"] + "x")] + self.baseline[1:]
        self.assertNotEqual(original, lineage.outputs_sha(self.notes, edited))

    def test_git_commit(self):
        self.assertRegex(lineage.git_commit(), r"^([0-9a-f]{7,}(-dirty)?|unknown)$")
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(lineage.git_commit(Path(tmp)), "unknown")

    def test_run_lineage(self):
        meta = lineage.run_lineage("baseline", self.notes, self.baseline, "phi_recall=1", commit="abc1234")
        self.assertEqual(set(meta), {"target", "model", "prompt_sha", "outputs_sha", "dataset_sha", "evaluator_versions", "git_commit"})
        self.assertTrue(all(isinstance(v, str) and v for v in meta.values()))
        self.assertEqual((meta["model"], meta["prompt_sha"], meta["git_commit"]), ("regex-v1", "n/a", "abc1234"))

        model_rows = [dict(r, target="m", model="m-2025", input_tokens=10, prompt_sha=targets.PROMPT_SHA) for r in self.baseline]
        self.assertEqual(lineage.run_lineage("m", self.notes, model_rows, "x", commit="c")["prompt_sha"], targets.PROMPT_SHA)
        unrecorded = [dict(r, prompt_sha=None) for r in model_rows]
        self.assertEqual(lineage.run_lineage("m", self.notes, unrecorded, "x", commit="c")["prompt_sha"], "unknown")
        mixed = [dict(model_rows[0], prompt_sha="aaa")] + model_rows[1:]
        self.assertTrue(lineage.run_lineage("m", self.notes, mixed, "x", commit="c")["prompt_sha"].startswith("mixed:"))
        self.assertEqual(lineage.run_lineage("m", self.notes, model_rows, "", commit="c")["evaluator_versions"], "unknown")
        self.assertTrue(re.fullmatch(r"[0-9a-f]{12}", meta["dataset_sha"]))


if __name__ == "__main__":
    unittest.main()
