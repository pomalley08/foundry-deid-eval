import csv
import tempfile
import unittest
from pathlib import Path

from deid_eval import dataset, report, targets
from deid_eval.phi_grader import CATEGORIES


def fake_results(notes, name, model, deid, tokens=None):
    rows = []
    for note in notes:
        rows.append(targets.make_result(
            note, name, model, deid_text=deid(note), latency_ms=100.0,
            input_tokens=tokens, output_tokens=tokens,
        ))
    return rows


def oracle(note):
    text = note["source_text"]
    for span in sorted(note["phi_spans"], key=lambda s: -s["start"]):
        text = text[:span["start"]] + "[%s]" % span["category"] + text[span["end"]:]
    return text


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notes = dataset.generate(seed=42, size=6)

    def test_report_files_and_rows(self):
        results = {
            "baseline": targets.run_target(targets.BaselineTarget(), self.notes),
            "model-a": fake_results(self.notes, "model-a", "model-a", oracle, tokens=1000),
            "model-b": fake_results(self.notes, "model-b", "model-b", lambda n: n["source_text"], tokens=1000),
        }
        with tempfile.TemporaryDirectory() as tmp:
            ranked = report.build_report(self.notes, results, {}, Path(tmp))
            markdown = Path(tmp, "report.md").read_text(encoding="utf-8")
            with open(Path(tmp, "report.csv"), encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
        self.assertEqual([s["target"] for s in ranked], ["model-a", "baseline", "model-b"])
        self.assertIn("not a compliance determination", markdown.split("## Ranking")[0].lower())
        self.assertIn("not a compliance determination", rows[0][0].lower())
        self.assertEqual(len(rows) - 2, 3)
        for category in CATEGORIES:
            self.assertIn("| %s |" % category, markdown)

    def test_recall_ranks_above_cost(self):
        a = {"recall": 0.98, "residual_phi_rate": 0.1, "_cost_sort": 5.0, "target": "a"}
        b = {"recall": 0.95, "residual_phi_rate": 0.1, "_cost_sort": 0.1, "target": "b"}
        self.assertEqual([s["target"] for s in report.rank([b, a])], ["a", "b"])

    def test_cost_cells(self):
        priced = {"model-a": {"input_per_1m": 2.0, "output_per_1m": 8.0}}
        model_rows = fake_results(self.notes, "model-a", "model-a", oracle, tokens=1_000_000)
        summary = report.summarize("model-a", report.score_items(self.notes, model_rows), priced)
        self.assertAlmostEqual(summary["cost"], len(self.notes) * 10.0)
        unpriced = report.summarize("model-a", report.score_items(self.notes, model_rows), {})
        self.assertEqual(unpriced["cost"], "unpriced")
        baseline_rows = targets.run_target(targets.BaselineTarget(), self.notes)
        baseline = report.summarize("baseline", report.score_items(self.notes, baseline_rows), priced)
        self.assertEqual(baseline["cost"], "n/a")

    def test_price_lookup_by_model_prefix(self):
        pricing = {"gpt-4o": {"input_per_1m": 1}, "gpt-4o-mini": {"input_per_1m": 2}}
        self.assertEqual(report.find_price(pricing, "my-deploy", "gpt-4o-mini-2024-07-18"), {"input_per_1m": 2})
        self.assertEqual(report.find_price(pricing, "gpt-4o", "anything"), {"input_per_1m": 1})

    def test_percentiles(self):
        self.assertEqual(report.percentile([1, 2, 3, 4, 5], 50), 3)
        self.assertAlmostEqual(report.percentile(list(range(1, 101)), 95), 95.05)
        self.assertIsNone(report.percentile([], 50))


if __name__ == "__main__":
    unittest.main()
