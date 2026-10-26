"""In-memory stand-ins for the Foundry project and OpenAI evals clients (shapes verified live).

Runs complete on first retrieve, scoring items with the local grader entry points, so tests
exercise the same per-criterion results the service produces.
"""

import itertools
from types import SimpleNamespace

from deid_eval import foundry, phi_grader

_clock = itertools.count(1_800_000_000)


class NotFound(Exception):
    status_code = 404


class FakeEvaluators:
    def __init__(self):
        self.versions = {}
        self.created = 0

    def list_versions(self, name, headers=None):
        assert headers == foundry.PREVIEW_HEADERS
        if name not in self.versions:
            raise NotFound(name)
        return list(reversed(self.versions[name]))

    def create_version(self, name, evaluator_version, headers=None):
        assert headers == foundry.PREVIEW_HEADERS
        versions = self.versions.setdefault(name, [])
        entry = {
            "name": name,
            "version": str(len(versions) + 1),
            "metadata": dict(evaluator_version["metadata"]),
            "definition": evaluator_version["definition"],
        }
        versions.append(entry)
        self.created += 1
        return SimpleNamespace(version=entry["version"])


class FakeProject:
    def __init__(self):
        self.evaluators = FakeEvaluators()


class FakeOutputItems:
    def __init__(self, runs):
        self._runs = runs

    def list(self, run_id, eval_id):
        return list(self._runs.get(run_id).output_items)


class FakeRuns:
    def __init__(self):
        self.by_eval = {}
        self.output_items = FakeOutputItems(self)
        self.errored_notes = set()
        # Live shape: one criterion errors on every item while run-level result_counts says 0.
        self.errored_criteria = set()

    def create(self, eval_id, name, metadata, data_source):
        run = SimpleNamespace(
            id="evalrun_%d" % next(_clock), eval_id=eval_id, name=name, metadata=dict(metadata), status="queued",
            created_at=next(_clock), report_url=None, result_counts=None, per_testing_criteria_results=None,
            created_by="tester@example.com", output_items=[],
            items=[content["item"] for content in data_source["source"]["content"]],
        )
        self.by_eval.setdefault(eval_id, []).append(run)
        return run

    def list(self, eval_id):
        return list(self.by_eval.get(eval_id, []))

    def get(self, run_id):
        return next(run for runs in self.by_eval.values() for run in runs if run.id == run_id)

    def retrieve(self, run_id, eval_id):
        run = self.get(run_id)
        if run.status == "queued":
            self.complete(run)
        return run

    def complete(self, run):
        counts = {criterion: {"passed": 0, "failed": 0, "errored": 0} for criterion in foundry.CRITERIA}
        for item in run.items:
            results = []
            for criterion, (_name, entry, _display, _desc) in foundry.CRITERIA.items():
                if item["id"] in self.errored_notes or criterion in self.errored_criteria:
                    results.append({"name": criterion, "passed": False, "score": 0.0, "status": "completed",
                                    "sample": {"error": {"message": "An error occurred during grading"}}})
                    counts[criterion]["errored"] += 1
                    continue
                score = getattr(phi_grader, entry)({}, item)
                passed = score >= foundry.PASS_THRESHOLD
                results.append({"name": criterion, "passed": passed, "score": score, "status": "completed", "sample": None})
                counts[criterion]["passed" if passed else "failed"] += 1
            run.output_items.append(SimpleNamespace(datasource_item=dict(item), results=results, status="completed"))
        total_errored = sum(1 for item in run.items if item["id"] in self.errored_notes)
        run.status = "completed"
        run.report_url = "https://ai.azure.com/fake/%s" % run.id
        run.result_counts = SimpleNamespace(errored=total_errored, total=len(run.items))
        run.per_testing_criteria_results = [dict(testing_criteria=c, **v) for c, v in counts.items()]
        return run


class FakeEvals:
    def __init__(self):
        self.items = []
        self.runs = FakeRuns()

    def list(self):
        return list(self.items)

    def create(self, name, metadata, data_source_config, testing_criteria):
        ev = SimpleNamespace(id="eval_%d" % next(_clock), name=name, metadata=dict(metadata), created_at=next(_clock),
                             testing_criteria=testing_criteria, data_source_config=data_source_config)
        self.items.append(ev)
        return ev


class FakeClient:
    def __init__(self):
        self.evals = FakeEvals()
