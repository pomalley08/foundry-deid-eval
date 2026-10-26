"""Foundry cloud evaluation: a standing eval that accumulates runs, scored by the local grader.

- Three code evaluators (recall, over-redaction, clinical preservation) are registered from
  phi_grader.py, with a new version only when the code changes (content hash in metadata).
- Runs accumulate in one standing eval per (dataset, evaluator versions), so the portal shows
  history and can compare any run against a baseline.
- Every run carries lineage metadata; unchanged outputs are not resubmitted.
- Each run is recorded locally in outputs/foundry_runs.json for the viewer's links.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from . import phi_grader
from .config import OUTPUT_DIR, env_list
from .lineage import dataset_sha, git_commit, run_lineage

DEFAULT_EVAL_NAME = "deid-quality-gate"


def standing_eval_name() -> str:
    """Name of the standing eval; set FOUNDRY_EVAL_NAME to keep separate histories (e.g. per pipeline)."""
    return os.environ.get("FOUNDRY_EVAL_NAME") or DEFAULT_EVAL_NAME

PASS_THRESHOLD = 1.0
PREVIEW_HEADERS = {"Foundry-Features": "Evaluations=V1Preview"}
RUN_RECORD = OUTPUT_DIR / "foundry_runs.json"
_TEXT_FIELDS = ("id", "note_type", "target", "model", "source_text", "deid_text", "phi_spans", "protected_values")

# criterion name -> (evaluator name, phi_grader entry point, display name, description)
CRITERIA = {
    "phi_recall": (
        "deid_phi_recall", "grade", "PHI recall (strict)",
        "Per-note strict PHI recall: a gold span counts only when fully replaced and its text no longer appears.",
    ),
    "over_redaction": (
        "deid_over_redaction", "grade_precision", "Over-redaction (precision)",
        "Per-note precision: caught PHI spans divided by caught spans plus non-PHI text that was replaced.",
    ),
    "clinical_preservation": (
        "deid_clinical_preservation", "grade_preservation", "Clinical preservation",
        "Per-note share of protected clinical values (acuity, IOP, laterality, medications, eponyms) kept verbatim.",
    ),
}


def grader_source() -> str:
    return Path(phi_grader.__file__).read_text(encoding="utf-8")


def evaluator_code(entry: str) -> str:
    """Code registered for one criterion: the shared grader source, with `grade` bound to `entry`.

    The sandbox calls a function named `grade`, so the non-default criteria append a wrapper that
    redefines it. All three evaluators therefore run the identical scoring source.
    """
    source = grader_source()
    if entry == "grade":
        return source
    return source.rstrip("\n") + "\n\n\ndef grade(sample, item):\n    return %s(sample, item)\n" % entry


def code_sha(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]


def build_items(notes: list[dict], results: list[dict]) -> list[dict]:
    """One Foundry item per note. Lists are sent as JSON strings, which the grader accepts."""
    by_id = {row["id"]: row for row in results}
    items = []
    for note in notes:
        row = by_id.get(note["id"], {})
        items.append({
            "id": note["id"],
            "note_type": note.get("note_type", ""),
            "target": row.get("target", ""),
            "model": row.get("model") or "",
            "source_text": note["source_text"],
            "deid_text": row.get("deid_text") or "",
            "phi_spans": json.dumps(note["phi_spans"], ensure_ascii=False),
            "protected_values": json.dumps(note["protected_values"], ensure_ascii=False),
        })
    return items


def _item_properties() -> dict:
    return {field: {"type": "string"} for field in _TEXT_FIELDS}


# --- Plumbing ---------------------------------------------------------------------------------


def _status(exc) -> int | None:
    return getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)


def _retry(call, attempts: int = 3, delay_s: float = 2.0):
    """Retry transient Foundry failures (a 403 on list calls was observed once, then cleared)."""
    for attempt in range(attempts):
        try:
            return call()
        except Exception as exc:
            transient = _status(exc) in (403, 408, 429, 500, 502, 503, 504) or "Connection" in type(exc).__name__
            if attempt == attempts - 1 or not transient:
                raise
            time.sleep(delay_s * (attempt + 1))


def _as_dict(obj) -> dict:
    if isinstance(obj, dict):
        return obj
    for method in ("as_dict", "model_dump"):
        if hasattr(obj, method):
            return getattr(obj, method)()
    return dict(vars(obj))


def _meta(obj) -> dict:
    return dict(getattr(obj, "metadata", None) or {})


def errored_count(run) -> int:
    """Items that errored in grading. Run-level counts can say 0 while one criterion errored on
    every item (observed live), so per-criterion counts are checked too."""
    counts = getattr(run, "result_counts", None)
    run_level = (getattr(counts, "errored", 0) or 0) if counts is not None else 0
    per_criterion = [int(_as_dict(r).get("errored") or 0) for r in getattr(run, "per_testing_criteria_results", None) or []]
    return max([run_level] + per_criterion)


def _evaluator_ops(project):
    # SDK versions differ on whether the evaluator operations live on .evaluators or .beta.evaluators.
    candidates = (getattr(project, "evaluators", None), getattr(getattr(project, "beta", None), "evaluators", None))
    return next(ops for ops in candidates if ops is not None and hasattr(ops, "create_version"))


def connect():
    """Return (project_client, openai_client) or raise ImportError/KeyError when not available."""
    from azure.ai.projects import AIProjectClient
    from azure.identity import DefaultAzureCredential

    project = AIProjectClient(endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"], credential=DefaultAzureCredential())
    return project, project.get_openai_client()


def not_configured_reason() -> str | None:
    endpoint = os.environ.get("FOUNDRY_PROJECT_ENDPOINT", "")
    if not endpoint or "<" in endpoint:
        return "set FOUNDRY_PROJECT_ENDPOINT in .env"
    return None


# --- Evaluators and the standing eval --------------------------------------------------------


def _latest_version(ops, name: str) -> dict | None:
    try:
        versions = [_as_dict(v) for v in _retry(lambda: list(ops.list_versions(name=name, headers=PREVIEW_HEADERS)))]
    except Exception as exc:
        if _status(exc) == 404 or "NotFound" in type(exc).__name__:
            return None
        raise
    if not versions:
        return None
    return max(versions, key=lambda v: int(v["version"]) if str(v.get("version", "")).isdigit() else -1)


def _evaluator_version(name: str, code: str, display: str, description: str, sha: str) -> dict:
    try:
        from azure.ai.projects.models import EvaluatorCategory, EvaluatorDefinitionType

        category, code_type = EvaluatorCategory.QUALITY, EvaluatorDefinitionType.CODE
    except ImportError:
        category, code_type = "quality", "code"

    return {
        "name": name,
        "categories": [category],
        "display_name": display,
        "description": "%s Code sha %s." % (description, sha),
        "metadata": {"code_sha": sha, "source": "deid_eval/phi_grader.py"},
        "definition": {
            "type": code_type,
            "code_text": code,
            "init_parameters": {
                "type": "object",
                "properties": {"deployment_name": {"type": "string"}, "pass_threshold": {"type": "number"}},
                "required": ["deployment_name", "pass_threshold"],
            },
            "metrics": {
                "result": {"type": "continuous", "desirable_direction": "increase", "min_value": 0.0, "max_value": 1.0}
            },
            "data_schema": {
                "type": "object",
                "required": ["item"],
                "properties": {"item": {"type": "object", "properties": _item_properties()}},
            },
        },
    }


def find_evaluator_versions(project) -> dict[str, str] | None:
    """Read-only: the registered version of each criterion whose code matches the local grader."""
    ops = _evaluator_ops(project)
    versions = {}
    for criterion, (name, entry, _display, _description) in CRITERIA.items():
        latest = _latest_version(ops, name)
        if not latest or (latest.get("metadata") or {}).get("code_sha") != code_sha(evaluator_code(entry)):
            return None
        versions[criterion] = str(latest["version"])
    return versions


def ensure_evaluators(project, log=print) -> dict[str, str]:
    """Register a new evaluator version only for criteria whose code changed."""
    ops = _evaluator_ops(project)
    versions = {}
    for criterion, (name, entry, display, description) in CRITERIA.items():
        code = evaluator_code(entry)
        if len(code.encode("utf-8")) >= 256 * 1024:
            raise RuntimeError("evaluator code for %s must be under 256 KB" % criterion)
        sha = code_sha(code)
        latest = _latest_version(ops, name)
        if latest and (latest.get("metadata") or {}).get("code_sha") == sha:
            versions[criterion] = str(latest["version"])
            log("  evaluator %-28s v%s (unchanged)" % (name, versions[criterion]))
            continue
        created = ops.create_version(
            name=name,
            evaluator_version=_evaluator_version(name, code, display, description, sha),
            headers=PREVIEW_HEADERS,
        )
        versions[criterion] = str(created.version)
        log("  evaluator %-28s v%s registered (code changed)" % (name, versions[criterion]))
    return versions


def versions_tag(versions: dict[str, str]) -> str:
    return ",".join("%s=%s" % (criterion, versions[criterion]) for criterion in CRITERIA if criterion in versions)


def find_standing_eval(client, versions: dict[str, str], dataset_hash: str):
    """Read-only: the newest standing eval for this dataset and these evaluator versions."""
    tag = versions_tag(versions)
    matches = [
        ev for ev in _retry(lambda: list(client.evals.list()))
        if ev.name == standing_eval_name() and _meta(ev).get("dataset_sha") == dataset_hash
        and _meta(ev).get("evaluator_versions") == tag
    ]
    return max(matches, key=lambda ev: getattr(ev, "created_at", 0) or 0) if matches else None


def ensure_standing_eval(client, versions: dict[str, str], dataset_hash: str, deployment: str, log=print):
    existing = find_standing_eval(client, versions, dataset_hash)
    if existing is not None:
        log("Standing eval %s reused (id %s)" % (standing_eval_name(), existing.id))
        return existing
    created = client.evals.create(
        name=standing_eval_name(),
        metadata={
            "dataset_sha": dataset_hash,
            "evaluator_versions": versions_tag(versions),
            "purpose": "De-ID quality gate: three criteria, pass threshold 1.0",
        },
        data_source_config={
            "type": "custom",
            "item_schema": {"type": "object", "properties": _item_properties(), "required": list(_TEXT_FIELDS)},
            "include_sample_schema": False,
        },
        testing_criteria=[
            {
                "type": "azure_ai_evaluator",
                "name": criterion,
                "evaluator_name": CRITERIA[criterion][0],
                # Pinned so every run in this eval is scored by exactly the same grader code.
                "evaluator_version": versions[criterion],
                "initialization_parameters": {"deployment_name": deployment, "pass_threshold": PASS_THRESHOLD},
            }
            for criterion in CRITERIA
        ],
    )
    log("Standing eval %s created (id %s) for dataset %s, evaluators %s"
        % (standing_eval_name(), created.id, dataset_hash, versions_tag(versions)))
    return created


def latest_runs(client, eval_id: str, completed_only: bool = False) -> dict[str, object]:
    """Newest run per target in an eval (by metadata.target), optionally only completed runs."""
    newest: dict[str, object] = {}
    for run in _retry(lambda: list(client.evals.runs.list(eval_id=eval_id))):
        target = _meta(run).get("target")
        if not target or (completed_only and getattr(run, "status", None) != "completed"):
            continue
        if target not in newest or (getattr(run, "created_at", 0) or 0) > (getattr(newest[target], "created_at", 0) or 0):
            newest[target] = run
    return newest


# --- Local run record -------------------------------------------------------------------------


def load_run_record(path: Path = RUN_RECORD) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("runs", [])
    except (ValueError, AttributeError):
        return []


def upsert_run_record(entries: list[dict], path: Path = RUN_RECORD) -> None:
    runs = load_run_record(path)
    index = {entry["run_id"]: i for i, entry in enumerate(runs)}
    for entry in entries:
        if entry["run_id"] in index:
            runs[index[entry["run_id"]]].update({k: v for k, v in entry.items() if v is not None})
        else:
            index[entry["run_id"]] = len(runs)
            runs.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"runs": runs}, indent=1) + "\n", encoding="utf-8")


def latest_recorded_runs(path: Path = RUN_RECORD) -> dict[str, dict]:
    """Newest recorded run per target that has a report URL (for the viewer's links)."""
    newest: dict[str, dict] = {}
    for entry in load_run_record(path):
        if entry.get("report_url") and entry.get("target"):
            if entry["target"] not in newest or entry.get("created_at", "") >= newest[entry["target"]].get("created_at", ""):
                newest[entry["target"]] = entry
    return newest


def _record_entry(eval_obj, run, metadata: dict | None = None) -> dict:
    created = getattr(run, "created_at", None)
    return {
        "eval_id": eval_obj.id,
        "eval_name": getattr(eval_obj, "name", standing_eval_name()),
        "run_id": run.id,
        "target": (metadata or _meta(run)).get("target"),
        "status": getattr(run, "status", None),
        "report_url": getattr(run, "report_url", None),
        "created_at": datetime.fromtimestamp(created, timezone.utc).isoformat() if isinstance(created, (int, float)) else created,
        "metadata": metadata or _meta(run),
    }


# --- Main step --------------------------------------------------------------------------------


def run_foundry(notes: list[dict], results_by_target: dict[str, list[dict]], wait: bool = False, force: bool = False,
                log=print, project=None, client=None, record_path: Path = RUN_RECORD, commit: str | None = None) -> int:
    reason = not_configured_reason() if project is None else None
    if reason:
        log("Foundry step skipped: %s. The local report and viewer are unaffected." % reason)
        return 0
    if not results_by_target:
        log("Foundry step skipped: no target results to evaluate.")
        return 0
    deployment = os.environ.get("FOUNDRY_EVAL_DEPLOYMENT") or next(iter(env_list("AZURE_OPENAI_DEPLOYMENTS")), "")
    if not deployment:
        log("Foundry step skipped: set FOUNDRY_EVAL_DEPLOYMENT (any deployment in the project; it is not called).")
        return 0
    if project is None:
        try:
            project, client = connect()
        except ImportError as exc:
            log("Foundry step skipped: %s. Run: pip install -r requirements.txt" % exc)
            return 0

    log("Evaluators:")
    versions = ensure_evaluators(project, log)
    eval_obj = ensure_standing_eval(client, versions, dataset_sha(notes), deployment, log)
    existing = latest_runs(client, eval_obj.id)
    commit = commit if commit is not None else git_commit()

    created, entries = [], []
    for target, results in results_by_target.items():
        metadata = run_lineage(target, notes, results, versions_tag(versions), commit=commit)
        previous = existing.get(target)
        reusable = (
            previous is not None and not force and _meta(previous).get("outputs_sha") == metadata["outputs_sha"]
            and getattr(previous, "status", None) not in ("failed", "canceled") and not errored_count(previous)
        )
        if reusable:
            log("  run %-24s unchanged outputs; latest run %s kept (use --force to resubmit)" % (target, previous.id))
            entries.append(_record_entry(eval_obj, previous))
            continue
        if previous is not None and not force and errored_count(previous):
            log("  run %-24s latest run %s had grading errors; resubmitting" % (target, previous.id))
        run = client.evals.runs.create(
            eval_id=eval_obj.id,
            name=target,
            metadata=metadata,
            data_source={
                "type": "jsonl",
                "source": {"type": "file_content", "content": [{"item": item} for item in build_items(notes, results)]},
            },
        )
        created.append(run)
        entries.append(_record_entry(eval_obj, run, metadata))
        log("  run %-24s created %s (model %s, outputs %s, commit %s)"
            % (target, run.id, metadata["model"], metadata["outputs_sha"], metadata["git_commit"]))
    upsert_run_record(entries, record_path)

    if wait and created:
        finished = _wait_for_runs(client, eval_obj.id, created, log)
        upsert_run_record([_record_entry(eval_obj, run) for run in finished], record_path)
    return 0


def _criteria_summary(run) -> str:
    parts = []
    for result in getattr(run, "per_testing_criteria_results", None) or []:
        data = _as_dict(result)
        total = sum(int(data.get(k) or 0) for k in ("passed", "failed", "errored"))
        parts.append("%s %s/%s" % (data.get("testing_criteria"), data.get("passed"), total))
    return ", ".join(parts)


def _wait_for_runs(client, eval_id: str, runs: list, log, timeout_s: int = 900, poll_s: float = 10.0) -> list:
    pending = {run.id: run for run in runs}
    finished = []
    deadline = time.time() + timeout_s
    while pending and time.time() < deadline:
        time.sleep(poll_s)
        for run_id in list(pending):
            run = _retry(lambda: client.evals.runs.retrieve(run_id=run_id, eval_id=eval_id))
            if run.status in ("completed", "failed", "canceled"):
                errored = errored_count(run)
                note = " (%d items errored in the grading sandbox; run the foundry step again to resubmit)" % errored if errored else ""
                log("  %-24s %s: %s%s\n    %s" % (run.name, run.status, _criteria_summary(run), note, getattr(run, "report_url", "") or ""))
                finished.append(run)
                del pending[run_id]
    for run in pending.values():
        log("  %-24s still running; check the Foundry portal" % run.name)
    return finished
