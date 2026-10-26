"""Regression gate: compare a candidate's latest Foundry run against a baseline's, per criterion.

Foundry is the system of record: the gate reads both runs and their per-note results back from
the standing eval, so it needs only credentials and the dataset (it runs on a fresh CI runner).
Exit codes: 0 pass, 1 regression, 2 cannot decide. Every outcome writes a decision record.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from . import foundry
from .config import OUTPUT_DIR
from .lineage import dataset_sha
from .report import DISCLAIMER

GATE_DIR = OUTPUT_DIR / "gate"
PASS, REGRESSION, INCONCLUSIVE = 0, 1, 2
_DECISIONS = {PASS: "pass", REGRESSION: "regression", INCONCLUSIVE: "cannot decide"}


class Inconclusive(Exception):
    pass


def _run_summary(run, eval_obj) -> dict:
    created = getattr(run, "created_at", None)
    return {
        "target": foundry._meta(run).get("target"),
        "run_id": run.id,
        "eval_id": eval_obj.id,
        "status": getattr(run, "status", None),
        "report_url": getattr(run, "report_url", None),
        "created_at": datetime.fromtimestamp(created, timezone.utc).isoformat() if isinstance(created, (int, float)) else created,
        "created_by": getattr(run, "created_by", None),
        "lineage": foundry._meta(run),
    }


def per_note_results(client, eval_id: str, run) -> dict[str, dict[str, bool]]:
    """note id -> {criterion: passed}. Raises Inconclusive for errored or missing results."""
    errored = foundry.errored_count(run)
    if errored:
        raise Inconclusive("run %s has %s items that errored in the grading sandbox; run `python -m deid_eval foundry` "
                           "again to resubmit it" % (run.id, errored))
    notes: dict[str, dict[str, bool]] = {}
    for raw in foundry._retry(lambda: list(client.evals.runs.output_items.list(run_id=run.id, eval_id=eval_id))):
        item = foundry._as_dict(raw)
        note_id = (item.get("datasource_item") or {}).get("id")
        if not note_id:
            raise Inconclusive("run %s has an output item without a note id" % run.id)
        outcome = {}
        for result in item.get("results") or []:
            result = foundry._as_dict(result)
            error = (result.get("sample") or {}).get("error") if isinstance(result.get("sample"), dict) else None
            if result.get("status") != "completed" or error or result.get("passed") is None:
                raise Inconclusive("run %s, note %s: %s did not grade cleanly" % (run.id, note_id, result.get("name")))
            outcome[result["name"]] = bool(result["passed"])
        missing = set(foundry.CRITERIA) - set(outcome)
        if missing:
            raise Inconclusive("run %s, note %s: no result for %s" % (run.id, note_id, ", ".join(sorted(missing))))
        notes[note_id] = outcome
    if not notes:
        raise Inconclusive("run %s has no output items" % run.id)
    return notes


def compare(baseline: dict[str, dict[str, bool]], candidate: dict[str, dict[str, bool]], tolerance_pp: float,
            allow_new_leaks: bool = False) -> list[dict]:
    if set(baseline) != set(candidate):
        raise Inconclusive("the two runs cover different notes; were they made from the same dataset?")
    ids = sorted(baseline)
    rows = []
    for criterion in foundry.CRITERIA:
        base_rate = sum(baseline[n][criterion] for n in ids) / len(ids)
        cand_rate = sum(candidate[n][criterion] for n in ids) / len(ids)
        delta_pp = (cand_rate - base_rate) * 100
        regressed_notes = [n for n in ids if baseline[n][criterion] and not candidate[n][criterion]]
        rate_drop = delta_pp < -tolerance_pp - 1e-9
        # A note that newly leaks PHI is a regression even if other notes improved enough to hide it.
        strict = criterion == "phi_recall" and not allow_new_leaks
        new_leaks = strict and bool(regressed_notes)
        rows.append({
            "criterion": criterion,
            "rule": "no new leaks, pass rate within tolerance" if strict else "pass rate within tolerance",
            "baseline_pass_rate": base_rate,
            "candidate_pass_rate": cand_rate,
            "delta_pp": round(delta_pp, 2),
            "regressed": rate_drop or new_leaks,
            "regressed_by": [why for why, hit in (("pass rate", rate_drop), ("new leaks", new_leaks)) if hit],
            "regressed_notes": regressed_notes,
            "improved_notes": [n for n in ids if candidate[n][criterion] and not baseline[n][criterion]],
            "notes": len(ids),
        })
    return rows


def _regression_reason(rows: list[dict]) -> str:
    parts = []
    for row in rows:
        if row["regressed"]:
            detail = []
            if "pass rate" in row["regressed_by"]:
                detail.append("pass rate %+.1f pp" % row["delta_pp"])
            if "new leaks" in row["regressed_by"]:
                detail.append("%d notes newly leak PHI" % len(row["regressed_notes"]))
            parts.append("%s (%s)" % (row["criterion"], ", ".join(detail)))
    return "regressed on " + "; ".join(parts)


def _new_decision(baseline: str, candidate: str, notes: list[dict], tolerance_pp: float,
                  allow_new_leaks: bool = False) -> dict:
    return {
        "baseline_target": baseline,
        "candidate_target": candidate,
        "tolerance_pp": tolerance_pp,
        "allow_new_leaks": allow_new_leaks,
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_sha": dataset_sha(notes),
        "disclaimer": DISCLAIMER,
        "criteria": [],
        "eval": None,
        "baseline": None,
        "candidate": None,
    }


def _finish(decision: dict, exit_code: int, reason: str) -> dict:
    decision.update(exit_code=exit_code, reason=reason, decision=_DECISIONS[exit_code])
    return decision


def evaluate(baseline: str, candidate: str, notes: list[dict], project, client, tolerance_pp: float = 0.0,
             allow_new_leaks: bool = False) -> dict:
    """Gate decision as a dict (always returned; `exit_code` says pass, regression or cannot decide)."""
    decision = _new_decision(baseline, candidate, notes, tolerance_pp, allow_new_leaks)
    try:
        versions = foundry.find_evaluator_versions(project)
        if versions is None:
            raise Inconclusive("no registered evaluator versions match the local grader; run `python -m deid_eval foundry` first")
        eval_obj = foundry.find_standing_eval(client, versions, decision["dataset_sha"])
        if eval_obj is None:
            raise Inconclusive("no standing eval for dataset %s and evaluators %s; run `python -m deid_eval foundry` first"
                               % (decision["dataset_sha"], foundry.versions_tag(versions)))
        decision["eval"] = {"id": eval_obj.id, "name": eval_obj.name, "metadata": foundry._meta(eval_obj)}
        completed = foundry.latest_runs(client, eval_obj.id, completed_only=True)
        for role, target in (("baseline", baseline), ("candidate", candidate)):
            if target not in completed:
                raise Inconclusive("no completed run for %s %r in eval %s; run `python -m deid_eval foundry --wait`"
                                   % (role, target, eval_obj.id))
            decision[role] = _run_summary(completed[target], eval_obj)
        base = per_note_results(client, eval_obj.id, completed[baseline])
        cand = per_note_results(client, eval_obj.id, completed[candidate])
        decision["criteria"] = compare(base, cand, tolerance_pp, allow_new_leaks)
        if any(row["regressed"] for row in decision["criteria"]):
            return _finish(decision, REGRESSION, _regression_reason(decision["criteria"]))
        return _finish(decision, PASS, "no criterion regressed")
    except Inconclusive as exc:
        return _finish(decision, INCONCLUSIVE, str(exc))


def _pct(value) -> str:
    return "n/a" if value is None else "%.1f%%" % (value * 100)


def render_markdown(decision: dict) -> str:
    base, cand = decision["baseline_target"], decision["candidate_target"]
    badge = {"pass": "PASS", "regression": "REGRESSION", "cannot decide": "CANNOT DECIDE"}[decision["decision"]]
    lines = [
        "# De-ID quality gate: %s vs %s" % (cand, base),
        "",
        "> **%s**" % decision["disclaimer"],
        "",
        "**Decision: %s** (exit code %d). %s." % (badge, decision["exit_code"], decision["reason"][:1].upper() + decision["reason"][1:]),
        "",
        "Evaluated %s with a tolerance of %s percentage points; %s. Dataset `%s`." % (
            decision["evaluated_at"], decision["tolerance_pp"],
            "phi_recall uses pass rate only (--allow-new-leaks)" if decision.get("allow_new_leaks")
            else "phi_recall also fails on any note that newly leaks PHI", decision["dataset_sha"]),
    ]
    if decision["eval"]:
        lines.append("Standing eval `%s` (`%s`), evaluators `%s`." % (
            decision["eval"]["name"], decision["eval"]["id"], decision["eval"]["metadata"].get("evaluator_versions", "unknown")))
    if decision["criteria"]:
        lines += [
            "",
            "## Criteria (share of notes passing at threshold 1.0)",
            "",
            "| Criterion | %s (baseline) | %s (candidate) | Change | Result |" % (base, cand),
            "|---|---|---|---|---|",
        ]
        for row in decision["criteria"]:
            result = "ok"
            if row["regressed"]:
                result = "regressed (%s)" % ", ".join(
                    "%d new leaks" % len(row["regressed_notes"]) if why == "new leaks" else why for why in row["regressed_by"])
            lines.append("| %s | %s | %s | %+.1f pp | %s |" % (
                row["criterion"], _pct(row["baseline_pass_rate"]), _pct(row["candidate_pass_rate"]), row["delta_pp"], result))
        lines += ["", "## Notes that passed for the baseline and failed for the candidate", ""]
        any_regressed = False
        for row in decision["criteria"]:
            for note in row["regressed_notes"]:
                any_regressed = True
                lines.append("- %s on `%s`: [open in viewer](../viewer.html#%s/%s)" % (row["criterion"], note, note, cand))
        if not any_regressed:
            lines.append("None.")
        improved = sorted({n for row in decision["criteria"] for n in row["improved_notes"]})
        if improved:
            lines += ["", "Notes that improved on at least one criterion: %s." % ", ".join("`%s`" % n for n in improved)]
    lines += ["", "## Lineage", ""]
    if decision["baseline"] and decision["candidate"]:
        keys = ["model", "prompt_sha", "outputs_sha", "dataset_sha", "evaluator_versions", "git_commit"]
        lines += ["| | %s (baseline) | %s (candidate) |" % (base, cand), "|---|---|---|"]
        lines.append("| run | [%s](%s) | [%s](%s) |" % (
            decision["baseline"]["run_id"], decision["baseline"]["report_url"] or "",
            decision["candidate"]["run_id"], decision["candidate"]["report_url"] or ""))
        for key in ("created_at", "created_by"):
            lines.append("| %s | %s | %s |" % (key, decision["baseline"][key], decision["candidate"][key]))
        for key in keys:
            lines.append("| %s | `%s` | `%s` |" % (key, decision["baseline"]["lineage"].get(key, "unknown"),
                                                  decision["candidate"]["lineage"].get(key, "unknown")))
    else:
        lines.append("Not available: %s." % decision["reason"])
    return "\n".join(lines) + "\n"


def write_record(decision: dict, out_dir: Path = GATE_DIR) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = "%s-%s-vs-%s" % (stamp, decision["candidate_target"], decision["baseline_target"])
    stem = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in stem)
    # Decision records are evidence: never overwrite one, even for two gates in the same second.
    unique, counter = stem, 2
    while (out_dir / (unique + ".md")).exists() or (out_dir / (unique + ".json")).exists():
        unique, counter = "%s-%d" % (stem, counter), counter + 1
    md_path, json_path = out_dir / (unique + ".md"), out_dir / (unique + ".json")
    md_path.write_text(render_markdown(decision), encoding="utf-8")
    json_path.write_text(json.dumps(decision, indent=1) + "\n", encoding="utf-8")
    return md_path, json_path


def run_gate(baseline: str, candidate: str, notes: list[dict], tolerance_pp: float = 0.0, project=None, client=None,
             out_dir: Path = GATE_DIR, log=print, allow_new_leaks: bool = False) -> int:
    reason = None
    if project is None:
        reason = foundry.not_configured_reason()
        if reason is None:
            try:
                project, client = foundry.connect()
            except ImportError as exc:
                reason = "%s; run: pip install -r requirements.txt" % exc
    if reason:
        decision = _finish(_new_decision(baseline, candidate, notes, tolerance_pp, allow_new_leaks), INCONCLUSIVE,
                           "Foundry is not configured (%s)" % reason)
    else:
        decision = evaluate(baseline, candidate, notes, project, client, tolerance_pp, allow_new_leaks)

    md_path, json_path = write_record(decision, out_dir)
    log("Gate %s vs %s: %s (exit %d). %s" % (candidate, baseline, decision["decision"].upper(), decision["exit_code"], decision["reason"]))
    for row in decision["criteria"]:
        status = ("REGRESSED (%s)" % ", ".join(row["regressed_by"])) if row["regressed"] else "ok"
        log("  %-22s %6s -> %6s  %+6.1f pp  %s%s" % (
            row["criterion"], _pct(row["baseline_pass_rate"]), _pct(row["candidate_pass_rate"]), row["delta_pp"], status,
            ("  notes: " + ", ".join(row["regressed_notes"])) if row["regressed_notes"] else ""))
    log("Decision record: %s" % md_path)
    return decision["exit_code"]
