"""CLI: python -m deid_eval {generate,run,import,report,view,foundry,gate}."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from . import dataset, report, targets, viewer
from .config import CACHE_DIR, NOTES_PATH, OUTPUT_DIR, PRICING_PATH, data_paths, load_env, read_jsonl


def _load_notes(generate_if_missing: bool = True) -> list[dict]:
    if not NOTES_PATH.exists():
        if not generate_if_missing:
            sys.exit("No dataset at %s. Run: python -m deid_eval generate" % NOTES_PATH)
        print("Dataset missing; generating %d notes (seed %d)." % (dataset.DEFAULT_SIZE, dataset.DEFAULT_SEED))
        dataset.write_dataset(NOTES_PATH, dataset.generate())
    return read_jsonl(NOTES_PATH)


def _foundry_runs() -> dict:
    from .foundry import latest_recorded_runs

    return latest_recorded_runs(OUTPUT_DIR / "foundry_runs.json")


def _write_report(notes, results_by_target) -> None:
    if not results_by_target:
        print("No target results to report.")
        return
    pricing = report.load_pricing(PRICING_PATH)
    ranked = report.build_report(notes, results_by_target, pricing, OUTPUT_DIR)
    viewer_path = viewer.write_viewer(OUTPUT_DIR / "viewer.html", notes, results_by_target, pricing, _foundry_runs())
    print()
    print(report.DISCLAIMER)
    print()
    print(report.console_table(ranked))
    print()
    print("Wrote %s and %s" % (OUTPUT_DIR / "report.md", OUTPUT_DIR / "report.csv"))
    print("Note viewer: %s" % viewer_path)


def cmd_generate(args) -> int:
    records = dataset.generate(seed=args.seed, size=args.size)
    dataset.write_dataset(args.out, records)
    spans = sum(len(r["phi_spans"]) for r in records)
    print("Wrote %d synthetic notes with %d gold PHI spans to %s" % (len(records), spans, args.out))
    return 0


def _selected_targets(args) -> list[targets.Target]:
    selected: list[targets.Target] = [targets.BaselineTarget()]
    models, message = targets.model_targets(concurrency=args.concurrency)
    if message and not args.use_cache:
        print(message)
    selected += models
    if args.targets:
        wanted = {name.strip() for name in args.targets.split(",")}
        selected = [t for t in selected if t.name in wanted]
    return selected


def cmd_run(args) -> int:
    started = time.perf_counter()
    notes = _load_notes()
    print("Running targets%s:" % (" from cache" if args.use_cache else ""))
    results = targets.collect_results(notes, _selected_targets(args), use_cache=args.use_cache, cache_dir=CACHE_DIR)
    if args.targets:
        wanted = {name.strip() for name in args.targets.split(",")}
        results = {name: rows for name, rows in results.items() if name in wanted}
    _write_report(notes, results)
    if args.foundry:
        from .foundry import run_foundry

        print()
        run_foundry(notes, results, wait=args.wait)
    print("Done in %.1fs" % (time.perf_counter() - started))
    return 0


def cmd_report(args) -> int:
    notes = _load_notes(generate_if_missing=False)
    results = targets.collect_results(notes, [], use_cache=True, cache_dir=CACHE_DIR)
    _write_report(notes, results)
    return 0


def cmd_import(args) -> int:
    from .importer import import_outputs

    notes = _load_notes(generate_if_missing=False)
    try:
        rows = import_outputs(notes, args.file, args.target, fmt=args.format, model=args.model,
                              allow_missing=args.allow_missing)
    except ValueError as exc:
        print("Import failed: %s" % exc)
        return 1
    path = targets.save_cache(rows, args.target, CACHE_DIR)
    errors = sum(1 for row in rows if row["error"])
    print("Imported %d notes for target %r into %s (%d recorded as errors)." % (len(rows), args.target, path, errors))
    print("Next: python -m deid_eval report   (or: view, foundry, gate)")
    return 0


def cmd_foundry(args) -> int:
    from .foundry import run_foundry

    notes = _load_notes(generate_if_missing=False)
    results = targets.collect_results(notes, [], use_cache=True, cache_dir=CACHE_DIR)
    if args.targets:
        wanted = {name.strip() for name in args.targets.split(",")}
        results = {name: rows for name, rows in results.items() if name in wanted}
    return run_foundry(notes, results, wait=args.wait, force=args.force)


def cmd_gate(args) -> int:
    from .gate import run_gate

    notes = _load_notes(generate_if_missing=False)
    return run_gate(args.baseline, args.candidate, notes, tolerance_pp=args.tolerance, allow_new_leaks=args.allow_new_leaks)


def cmd_view(args) -> int:
    notes = _load_notes(generate_if_missing=False)
    results = targets.collect_results(notes, [], use_cache=True, cache_dir=CACHE_DIR, log=lambda *_: None)
    if not results:
        print("No cached target results in %s. Run: python -m deid_eval run" % CACHE_DIR)
        return 1
    path = viewer.write_viewer(OUTPUT_DIR / "viewer.html", notes, results, report.load_pricing(PRICING_PATH), _foundry_runs())
    print("Note viewer: %s" % path)
    if not args.no_open:
        import webbrowser

        webbrowser.open(path.resolve().as_uri())
    return 0


def main(argv=None) -> int:
    global NOTES_PATH, CACHE_DIR
    load_env()
    if os.environ.get("DEID_DATA_DIR"):
        NOTES_PATH, CACHE_DIR = data_paths()
    parser = argparse.ArgumentParser(
        prog="python -m deid_eval",
        description="Evaluate de-identification outputs for PHI leakage, over-redaction and clinical preservation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="write the synthetic sample dataset")
    p.add_argument("--seed", type=int, default=dataset.DEFAULT_SEED)
    p.add_argument("--size", type=int, default=dataset.DEFAULT_SIZE)
    p.add_argument("--out", type=Path, default=NOTES_PATH)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("run", help="generate if missing, run targets, score and report")
    p.add_argument("--use-cache", action="store_true", help="replay cached outputs instead of calling models")
    p.add_argument("--targets", help="comma-separated subset of target names")
    p.add_argument("--concurrency", type=int, default=4, help="parallel calls per model target")
    p.add_argument("--foundry", action="store_true", help="also create Foundry cloud evaluation runs")
    p.add_argument("--wait", action="store_true", help="with --foundry, wait for runs to finish")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("import", help="import outputs produced elsewhere (your pipeline, AHDS) as a target")
    p.add_argument("--target", required=True, help="name for this target in reports, the viewer and Foundry")
    p.add_argument("--file", required=True, type=Path, help="JSONL file with one row per note (see ADAPTING.md)")
    p.add_argument("--format", choices=("text", "spans", "ahds-tag"), default="text",
                   help="text: de-identified text; spans: detected spans; ahds-tag: AHDS Tag operation results")
    p.add_argument("--model", help="model or version label to record (default: the row's model, else the target name)")
    p.add_argument("--allow-missing", action="store_true", help="record notes missing from the file as errors instead of failing")
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("report", help="score cached target outputs and write the report")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("foundry", help="add runs for cached target outputs to the standing Foundry eval")
    p.add_argument("--wait", action="store_true", help="wait for runs to finish and print per-criterion pass counts")
    p.add_argument("--force", action="store_true", help="resubmit targets whose outputs are unchanged since their last run")
    p.add_argument("--targets", help="comma-separated subset of target names")
    p.set_defaults(func=cmd_foundry)

    p = sub.add_parser("gate", help="compare a candidate's latest Foundry run to a baseline's; exit 0 pass, 1 regression, 2 cannot decide")
    p.add_argument("--baseline", required=True, help="target whose latest completed run is the reference")
    p.add_argument("--candidate", required=True, help="target being evaluated for adoption")
    p.add_argument("--tolerance", type=float, default=0.0, help="allowed drop in pass rate, in percentage points (default 0)")
    p.add_argument("--allow-new-leaks", action="store_true",
                   help="judge phi_recall on pass rate only; by default any note that newly leaks PHI fails the gate")
    p.set_defaults(func=cmd_gate)

    p = sub.add_parser("view", help="rebuild the note viewer from cached outputs and open it")
    p.add_argument("--no-open", action="store_true", help="write outputs/viewer.html without opening a browser")
    p.set_defaults(func=cmd_view)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
