"""Score target outputs and write the per-target quality, latency and cost comparison."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from .phi_grader import CATEGORIES, score_note

DISCLAIMER = (
    "Not a compliance determination. Results describe only the evaluated dataset; the bundled "
    "sample data is synthetic and too small to choose a model on its own."
)
COLUMNS = (
    ("target", "Target"),
    ("model", "Model"),
    ("notes", "Notes"),
    ("errors", "Errors"),
    ("recall", "PHI recall"),
    ("precision", "Precision"),
    ("f1", "F1"),
    ("residual_phi_rate", "Residual-PHI notes"),
    ("clinical_preservation", "Clinical preservation"),
    ("latency_p50_ms", "p50 ms"),
    ("latency_p95_ms", "p95 ms"),
    ("input_tokens", "Input tokens"),
    ("output_tokens", "Output tokens"),
    ("cost", "Est. cost (USD)"),
)


def percentile(values: list[float], pct: float):
    """Linear-interpolated percentile (pct in [0, 100]); None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100
    low, high = math.floor(rank), math.ceil(rank)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def load_pricing(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("models", {})


def find_price(pricing: dict, target: str, model: str):
    """Match a deployment name, then a model name, then the longest model-name prefix."""
    for key in (target, model):
        if key in pricing:
            return pricing[key]
    prefixes = [key for key in pricing if model and model.startswith(key)]
    return pricing[max(prefixes, key=len)] if prefixes else None


def score_items(notes: list[dict], results: list[dict]) -> list[dict]:
    by_id = {row["id"]: row for row in results}
    scored = []
    for note in notes:
        row = by_id.get(note["id"], {})
        item = {
            "source_text": note["source_text"],
            "deid_text": row.get("deid_text") or "",
            "phi_spans": note["phi_spans"],
            "protected_values": note["protected_values"],
        }
        scored.append({"note": note, "result": row, "score": score_note(item)})
    return scored


def summarize(target: str, scored: list[dict], pricing: dict) -> dict:
    scores = [s["score"] for s in scored]
    results = [s["result"] for s in scored]
    tp = sum(s["tp"] for s in scores)
    fn = sum(s["fn"] for s in scores)
    fp = sum(s["fp"] for s in scores)
    recall = tp / (tp + fn) if tp + fn else 1.0
    precision = tp / (tp + fp) if tp + fp else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    latencies = [r["latency_ms"] for r in results if r.get("latency_ms") is not None and not r.get("error")]
    models = sorted({r.get("model") for r in results if r.get("model")})
    model = ", ".join(models) if models else ""
    has_tokens = any(r.get("input_tokens") is not None for r in results)
    input_tokens = sum(r.get("input_tokens") or 0 for r in results) if has_tokens else None
    output_tokens = sum(r.get("output_tokens") or 0 for r in results) if has_tokens else None

    if not has_tokens:
        cost, cost_sort = "n/a", 0.0
    else:
        price = find_price(pricing, target, models[0] if models else "")
        if not price or price.get("input_per_1m") is None or price.get("output_per_1m") is None:
            cost, cost_sort = "unpriced", math.inf
        else:
            cost_sort = input_tokens / 1e6 * price["input_per_1m"] + output_tokens / 1e6 * price["output_per_1m"]
            cost = cost_sort

    per_category = {}
    for category in CATEGORIES:
        c_tp = sum(s["by_category"].get(category, {}).get("tp", 0) for s in scores)
        c_fn = sum(s["by_category"].get(category, {}).get("fn", 0) for s in scores)
        per_category[category] = c_tp / (c_tp + c_fn) if c_tp + c_fn else None

    return {
        "target": target,
        "model": model,
        "notes": len(scored),
        "errors": sum(1 for r in results if r.get("error") or not r),
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "residual_phi_rate": sum(1 for s in scores if s["residual_phi"]) / len(scores) if scores else 0.0,
        "clinical_preservation": sum(s["clinical_preservation"] for s in scores) / len(scores) if scores else 0.0,
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": cost,
        "_cost_sort": cost_sort,
        "_per_category": per_category,
        "_scored": scored,
    }


def rank(summaries: list[dict]) -> list[dict]:
    """Recall first (high to low), then residual-PHI note rate (low), then cost (low)."""
    return sorted(summaries, key=lambda s: (-s["recall"], s["residual_phi_rate"], s["_cost_sort"]))


def _fmt(key: str, value) -> str:
    if value is None:
        return "n/a"
    if key == "cost":
        return value if isinstance(value, str) else "$%.4f" % value
    if key in ("recall", "precision", "f1", "residual_phi_rate", "clinical_preservation"):
        return "%.1f%%" % (value * 100)
    if key in ("latency_p50_ms", "latency_p95_ms"):
        return "%.0f" % value
    if isinstance(value, int):
        return "{:,}".format(value)
    return str(value)


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _examples(summary: dict, limit: int = 3) -> list[str]:
    lines = []
    for entry in summary["_scored"]:
        for span in entry["score"]["leaked"]:
            if len(lines) < limit:
                lines.append("missed %s `%s` (%s)" % (span["category"], span["text"], entry["note"]["id"]))
    over = []
    for entry in summary["_scored"]:
        for pred in entry["score"]["false_positives"]:
            if len(over) < limit:
                label = " as [%s]" % pred["label"] if pred["label"] else ""
                over.append("replaced non-PHI `%s`%s (%s)" % (pred["text"].strip(), label, entry["note"]["id"]))
    return lines + over


def render_markdown(ranked: list[dict], notes: list[dict]) -> str:
    gold_counts = {c: sum(1 for n in notes for s in n["phi_spans"] if s["category"] == c) for c in CATEGORIES}
    types = sorted({n.get("note_type", "note") for n in notes})
    parts = [
        "# De-ID model comparison",
        "",
        "> **%s**" % DISCLAIMER,
        "",
        "Generated %s from %d notes (%s) with %d gold PHI spans." % (
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            len(notes),
            ", ".join(types),
            sum(gold_counts.values()),
        ),
        "",
        "## Ranking (recall first, then residual-PHI note rate, then cost)",
        "",
        _md_table([label for _, label in COLUMNS], [[_fmt(k, s[k]) for k, _ in COLUMNS] for s in ranked]),
        "",
        "## Recall by PHI category",
        "",
        _md_table(
            ["Category", "Gold spans"] + [s["target"] for s in ranked],
            [
                [c, str(gold_counts[c])] + [_fmt("recall", s["_per_category"][c]) for s in ranked]
                for c in CATEGORIES
            ],
        ),
        "",
        "## Examples",
        "",
    ]
    for summary in ranked:
        examples = _examples(summary)
        parts.append("- **%s**: %s" % (summary["target"], "; ".join(examples) if examples else "no misses or over-redactions"))
    parts += [
        "",
        "## How to read this",
        "",
        "- **PHI recall** is pooled over every gold span. A span counts as caught only when the whole span is "
        "replaced and its text no longer appears anywhere in the output, so partial redaction and missed "
        "repeat mentions count as leaks.",
        "- **Residual-PHI notes** is the share of notes with at least one leaked span. The Foundry evaluator "
        "scores each note's own recall; with pass threshold 1.0, the Foundry pass rate equals 1 minus this rate.",
        "- **Precision** penalizes replacing non-PHI text, such as visual acuity read as a date. "
        "**Clinical preservation** is the share of protected clinical values (acuity, IOP, laterality, "
        "medications, eponyms) kept verbatim.",
        "- **Est. cost** uses list-price estimates from `pricing.json`. `unpriced` means no price is set; "
        "`n/a` means the target uses no tokens.",
        "",
    ]
    return "\n".join(parts)


def write_csv(path: Path, ranked: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["# " + DISCLAIMER])
        writer.writerow([key for key, _ in COLUMNS] + ["recall_%s" % c for c in CATEGORIES])
        for s in ranked:
            row = []
            for key, _ in COLUMNS:
                value = s[key]
                row.append("" if value is None else (round(value, 6) if isinstance(value, float) else value))
            row += ["" if s["_per_category"][c] is None else round(s["_per_category"][c], 6) for c in CATEGORIES]
            writer.writerow(row)


def console_table(ranked: list[dict]) -> str:
    keys = ("target", "recall", "precision", "residual_phi_rate", "clinical_preservation",
            "latency_p50_ms", "latency_p95_ms", "cost", "errors")
    headers = ("Target", "Recall", "Precision", "Resid.notes", "Clin.pres", "p50 ms", "p95 ms", "Est.cost", "Errors")
    rows = [[_fmt(k, s[k]) for k in keys] for s in ranked]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    line = lambda cells: "  ".join(c.ljust(w) for c, w in zip(cells, widths))
    return "\n".join([line(headers), line(["-" * w for w in widths])] + [line(r) for r in rows])


def build_report(notes: list[dict], results_by_target: dict[str, list[dict]], pricing: dict,
                 output_dir: Path) -> list[dict]:
    """Score every target, write outputs/report.md and outputs/report.csv, return ranked rows."""
    summaries = [summarize(name, score_items(notes, rows), pricing) for name, rows in results_by_target.items()]
    ranked = rank(summaries)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.md").write_text(render_markdown(ranked, notes), encoding="utf-8")
    write_csv(output_dir / "report.csv", ranked)
    return ranked
