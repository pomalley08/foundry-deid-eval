"""Import de-identification outputs produced outside this repo as a target.

Use this to evaluate your own pipeline, Azure Health Data Services (AHDS) or a vendor tool with
the same grader, report, viewer, Foundry runs and gate. The input is JSONL with one row per note,
keyed by the note `id` in your dataset. Three formats are supported:

- ``text``: ``{"id", "deid_text"}``, the de-identified note as your pipeline produced it.
- ``spans``: ``{"id", "spans": [{"start", "end", "category"}]}``, detected PHI as character
  offsets into the source note (Python string indexes, i.e. Unicode code points). Each span is
  replaced with ``[CATEGORY]`` to build the de-identified text.
- ``ahds-tag``: ``{"id", "taggerResult": {"entities": [...]}}`` (or ``"entities"`` at the top
  level), the response of the AHDS de-identification ``Tag`` operation. Entity offsets are read
  from ``offset.codePoint`` and ``length.codePoint``.

Any row may also carry ``model``, ``latency_ms``, ``input_tokens``, ``output_tokens``,
``prompt_sha`` and ``error``.
"""

from __future__ import annotations

from pathlib import Path

from .config import read_jsonl
from .targets import make_result

# AHDS Tag categories -> placeholder labels. Labels only affect display and over-redaction
# bucketing; recall does not depend on them. Unlisted categories are upper-cased.
AHDS_CATEGORY_MAP = {
    "Patient": "PATIENT_NAME",
    "Doctor": "PROVIDER_NAME",
    "Date": "DATE",
    "Age": "AGE",
    "MedicalRecord": "MRN",
    "Phone": "PHONE",
    "Fax": "PHONE",
    "Email": "EMAIL",
    "Street": "ADDRESS",
    "City": "ADDRESS",
    "State": "ADDRESS",
    "Zip": "ADDRESS",
    "LocationOther": "ADDRESS",
    "Device": "DEVICE_ID",
}


def apply_spans(source: str, spans: list[dict]) -> str:
    """Replace each span with [CATEGORY]; overlapping or touching spans are merged."""
    ordered = []
    for span in spans:
        start, end = int(span["start"]), int(span["end"])
        if not 0 <= start < end <= len(source):
            raise ValueError("span %s-%s is outside the note (length %d)" % (start, end, len(source)))
        ordered.append([start, end, str(span.get("category") or "PHI").upper()])
    ordered.sort()
    merged: list[list] = []
    for start, end, category in ordered:
        if merged and start < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end, category])
    out, pos = [], 0
    for start, end, category in merged:
        out.append(source[pos:start])
        out.append("[%s]" % category)
        pos = end
    out.append(source[pos:])
    return "".join(out)


def _code_point(value, field: str) -> int:
    if isinstance(value, dict):
        for key in ("codePoint", "code_point"):
            if key in value:
                return int(value[key])
        raise ValueError("entity %s has no codePoint value" % field)
    return int(value)


def ahds_tag_spans(row: dict) -> list[dict]:
    tagger = row.get("taggerResult") or row.get("tagger_result") or {}
    entities = tagger.get("entities") if isinstance(tagger, dict) else None
    if entities is None:
        entities = row.get("entities")
    if entities is None:
        raise ValueError("row %s has no taggerResult.entities" % row.get("id"))
    spans = []
    for entity in entities:
        start = _code_point(entity["offset"], "offset")
        length = _code_point(entity["length"], "length")
        category = str(entity.get("category", "PHI"))
        spans.append({"start": start, "end": start + length, "category": AHDS_CATEGORY_MAP.get(category, category.upper())})
    return spans


def _deid_text(row: dict, source: str, fmt: str) -> str:
    if fmt == "text":
        text = row.get("deid_text")
        if not isinstance(text, str):
            raise ValueError("row %s has no deid_text string" % row.get("id"))
        return text
    if fmt == "spans":
        if not isinstance(row.get("spans"), list):
            raise ValueError("row %s has no spans list" % row.get("id"))
        return apply_spans(source, row["spans"])
    if fmt == "ahds-tag":
        return apply_spans(source, ahds_tag_spans(row))
    raise ValueError("unknown format %r" % fmt)


def import_outputs(notes: list[dict], path: Path, target: str, fmt: str = "text", model: str | None = None,
                   allow_missing: bool = False) -> list[dict]:
    """Build cache rows (one per dataset note, in dataset order) from an external outputs file."""
    rows = read_jsonl(Path(path))
    by_id: dict[str, dict] = {}
    for row in rows:
        note_id = row.get("id")
        if not note_id:
            raise ValueError("every row needs an `id` matching a note in the dataset")
        if note_id in by_id:
            raise ValueError("note %s appears more than once" % note_id)
        by_id[note_id] = row
    known = {note["id"] for note in notes}
    unknown = sorted(set(by_id) - known)
    if unknown:
        raise ValueError("%d ids are not in the dataset, e.g. %s" % (len(unknown), ", ".join(unknown[:5])))
    missing = [note["id"] for note in notes if note["id"] not in by_id]
    if missing and not allow_missing:
        raise ValueError("%d dataset notes have no row, e.g. %s (use --allow-missing to record them as errors)"
                         % (len(missing), ", ".join(missing[:5])))

    results = []
    for note in notes:
        row = by_id.get(note["id"])
        if row is None:
            results.append(make_result(note, target, model or target, error="missing from import file"))
            continue
        label = model or row.get("model") or target
        if row.get("error"):
            results.append(make_result(note, target, label, error=str(row["error"]), prompt_sha=row.get("prompt_sha")))
            continue
        results.append(make_result(
            note, target, label,
            deid_text=_deid_text(row, note["source_text"], fmt),
            latency_ms=row.get("latency_ms"),
            input_tokens=row.get("input_tokens"),
            output_tokens=row.get("output_tokens"),
            prompt_sha=row.get("prompt_sha"),
        ))
    return results
