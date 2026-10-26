"""PHI leakage grader for de-identification output.

Standalone and standard-library only. Local scoring imports this module, and the Foundry
code-based evaluator is registered from this file's source, so both score identically.

Entry point for Foundry: ``grade(sample, item) -> float`` returns the note's strict PHI recall.
The Foundry grader sandbox statically rejects code-execution builtins, including any call named
"compile" (it flagged re.compile), so this module passes pattern strings to re functions.

Placeholders are any bracketed word such as ``[PATIENT_NAME]`` or ``[name]`` (case-insensitive),
so redaction output from other tools can be scored. Bracketed words that are part of the
original note (for example ``[sic]``) would also be treated as placeholders.
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

CATEGORIES = (
    "PATIENT_NAME",
    "PROVIDER_NAME",
    "DATE",
    "AGE_OVER_89",
    "MRN",
    "PHONE",
    "EMAIL",
    "ADDRESS",
    "DEVICE_ID",
)

_PLACEHOLDER_PATTERN = r"\[([A-Za-z][A-Za-z0-9_-]*)\]"
# Each placeholder is collapsed to one private-use character before alignment, so letters
# inside "[PATIENT_NAME]" can never align with letters of the original text.
_SENTINEL = "\ue000"
# A verbatim match must not be embedded in a longer token: "20/40" does not match inside
# "20/400" and "12/11" does not match inside "12/11/2025".
_TOKEN_CHARS = "A-Za-z0-9/"


def occurrences(needle: str, haystack: str) -> list[tuple[int, int]]:
    """Return (start, end) of each verbatim, token-bounded occurrence of needle in haystack."""
    if not needle:
        return []
    pattern = "(?<![%s])%s(?![%s])" % (_TOKEN_CHARS, re.escape(needle), _TOKEN_CHARS)
    return [(m.start(), m.end()) for m in re.finditer(pattern, haystack)]


def _as_list(value, field):
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError("%s must be a list or a JSON list string" % field)
    return value


def _parse_item(item):
    if not isinstance(item, dict):
        raise ValueError("item must be a dict")
    source = item.get("source_text")
    deid = item.get("deid_text")
    if not isinstance(source, str) or not isinstance(deid, str):
        raise ValueError("source_text and deid_text must be strings")
    gold = []
    for raw in _as_list(item.get("phi_spans"), "phi_spans"):
        start, end = int(raw["start"]), int(raw["end"])
        text, category = str(raw["text"]), str(raw["category"])
        if not 0 <= start < end <= len(source) or source[start:end] != text:
            raise ValueError("phi span offsets do not match source_text")
        gold.append({"start": start, "end": end, "category": category, "text": text})
    protected_raw = item.get("protected_values")
    protected = [] if protected_raw is None else [str(v) for v in _as_list(protected_raw, "protected_values")]
    return source, deid, gold, protected


def _mask_placeholders(text):
    """Collapse each placeholder to one sentinel; also return each masked char's original offset."""
    parts, starts, pos = [], [], 0
    for match in re.finditer(_PLACEHOLDER_PATTERN, text):
        parts.append(text[pos:match.start()])
        starts.extend(range(pos, match.start()))
        parts.append(_SENTINEL)
        starts.append(match.start())
        pos = match.end()
    parts.append(text[pos:])
    starts.extend(range(pos, len(text)))
    starts.append(len(text))
    return "".join(parts), starts


def align(source: str, deid: str) -> list[tuple[str, int, int, int, int]]:
    """Character alignment of source to output as SequenceMatcher opcodes.

    Each placeholder aligns as a single unit. Output offsets (j1, j2) are in the original
    ``deid`` coordinates, so ``deid[j1:j2]`` is the output text for each opcode.
    """
    masked, starts = _mask_placeholders(deid)
    matcher = SequenceMatcher(None, source, masked, autojunk=False)
    return [(tag, i1, i2, starts[j1], starts[j2]) for tag, i1, i2, j1, j2 in matcher.get_opcodes()]


def predicted_spans(source: str, deid: str) -> list[dict]:
    """Source ranges that the de-ID output replaced or deleted, from character alignment."""
    spans = []
    for tag, i1, i2, j1, j2 in align(source, deid):
        # Whitespace-only edits carry no information and are not predictions.
        if tag not in ("replace", "delete") or not source[i1:i2].strip():
            continue
        label = re.search(_PLACEHOLDER_PATTERN, deid[j1:j2])
        spans.append({"start": i1, "end": i2, "text": source[i1:i2], "label": label.group(1) if label else None})
    return spans


def _ratio(num, den):
    return num / den if den else 1.0


def _f1(precision, recall):
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def score_note(item: dict) -> dict:
    """Score one de-identified note against its gold PHI spans.

    Raises ValueError for malformed items. An empty ``deid_text`` is treated as a failed
    output: every gold span counts as leaked and clinical preservation is 0. A gold span is
    caught when every letter and digit in it was replaced (punctuation and spaces carry no
    identifying information, so `([PHONE])` or `[STREET], [CITY]` still count) and its text no
    longer appears in the output. Each leaked span carries a ``reason``: ``missed`` (nothing
    replaced), ``partial`` (only some characters replaced), ``repeat`` (fully replaced, but its
    text still appears in the output) or ``failed`` (empty output).
    """
    source, deid, gold, protected = _parse_item(item)
    failed = not deid.strip()
    preds = [] if failed else predicted_spans(source, deid)
    covered = set()
    for pred in preds:
        covered.update(range(pred["start"], pred["end"]))

    by_category: dict[str, dict[str, int]] = {}

    def bucket(category):
        return by_category.setdefault(category, {"tp": 0, "fn": 0, "fp": 0})

    leaked = []
    for span in gold:
        needed = [i for i in range(span["start"], span["end"]) if source[i].isalnum()]
        hit = sum(1 for i in needed if i in covered)
        if failed:
            reason = "failed"
        elif needed and hit == 0:
            reason = "missed"
        elif hit < len(needed):
            reason = "partial"
        elif occurrences(span["text"], deid):
            reason = "repeat"
        else:
            reason = None
        bucket(span["category"])["tp" if reason is None else "fn"] += 1
        if reason is not None:
            leaked.append(dict(span, reason=reason))

    false_positives = [
        pred
        for pred in preds
        if not any(pred["start"] < g["end"] and g["start"] < pred["end"] for g in gold)
    ]
    for pred in false_positives:
        label = (pred["label"] or "").upper()
        bucket(label if label in CATEGORIES else "OTHER")["fp"] += 1

    tp = len(gold) - len(leaked)
    fn = len(leaked)
    fp = len(false_positives)
    recall = _ratio(tp, tp + fn)
    precision = _ratio(tp, tp + fp)
    if failed:
        preservation = 0.0
    else:
        kept = sum(1 for value in protected if occurrences(value, deid))
        preservation = _ratio(kept, len(protected))
    return {
        "failed": failed,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "recall": recall,
        "precision": precision,
        "f1": _f1(precision, recall),
        "residual_phi": fn > 0,
        "clinical_preservation": preservation,
        "by_category": by_category,
        "leaked": leaked,
        "false_positives": false_positives,
    }


def _entry(item, key):
    try:
        result = score_note(item)
        if result["failed"]:
            return 0.0
        return float(min(1.0, max(0.0, result[key])))
    except Exception:
        return 0.0


def grade(sample: dict, item: dict) -> float:
    """Foundry code-evaluator entry point: the note's strict PHI recall in [0.0, 1.0].

    Reads source_text, deid_text, phi_spans and protected_values from ``item``. phi_spans and
    protected_values may be lists or JSON strings. Returns 1.0 when the note has no gold spans
    and 0.0 for malformed or empty output. Never raises.
    """
    return _entry(item, "recall")


def grade_precision(sample: dict, item: dict) -> float:
    """The note's precision: caught spans / (caught spans + over-redactions); 1.0 when neither."""
    return _entry(item, "precision")


def grade_preservation(sample: dict, item: dict) -> float:
    """The note's clinical preservation: share of protected values kept; 1.0 when there are none."""
    return _entry(item, "clinical_preservation")
