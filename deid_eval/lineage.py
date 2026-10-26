"""Run lineage: short hashes and identifiers that tie a Foundry run to exactly what produced it."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from .config import ROOT

UNKNOWN = "unknown"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def dataset_sha(notes: list[dict]) -> str:
    """Hash of the dataset records (canonical JSON, so line endings and key order don't matter)."""
    return _sha(json.dumps(notes, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def outputs_sha(notes: list[dict], results: list[dict]) -> str:
    """Hash of one target's de-identified texts in note order; changes whenever any output changes."""
    by_id = {row["id"]: row for row in results}
    texts = [(note["id"], (by_id.get(note["id"]) or {}).get("deid_text") or "") for note in notes]
    return _sha(json.dumps(texts, ensure_ascii=False, separators=(",", ":")))


def git_commit(root: Path = ROOT) -> str:
    """Short HEAD commit, suffixed `-dirty` when tracked or untracked files differ; else `unknown`."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN
    if not head:
        return UNKNOWN
    return head + ("-dirty" if status else "")


def _single(values, empty: str) -> str:
    distinct = sorted({str(v) for v in values if v})
    if not distinct:
        return empty
    return distinct[0] if len(distinct) == 1 else "mixed:" + ",".join(distinct)


def run_lineage(target: str, notes: list[dict], results: list[dict], evaluator_versions: str,
                commit: str | None = None) -> dict[str, str]:
    """Metadata for one Foundry run. Every value is a short string; unknown values say so."""
    is_model = any(row.get("input_tokens") is not None or row.get("prompt_sha") for row in results)
    return {
        "target": target,
        "model": _single((row.get("model") for row in results), UNKNOWN),
        # The regex baseline has no prompt; a model target without a recorded hash is unknown.
        "prompt_sha": _single((row.get("prompt_sha") for row in results), UNKNOWN if is_model else "n/a"),
        "outputs_sha": outputs_sha(notes, results),
        "dataset_sha": dataset_sha(notes),
        "evaluator_versions": evaluator_versions or UNKNOWN,
        "git_commit": commit if commit is not None else git_commit(),
    }
