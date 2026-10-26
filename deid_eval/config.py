"""Paths, .env loading and JSONL helpers shared by the CLI modules."""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
NOTES_PATH = DATA_DIR / "notes.jsonl"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = ROOT / "outputs"
PRICING_PATH = ROOT / "pricing.json"


def load_env(path: Path = ROOT / ".env") -> None:
    """Load .env into os.environ without overriding variables that are already set."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None
    if load_dotenv is not None:
        load_dotenv(path, override=False)
        return
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def env_list(name: str) -> list[str]:
    return [part.strip() for part in os.environ.get(name, "").split(",") if part.strip()]


def data_paths() -> tuple[Path, Path]:
    """(notes file, cache dir) under DEID_DATA_DIR when set, else under data/.

    Point DEID_DATA_DIR at a gitignored folder (for example data/private) for your own labeled
    notes and target outputs. Relative paths are resolved from the repository root.
    """
    value = os.environ.get("DEID_DATA_DIR")
    base = DATA_DIR if not value else (Path(value) if Path(value).is_absolute() else ROOT / value)
    return base / "notes.jsonl", base / "cache"


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path | str, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
