"""De-ID targets: a regex baseline and Azure OpenAI / Foundry chat deployments.

Every target returns the same result shape per note: deid_text, latency_ms, input_tokens,
output_tokens, target and model (plus the note id, an error field and a source hash used to
detect stale cache files).
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import CACHE_DIR, env_list, read_jsonl, write_jsonl

DEID_PROMPT = """You are a clinical de-identification engine. Replace protected health information (PHI) in the ophthalmology note from the user with typed placeholders, then return the note.

Replace each PHI item with exactly one placeholder:
[PATIENT_NAME] - patient names, including partial names and DICOM-style names such as LAST^FIRST
[PROVIDER_NAME] - names of clinicians, technicians and other staff
[DATE] - any date in any format, including dates of birth and partial dates
[AGE_OVER_89] - ages of 90 or older (ages 89 and under are not PHI)
[MRN] - medical record numbers and patient IDs
[PHONE] - phone and fax numbers
[EMAIL] - email addresses
[ADDRESS] - street addresses, including city, state and ZIP code
[DEVICE_ID] - device serial numbers

Do not replace clinical content, even when it resembles PHI: visual acuity (for example 20/40), intraocular pressure (for example 16/18 mmHg), laterality (OD, OS, OU), medication names and doses, eponymous disease and test names, credentials (MD, OD), organization names and device model names.

Preserve every non-PHI character exactly, including line breaks, spacing and punctuation. Return only the de-identified note, with no commentary and no code fences."""


def source_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


PROMPT_SHA = prompt_hash(DEID_PROMPT)


def make_result(note: dict, target: str, model: str, deid_text: str = "", latency_ms=None,
                input_tokens=None, output_tokens=None, error=None, temperature=None,
                prompt_sha=None) -> dict:
    return {
        "id": note["id"],
        "target": target,
        "model": model,
        "deid_text": deid_text,
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "temperature": temperature,
        "prompt_sha": prompt_sha,
        "error": error,
        "source_sha1": source_hash(note["source_text"]),
    }


class Target:
    name = "target"
    model = "unknown"
    concurrency = 1
    prompt_sha = None

    def deidentify(self, text: str) -> dict:
        """Return a dict with deid_text and optionally model, input_tokens, output_tokens."""
        raise NotImplementedError


# --- Regex baseline ------------------------------------------------------------------------

_MONTH_NAMES = "January|February|March|April|May|June|July|August|September|October|November|December"
_NAME = r"[A-Z][a-z]+(?:-[A-Z][a-z]+)?"

# (category, pattern). Group 1, when present, is the part that gets replaced.
_BASELINE_RULES = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("PHONE", re.compile(r"(?:\(\d{3}\)\s?|\b\d{3}[-.])\d{3}[-.]\d{4}\b")),
    ("DATE", re.compile(r"\b\d{4}-\d{2}-\d{2}\b")),
    ("DATE", re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")),
    ("DATE", re.compile(r"\b(?:%s)\s+\d{1,2},\s+\d{4}\b" % _MONTH_NAMES)),
    # Naive partial-date rule (mm/dd). Common in regex de-ID, and it over-redacts visual acuity
    # and IOP; the evaluator's precision and clinical-preservation metrics expose this.
    ("DATE", re.compile(r"\b\d{1,2}/\d{1,2}\b")),
    ("ADDRESS", re.compile(
        r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s+){1,3}(?:Street|St|Road|Rd|Lane|Ln|Drive|Avenue|Ave|Court|Ct|Way|Boulevard|Blvd)\.?"
        r"(?:,\s*Suite\s+\d+)?(?:,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2}\s+\d{5})?"
    )),
    ("MRN", re.compile(r"\b(?:MRN|ID)\s*[:#]?\s*(\d{6,10})\b")),
    ("DEVICE_ID", re.compile(r"\b(?:S/N|SN|Serial)\s*[:#]?\s*([A-Z0-9]+-?[0-9]{4,})")),
    ("AGE_OVER_89", re.compile(r"\b(9\d|1[01]\d)(?=-year-old|\s*(?:yo|y/o)\b)")),
    ("AGE_OVER_89", re.compile(r"\bAge:\s*(9\d|1[01]\d)\b")),
    ("PROVIDER_NAME", re.compile(r"\bDr\.\s+((?:%s\s+)?%s)" % (_NAME, _NAME))),
    ("PROVIDER_NAME", re.compile(r"\b((?:%s\s+)?%s),\s+(?:MD|DO|OD)\b" % (_NAME, _NAME))),
    ("PATIENT_NAME", re.compile(r"\b(?:Patient|Pt|RE):\s+(%s(?: %s)*)" % (_NAME, _NAME))),
    ("PATIENT_NAME", re.compile(r"\b(?:Mr|Ms|Mrs)\.\s+(%s)" % _NAME)),
]


def baseline_deidentify(text: str) -> str:
    """Replace regex matches with typed placeholders. Earlier rules win on overlap."""
    taken: list[tuple[int, int, str]] = []
    for category, pattern in _BASELINE_RULES:
        for match in pattern.finditer(text):
            group = 1 if pattern.groups else 0
            start, end = match.span(group)
            if start < end and not any(start < b and a < end for a, b, _ in taken):
                taken.append((start, end, category))
    out = text
    for start, end, category in sorted(taken, reverse=True):
        out = out[:start] + "[%s]" % category + out[end:]
    return out


class BaselineTarget(Target):
    name = "baseline"
    model = "regex-v1"

    def deidentify(self, text: str) -> dict:
        return {"deid_text": baseline_deidentify(text)}


# --- Model deployment targets ----------------------------------------------------------------

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n(.*)\n```$", re.S)


class ModelTarget(Target):
    """One Azure OpenAI / Foundry chat deployment, called with the shared de-ID prompt."""

    prompt_sha = PROMPT_SHA

    def __init__(self, client, deployment: str, concurrency: int = 4):
        self.client = client
        self.name = deployment
        self.model = deployment
        self.concurrency = concurrency
        self._temperature = 0

    def _create(self, text: str, temperature):
        kwargs = {
            "model": self.name,
            "messages": [
                {"role": "system", "content": DEID_PROMPT},
                {"role": "user", "content": text},
            ],
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        return self.client.chat.completions.create(**kwargs)

    def deidentify(self, text: str) -> dict:
        temperature = self._temperature
        try:
            started = time.perf_counter()
            response = self._create(text, temperature)
        except Exception as exc:
            # Reasoning models reject temperature; retry without it and stop sending it.
            if temperature is None or "temperature" not in str(exc):
                raise
            self._temperature = temperature = None
            started = time.perf_counter()
            response = self._create(text, None)
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        content = (response.choices[0].message.content or "").strip("\n")
        fenced = _FENCE_RE.match(content.strip())
        usage = getattr(response, "usage", None)
        return {
            "deid_text": fenced.group(1) if fenced else content,
            "model": getattr(response, "model", None) or self.name,
            "latency_ms": latency_ms,
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
            "temperature": temperature,
        }


def _openai_client():
    from openai import AzureOpenAI

    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if api_key:
        return AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )
    return AzureOpenAI(azure_endpoint=endpoint, azure_ad_token_provider=token_provider, api_version=api_version)


def model_targets(concurrency: int = 4) -> tuple[list[Target], str | None]:
    """Build one target per configured deployment, or return a skip message."""
    deployments = env_list("AZURE_OPENAI_DEPLOYMENTS")
    if not os.environ.get("AZURE_OPENAI_ENDPOINT") or not deployments:
        return [], "Model targets skipped: set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENTS in .env."
    try:
        client = _openai_client()
    except ImportError as exc:
        return [], "Model targets skipped: %s. Run: pip install -r requirements.txt" % exc
    return [ModelTarget(client, deployment, concurrency) for deployment in deployments], None


# --- Running and caching ---------------------------------------------------------------------


def _run_one(target: Target, note: dict) -> dict:
    started = time.perf_counter()
    try:
        output = target.deidentify(note["source_text"])
    except Exception as exc:
        return make_result(note, target.name, target.model, error="%s: %s" % (type(exc).__name__, exc),
                           prompt_sha=target.prompt_sha)
    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    return make_result(
        note,
        target.name,
        output.get("model") or target.model,
        deid_text=output.get("deid_text") or "",
        latency_ms=output.get("latency_ms", latency_ms),
        input_tokens=output.get("input_tokens"),
        output_tokens=output.get("output_tokens"),
        temperature=output.get("temperature"),
        prompt_sha=target.prompt_sha,
    )


def run_target(target: Target, notes: list[dict]) -> list[dict]:
    """De-identify every note. A failure on one note is recorded on that note only."""
    if target.concurrency <= 1:
        return [_run_one(target, note) for note in notes]
    with ThreadPoolExecutor(max_workers=target.concurrency) as pool:
        return list(pool.map(lambda note: _run_one(target, note), notes))


def cache_path(target_name: str, cache_dir: Path = CACHE_DIR) -> Path:
    return cache_dir / ("%s.jsonl" % re.sub(r"[^A-Za-z0-9._-]+", "_", target_name))


def save_cache(results: list[dict], target_name: str, cache_dir: Path = CACHE_DIR) -> Path:
    path = cache_path(target_name, cache_dir)
    write_jsonl(path, results)
    return path


def load_cache(path: Path, notes: list[dict]) -> list[dict] | None:
    """Load cached results, or None when they do not match the current dataset."""
    rows = read_jsonl(path)
    by_id = {row["id"]: row for row in rows}
    if set(by_id) != {note["id"] for note in notes}:
        return None
    if any(by_id[note["id"]].get("source_sha1") != source_hash(note["source_text"]) for note in notes):
        return None
    return [by_id[note["id"]] for note in notes]


def collect_results(notes: list[dict], targets: list[Target], use_cache: bool,
                    cache_dir: Path = CACHE_DIR, log=print) -> dict[str, list[dict]]:
    """Run targets (or replay them from cache) and return results keyed by target name.

    In cache mode, every matching cache file is replayed and no model calls are made. The
    offline baseline still runs when it has no cache.
    """
    results: dict[str, list[dict]] = {}
    if use_cache:
        for path in sorted(cache_dir.glob("*.jsonl")):
            cached = load_cache(path, notes)
            if cached is None:
                log("  cache %s does not match the current dataset; ignored" % path.name)
                continue
            name = cached[0]["target"] if cached else path.stem
            results[name] = cached
            log("  %-24s replayed from cache (%s)" % (name, path.name))
    for target in targets:
        if target.name in results:
            continue
        if use_cache and not isinstance(target, BaselineTarget):
            log("  %-24s no cache; skipped (cache mode makes no model calls)" % target.name)
            continue
        started = time.perf_counter()
        rows = run_target(target, notes)
        errors = sum(1 for row in rows if row["error"])
        save_cache(rows, target.name, cache_dir)
        log("  %-24s ran %d notes in %.1fs (%d errors)" % (target.name, len(rows), time.perf_counter() - started, errors))
        results[target.name] = rows
    return results
