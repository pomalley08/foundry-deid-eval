# Adapting this to your data and pipeline

The repository ships with a synthetic sample so it runs out of the box. To evaluate your own
de-ID pipeline you replace four things:

```
 YOUR DATA ─────▶ notes.jsonl + labels      (a labeled dataset in your note formats)
 YOUR PIPELINE ─▶ a target                  (import its outputs, or call a model deployment)
 YOUR POLICY ───▶ phi_grader.py settings    (PHI categories, placeholders, protected values)
 YOUR GATE ─────▶ Foundry eval + CI         (eval name, baseline, tolerance)
```

## 1. Keep real data out of git

- Put your dataset and outputs in a gitignored folder and point the tools at it:

  ```
  DEID_DATA_DIR=data/private      # in .env; data/private/ is already in .gitignore
  ```

  All commands then read `data/private/notes.jsonl` and write `data/private/cache/`.
- `outputs/` (reports, the note viewer, gate records) contains note text and is gitignored.
- Foundry runs store each item, including the source note, in your Foundry project. Before
  submitting real PHI, confirm with your compliance team that the project, region and services
  meet your requirements (for example, your HIPAA BAA scope and data-residency rules).
- Run labeling tools and the pipeline inside the same compliant environment.

## 2. Prepare a labeled dataset

`notes.jsonl` has one JSON object per line:

```json
{"id": "note-001",
 "note_type": "exam_note",
 "source_text": "Pt: Ana Ruiz  MRN 40418827\nVA cc: OD 20/40 ...",
 "phi_spans": [
   {"start": 4, "end": 12, "category": "PATIENT_NAME", "text": "Ana Ruiz"},
   {"start": 18, "end": 26, "category": "MRN", "text": "40418827"}],
 "protected_values": ["OD", "20/40"]}
```

- `start` and `end` are Python string indexes (Unicode code points) into `source_text`, and
  `source_text[start:end]` must equal `text`.
- `protected_values` are clinical strings that must survive verbatim (acuity, IOP, laterality,
  medications, eponyms). They can usually be derived by rules, then reviewed by a clinician.
- `note_type` is free text used for filtering in the viewer.

**Labeling tools.** Use a span-annotation tool that records character offsets automatically
rather than building one, for example Label Studio (open source, self-hostable), Azure AI
Language custom NER labeling, or Doccano. Then convert the export. For a Label Studio JSON export
of a named-entity task:

```python
import json
from deid_eval.config import write_jsonl
from deid_eval.dataset import check_record

tasks = json.load(open("label_studio_export.json", encoding="utf-8"))
records = []
for task in tasks:
    text = task["data"]["text"]
    spans = []
    for result in task["annotations"][0]["result"]:
        value = result["value"]
        spans.append({"start": value["start"], "end": value["end"],
                      "category": value["labels"][0], "text": text[value["start"]:value["end"]]})
    records.append({"id": str(task["id"]), "note_type": task["data"].get("note_type", "note"),
                    "source_text": text, "phi_spans": sorted(spans, key=lambda s: s["start"]),
                    "protected_values": []})          # fill from your rules
for record in records:
    problems = check_record(record)                   # offsets, unlabeled repeats, protected values
    if problems:
        print(record["id"], problems)
write_jsonl("data/private/notes.jsonl", records)
```

`check_record` also flags a labeled PHI string that appears again verbatim elsewhere in the note
without its own label, a common annotation miss. (It can't spot a partial mention, such as a
surname alone, so the guidelines should require labeling every mention.)

**Label quality.**
- Write labeling guidelines that fix span boundaries (is "Dr." part of a name? is a full
  address, including city and ZIP, one span?). Strict recall depends on them.
- Pre-labeling with a tool (for example AHDS `Tag` or a model) and having humans correct it is
  much faster. If that tool is also being evaluated, pre-label with something else or label a
  subset blind, so the gold labels don't favor it.
- Double-label a sample and resolve disagreements to measure label quality.
- Use enough notes per note type to report confidence intervals; 30 notes only show the method.

## 3. Set your PHI policy

In `deid_eval/phi_grader.py`:

- **`CATEGORIES`** lists the category names used for reporting. Recall doesn't depend on the
  category: any replacement of a gold span counts, so labels mainly drive the per-category
  tables.
- **`_PLACEHOLDER_PATTERN`** recognizes placeholders: any bracketed word such as `[NAME]` or
  `[name]`. If your pipeline uses another marker (for example `<NAME>` or `***`), change the
  pattern so placeholders are treated as single units during alignment. Bracketed words that
  occur in your source notes (for example `[sic]`) would also be treated as placeholders.
- The strictness rules (every letter and digit replaced, no verbatim repeat elsewhere) are in
  `score_note`.

Changing the grader changes its code hash, so the next `foundry` step registers new evaluator
versions and starts a new standing eval. That is intended: runs scored by different graders
aren't compared.

## 4. Bring your pipeline's outputs

**Import outputs produced anywhere.** Write one JSONL row per note, keyed by the note `id`, and
import it as a target:

```powershell
python -m deid_eval import --target my-pipeline-v3 --file my_outputs.jsonl --format text
python -m deid_eval report
```

| `--format` | Row shape | Use for |
|---|---|---|
| `text` | `{"id", "deid_text"}` | Any pipeline that returns de-identified text with placeholders |
| `spans` | `{"id", "spans": [{"start", "end", "category"}]}` | Pipelines that return detected PHI positions; each span becomes `[CATEGORY]` |
| `ahds-tag` | `{"id", "taggerResult": {"entities": [...]}}` | Azure Health Data Services de-identification `Tag` results (offsets from `codePoint`) |

Rows may also carry `model`, `latency_ms`, `input_tokens`, `output_tokens`, `prompt_sha` and
`error`. Rows for unknown note ids are rejected. Missing notes fail the import unless you pass
`--allow-missing`, which records them as errors, so they count as failures, not passes.

**Azure Health Data Services (AHDS).** Score the `Tag` operation output (`--format ahds-tag`),
which gives exact entity positions. The `Redact` output also works (`[name]` placeholders), but
surrogated text can't be scored by alignment: realistic replacement values share characters
with the originals and would show as false partial leaks. Check surrogated output separately,
for example that no original PHI string appears in it and that protected values survive.

**Model deployments.** To compare prompts or models directly, set `AZURE_OPENAI_DEPLOYMENTS` and
edit `DEID_PROMPT` in `deid_eval/targets.py`; each run records the prompt hash. For other
callable pipelines, subclass `Target` and add it in `__main__._selected_targets`.

## 5. Run the gate in CI

1. Produce candidate outputs (your pipeline, then `import`; or `run --targets <deployment>`).
2. `python -m deid_eval foundry --wait` adds runs for changed outputs to the standing eval.
3. `python -m deid_eval gate --baseline <approved> --candidate <proposed>` fails the build on
   exit 1 (regression) or 2 (cannot decide).
4. Publish `outputs/gate/` as a build artifact; the Markdown record is meant for the change
   ticket.

Example GitHub Actions job using keyless workload identity federation (the identity needs the
**Foundry User** role on the project):

```yaml
jobs:
  deid-gate:
    runs-on: ubuntu-latest
    permissions:
      id-token: write
      contents: read
    env:
      FOUNDRY_PROJECT_ENDPOINT: ${{ vars.FOUNDRY_PROJECT_ENDPOINT }}
      FOUNDRY_EVAL_DEPLOYMENT: ${{ vars.FOUNDRY_EVAL_DEPLOYMENT }}
      FOUNDRY_EVAL_NAME: deid-quality-gate
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - uses: azure/login@v2
        with:
          client-id: ${{ vars.AZURE_CLIENT_ID }}
          tenant-id: ${{ vars.AZURE_TENANT_ID }}
          subscription-id: ${{ vars.AZURE_SUBSCRIPTION_ID }}
      - run: pip install -r requirements.txt
      # Produce or import the candidate's outputs here.
      - run: python -m deid_eval foundry --wait
      - run: python -m deid_eval gate --baseline approved-pipeline --candidate my-pipeline-v3
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: deid-gate-record
          path: outputs/gate/
```

Gate options: `--tolerance N` allows a pass-rate drop of N percentage points, and
`--allow-new-leaks` judges `phi_recall` on pass rate only. Use one `FOUNDRY_EVAL_NAME` per
pipeline or project so histories stay separate.

## 6. What is sample-only

- `deid_eval/dataset.py` and `data/notes.jsonl`: the synthetic ophthalmology generator and
  dataset. Keep them for tests, or replace `data/` with your own sample.
- `data/cache/*.jsonl`: cached sample outputs from a regex baseline and three Azure OpenAI
  models.
- `pricing.json`: list-price estimates retrieved from the Azure Retail Prices API; update them
  for your models, region and agreements.
- `DEID_PROMPT` in `targets.py`: a sample prompt written for ophthalmology notes.
