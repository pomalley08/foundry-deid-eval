# Foundry De-ID Eval

A starter kit for evaluating clinical-note de-identification (de-ID). It scores any de-ID
pipeline (a model prompt, Azure Health Data Services, a vendor tool or your own code) on PHI
leakage, over-redaction and clinical-value preservation, alongside latency and cost. It runs
locally and as custom evaluators in Microsoft Foundry, with a regression gate for CI.

> **Never commit real PHI.** The bundled dataset is synthetic. Keep real data in a gitignored
> folder (see [ADAPTING.md](ADAPTING.md)). Results are not a compliance determination.

## What's included

| Component | What it does |
|---|---|
| `deid_eval/phi_grader.py` | Standard-library grader: strict PHI recall, over-redaction and clinical preservation per note. The same file is registered as Foundry code evaluators, so local and cloud scores match. |
| `deid_eval/targets.py` | De-ID targets: a regex baseline and one target per Azure OpenAI / Foundry model deployment, with a shared prompt. Records latency and tokens. |
| `deid_eval/importer.py` | Imports outputs produced elsewhere (your pipeline, AHDS Tag results) as a target. |
| `deid_eval/report.py`, `viewer.py` | Ranked comparison (Markdown and CSV) and an offline HTML note viewer with highlighted input-versus-output diffs. |
| `deid_eval/foundry.py`, `gate.py` | Foundry standing eval with three criteria and run lineage, and a gate that compares a candidate against a baseline and exits non-zero on regression. |
| `deid_eval/dataset.py`, `data/` | A synthetic sample: 30 ophthalmology notes (exam notes, OCT reports, referral letters) with gold PHI spans, plus cached outputs from a regex baseline and three Azure OpenAI models. |

## Quick start (offline, no installs)

Python 3.11+.

```powershell
python -m deid_eval run --use-cache   # score the cached sample outputs and write outputs/
python -m deid_eval view              # open the note viewer
```

`outputs/report.md` ranks the targets; `outputs/viewer.html` lists the notes where targets
struggled and shows each note's input and output side by side.

## Setup for live models and Foundry

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # or: source .venv/bin/activate
pip install -r requirements.txt
Copy-Item .env.example .env           # then fill in endpoints and deployments
az login                              # keyless Microsoft Entra ID auth
```

Roles: **Cognitive Services OpenAI User** on the model resource and **Foundry User** (formerly
Azure AI User) on the Foundry project. Update `pricing.json` with current list prices (USD per
1M tokens); models without a price show `unpriced`.

## Commands

| Command | What it does |
|---|---|
| `python -m deid_eval run [--use-cache] [--targets a,b]` | Run the baseline and each configured deployment (or replay the cache), score and report. |
| `python -m deid_eval import --target NAME --file F.jsonl [--format text\|spans\|ahds-tag]` | Import outputs produced elsewhere as a target. |
| `python -m deid_eval report` | Re-score all cached targets; rewrite the report and viewer. |
| `python -m deid_eval view [--no-open]` | Rebuild and open the note viewer. |
| `python -m deid_eval foundry [--wait] [--targets a,b] [--force]` | Add runs for cached targets to the standing Foundry eval; unchanged outputs are skipped. |
| `python -m deid_eval gate --baseline A --candidate B [--tolerance N] [--allow-new-leaks]` | Compare B's latest Foundry run with A's: exit 0 pass, 1 regression, 2 cannot decide. |
| `python -m deid_eval generate [--seed 42 --size 30]` | Regenerate the synthetic sample dataset. |
| `python -m unittest discover -s tests` | Run the tests. |

## How scoring works

For each note, the grader aligns the de-identified output with the source text character by
character (Python `difflib`). Each placeholder, such as `[PATIENT_NAME]` or `[name]`, counts as one
unit, so its letters can't match the original text. Then:

- **Strict PHI recall.** A gold PHI span is caught only when every letter and digit in it was
  replaced *and* its text appears nowhere else in the output. Leaks are labeled `missed`,
  `partial` (some characters left), `repeat` (replaced here but still present elsewhere) or
  `failed` (empty output). Punctuation and spaces left between placeholders don't count.
- **Over-redaction.** Replaced text that overlaps no gold PHI, such as visual acuity `20/40`
  turned into `[DATE]`, lowers precision.
- **Clinical preservation.** The share of protected values (acuity, IOP, laterality,
  medications, eponyms) kept verbatim. Matching respects token boundaries, so `20/40` doesn't
  match inside `20/400`.

Targets only need to return text, so any tool can be scored. In Foundry, the three criteria
`phi_recall`, `over_redaction` and `clinical_preservation` are separate code evaluators built
from the same file, each scoring a note from 0 to 1 with pass threshold 1.0. The pass rate is
therefore the share of notes that are perfect on that criterion.

**Pooled versus per-note recall.** The local report pools all gold spans. Foundry's
`phi_recall` pass rate equals `1 - residual-PHI note rate`. A target can catch 98% of spans and
still leave PHI in a third of its notes.

## Reading the note viewer

- **List.** Each target column shows the note's recall and counts of leaked spans,
  over-redactions and lost clinical values. Notes where targets struggled come first. Click a
  header to sort, a row to open the note, or a target cell to open that note for that target.
- **Input pane.** Green is caught PHI; red is leaked PHI (hover for the reason); an orange dashed
  box is over-redacted text; blue and purple underlines are protected values kept and lost.
- **Output pane.** Gray chips are placeholders (orange when they replaced non-PHI); red is PHI
  still present; yellow is text not in the source.
- Arrow keys move between notes and targets. The URL keeps the note and target, and each target
  links to its latest Foundry run when one is recorded.

## Foundry quality gate

The local viewer explains a single run. Foundry keeps the history and serves as the gate.

- **History.** Runs accumulate in one standing eval (`FOUNDRY_EVAL_NAME`, default
  `deid-quality-gate`) for as long as the grader and dataset are unchanged. A grader or dataset
  change starts a new eval, so runs scored differently are never compared.
- **Comparison.** In the portal, select runs, choose **Compare** and set a baseline run; each
  criterion shows its change with significance testing.
- **Lineage.** Every run is tagged with target, model version, prompt hash, outputs hash,
  dataset hash, evaluator versions and git commit (`-dirty` if uncommitted). Foundry adds who
  created it. Evaluator versions are registered only when the grader code changes and are
  pinned in the eval.
- **Gate.** `gate` reads both runs back from Foundry. A criterion regresses when the candidate's
  pass rate drops more than `--tolerance` points. For `phi_recall`, any note that passed for the
  baseline and leaks for the candidate is also a regression, even at an equal pass rate, unless
  `--allow-new-leaks` is given. Each gate run writes a Markdown and JSON decision record to
  `outputs/gate/` with per-criterion results, the regressed notes linked into the viewer, and
  both runs' lineage.
- **Grading errors.** The Foundry grading sandbox occasionally errors on a whole criterion. The
  next `foundry` step resubmits such runs automatically, and the gate exits 2 rather than
  deciding on them.

Caveats: the portal's comparison view isn't saved as a dashboard (the decision record is the
durable artifact), and Foundry's GitHub and Azure DevOps evaluation actions target Foundry
agents, so the gate runs as a plain CLI step. See [ADAPTING.md](ADAPTING.md) for a CI example.

## Adapting to your data and pipeline

[ADAPTING.md](ADAPTING.md) covers keeping real data out of git, preparing a labeled dataset,
setting your PHI policy, importing your pipeline's or AHDS outputs, and running the gate in CI.

## Limitations

- The grader can only score PHI that is labeled; label quality bounds every metric.
- The placeholder category isn't checked for recall: a name replaced with `[DATE]` still counts
  as caught.
- Heavy rewording by a model shows up as over-redaction.
- The sample is 30 synthetic notes: enough to show the method, not to choose a model. Use a
  larger expert-labeled set that matches your note formats, and report confidence intervals.

## Project layout

```
deid_eval/      grader, targets, importer, report, viewer, Foundry and gate modules, CLI
data/           synthetic sample notes and cached sample outputs
tests/          unit tests (standard library unittest; a fake Foundry client for cloud paths)
pricing.json    list-price estimates used for cost
```

## License

[MIT](LICENSE)
