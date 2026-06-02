# Programming Ontology

Programming Ontology builds NetworkX ontology graphs from programming, Genesis
Mission Challenge, and program-office mapping CSV inputs. It then classifies
row-level programming funding evidence against Genesis Mission Challenges and
scores how directly each row supports AI/ML work. It publishes a self-contained
HTML dashboard for team review.

The primary analysis asks whether offices that claimed equity in
`ProgramOffice_Mapping.csv` have BASELINE, UFR, or ROT funding rows in the
combined FYNSP programming data that map to those same challenges. A
programming row's funding owner always comes from its source `Sub Office Number`;
`ProgramOffice_Mapping.csv` entries are claims to validate, not office
assignments for WBS items.

## Repository Layout

```text
data/raw/        Source CSV inputs supplied by the team
data/processed/  Generated classification, validation, and audit outputs
data/review/     Human-editable SME review override files
outputs/graphs/  Generated GraphML ontology artifacts
outputs/reports/ Generated static HTML reports
outputs/publish/ Team-shareable dashboard HTML and zip exports
notebooks/       Exploratory analysis and dashboard notebooks
src/             Importable Python package
tests/           Unit, path, artifact, and dashboard smoke tests
```

`data/processed/` and `outputs/` are reproducible outputs. Treat them as local
artifacts unless the team explicitly approves publishing a snapshot.

## Setup

Run commands from the repository root with Python 3.11 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` is a development bootstrap file. Runtime dependencies live in
`pyproject.toml`, and `requirements.txt` installs the package in editable mode
plus test tooling.

## Data Inputs

Place approved source files in `data/raw/` with these expected names:

```text
MSTC_data.csv
ProgramOffice_Mapping.csv
programming_data.csv
programming_data_FYNSP27_wide.csv
```

The pipeline preprocesses the FYNSP 2028 and FYNSP 2027 programming extracts
into `data/processed/combined_programming_data.csv` with `source_file`,
`source_row_number`, and `fynsp_cycle` provenance columns before building the
ontology.

The pipeline also maintains `data/review/sme_review_overrides.csv`. This file is
intended for SME review decisions and can be kept under version control when it
does not contain sensitive notes.

Before the first GitHub push, confirm with the team which CSV snapshots are
approved for the repo. Keep raw extracts, generated CSVs, published dashboards,
and audit JSONL files out of git unless they have been explicitly cleared.

## Common Workflow

Build and inspect the combined ontology from raw CSVs:

```powershell
python -m programming_ontology.build_ontology
```

Optionally run the preprocessing step directly:

```powershell
python -m programming_ontology.preprocess_programming
```

Run the deterministic offline classifier and write processed artifacts:

```powershell
python -m programming_ontology.classify_programming --classifier lexical
```

Publish a self-contained dashboard from processed artifacts:

```powershell
python -m programming_ontology.publish_dashboard --zip
```

Run the test suite:

```powershell
python -m pytest
```

Key generated outputs include:

```text
data/processed/combined_programming_data.csv
data/processed/programming_challenge_classifications.csv
data/processed/reviewed_programming_challenge_classifications.csv
data/processed/office_claim_validation.csv
data/processed/challenge_coverage_summary.csv
data/processed/office_challenge_evidence.csv
data/processed/wbs_challenge_summary.csv
data/processed/openai_classification_audit.jsonl
data/processed/asksage_classification_audit.jsonl
data/processed/openai_ai_directness_audit.jsonl
data/processed/asksage_ai_directness_audit.jsonl
outputs/graphs/combined_ontology.graphml
outputs/graphs/classified_ontology.graphml
outputs/reports/classification_dashboard.html
outputs/publish/genesis_dashboard_latest.html
```

## Optional LLM Classifiers

The default `lexical` classifier is offline and repeatable. OpenAI-backed and
Ask Sage-backed classification are opt-in and write provider audit/cache JSONL
files under `data/processed/` for both Genesis challenge mapping and
AI-directness scoring.

OpenAI classification requires `OPENAI_API_KEY`:

```powershell
$env:OPENAI_API_KEY = "..."
python -m programming_ontology.classify_programming --classifier openai --resume-cache
```

Ask Sage classification uses `credentials.json` from the project root by
default. The file is ignored by git and must never be committed.

```json
{
  "credentials": {
    "api_key": "YOUR_ASKSAGE_API_KEY",
    "Ask_sage_user_info": {
      "username": "YOUR_ASKSAGE_EMAIL"
    },
    "user_base_url": "https://api.nnsa.asksage.ai/user",
    "server_base_url": "https://api.nnsa.asksage.ai/server",
    "openai_url": "https://api.nnsa.asksage.ai/server/openai/v1",
    "model": "OPTIONAL_MODEL_NAME"
  }
}
```

```powershell
python -m programming_ontology.classify_programming --classifier asksage --resume-cache
```

If your Ask Sage OpenAI-compatible endpoint differs from the server API URL, set
`openai_url` in `credentials.json` or pass `--asksage-openai-url`.

## Publishing Notes

`python -m programming_ontology.publish_dashboard --zip` creates timestamped
HTML, updates `genesis_dashboard_latest.html`, and optionally writes a zip
bundle under `outputs/publish/`. The published HTML is self-contained and does
not expose local `data/processed/` file paths.

For team sharing, publish from a reviewed data snapshot, upload the timestamped
HTML or zip bundle to the approved location, and replace the shared latest copy
when the underlying programming artifacts are refreshed.

## Git Safety

The `.gitignore` excludes local environments, caches, credentials, generated
artifacts, audit logs, published dashboards, and accidental root-level CSV
exports. It does not by itself remove files that were already committed. If a
secret or unapproved data extract has ever been committed, rotate the secret and
clean the repository history before publishing.
