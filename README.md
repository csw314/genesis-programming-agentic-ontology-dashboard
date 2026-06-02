# Programming Ontology

Programming Ontology maps NNSA FYNSP programming rows to Genesis Mission
Challenges, applies SME review decisions, validates program-office claims, and
publishes a self-contained HTML dashboard for review and sharing.

The core data lifecycle is:

```text
raw CSVs
-> data/processed/combined_programming_data.csv
-> classified rows
-> SME overrides applied
-> validation summaries
-> static dashboard + zip
```

The package treats `BASELINE`, `UFR`, and `ROT` as positive funding for
validation and dashboard views. `DECREMENT` rows remain in row-level outputs but
are excluded from positive funding evidence.

## Quick Start

Run commands from the repository root in PowerShell.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
```

Preprocess the FYNSP programming extracts:

```powershell
.\.venv\Scripts\python.exe -m programming_ontology.preprocess_programming
```

Refresh classifications, SME overrides, validation tables, and graphs using
Ask Sage cache:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m programming_ontology.classify_programming --classifier asksage --resume-cache
```

Publish the latest dashboard and create a timestamped zip:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m programming_ontology.publish_dashboard --zip
```

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Important Warning

Do not use the lexical classifier unless you intentionally want to replace
uncorrected LLM classifications with deterministic lexical classifications.
SME rows with `corrected_challenge` still win, but accepted or discussion-only
LLM rows can be overwritten.

```powershell
.\.venv\Scripts\python.exe -m programming_ontology.classify_programming --classifier lexical
```

Use this command only for offline smoke tests, deterministic debugging, or a
deliberate lexical refresh.

## Repository Layout

```text
data/raw/        Approved source CSV inputs
data/processed/  Generated combined data, classifications, validation, audits
data/review/     Human-editable SME review overrides
docs/            Operational documentation
notebooks/       Exploratory and dashboard notebooks
outputs/graphs/  Generated GraphML ontology artifacts
outputs/reports/ Generated dashboard report HTML
outputs/publish/ Shareable latest and timestamped dashboard HTML/zip files
src/             Python package
tests/           Unit, path, artifact, and dashboard smoke tests
```

## Required Raw Inputs

```text
data/raw/MSTC_data.csv
data/raw/ProgramOffice_Mapping.csv
data/raw/programming_data.csv
data/raw/programming_data_FYNSP27_wide.csv
```

The FYNSP 2028 and FYNSP 2027 programming extracts are combined into
`data/processed/combined_programming_data.csv` with provenance columns such as
`source_file`, `source_row_number`, `fynsp_cycle`, and
`programming_record_id`.

## Documentation

- [Overview](docs/01_overview.md)
- [Data Inputs](docs/02_data_inputs.md)
- [Pipeline Architecture](docs/03_pipeline_architecture.md)
- [Classification And SME Review](docs/04_classification_and_sme_review.md)
- [Dashboard Publishing](docs/05_dashboard_publishing.md)
- [Outputs Data Dictionary](docs/06_outputs_data_dictionary.md)
- [Troubleshooting](docs/07_troubleshooting.md)

## Common Outputs

```text
data/processed/combined_programming_data.csv
data/processed/programming_challenge_classifications.csv
data/processed/reviewed_programming_challenge_classifications.csv
data/processed/office_claim_validation.csv
data/processed/challenge_coverage_summary.csv
data/processed/office_challenge_evidence.csv
data/processed/wbs_challenge_summary.csv
outputs/reports/classification_dashboard.html
outputs/publish/genesis_dashboard_latest.html
outputs/publish/genesis_dashboard_YYYYMMDD_HHMMSS.zip
```

Treat generated data, dashboards, audit logs, and raw CSV snapshots according
to team approval rules before committing or sharing them.
