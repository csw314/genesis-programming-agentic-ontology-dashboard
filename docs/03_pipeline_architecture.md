# Pipeline Architecture

The package is organized as a reproducible data pipeline. Each stage consumes
the previous stage and writes artifacts used by the next stage.

## Lifecycle

```text
raw CSVs
-> preprocessing
-> ontology graph
-> row classification
-> SME overrides
-> validation outputs
-> dashboard report
-> published dashboard zip
```

## 1. Preprocessing

Module:

```text
src/programming_ontology/preprocess_programming.py
```

Command:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m programming_ontology.preprocess_programming
```

This combines FYNSP 2028 and FYNSP 2027 programming extracts into:

```text
data/processed/combined_programming_data.csv
```

The output normalizes fiscal-year columns from FY2027 through FY2032 and adds
provenance fields such as `source_file`, `source_row_number`, `fynsp_cycle`,
and `programming_record_id`.

## 2. Ontology Graph

Module:

```text
src/programming_ontology/build_ontology.py
```

The ontology graph turns CSV rows into typed NetworkX nodes and edges. It
includes:

- Genesis Mission Challenge context from `MSTC_data.csv`.
- Program-office claim mappings from `ProgramOffice_Mapping.csv`.
- Programming row evidence from `combined_programming_data.csv`.

Generated graph outputs:

```text
outputs/graphs/combined_ontology.graphml
outputs/graphs/classified_ontology.graphml
```

## 3. Row Classification

Module:

```text
src/programming_ontology/classify_programming.py
```

The classifier maps each programming row to one Genesis Mission Challenge and
scores AI-directness. It writes:

```text
data/processed/programming_challenge_classifications.csv
data/processed/classification_df.csv
```

Classifier choices:

```text
lexical
asksage
openai
```

The recommended refresh path after SME review edits is Ask Sage with cache:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m programming_ontology.classify_programming --classifier asksage --resume-cache
```

## 4. SME Overrides

The classifier output is merged with:

```text
data/review/sme_review_overrides.csv
```

Rows with `corrected_challenge` override classifier predictions. Rows without
corrections use the current classifier output plus their review decision.

Reviewed output:

```text
data/processed/reviewed_programming_challenge_classifications.csv
```

## 5. Validation Outputs

Reviewed rows are used to validate program-office claims. Positive evidence is
limited to `BASELINE`, `UFR`, and `ROT` rows with positive funding totals.

Generated validation outputs:

```text
data/processed/office_claim_validation.csv
data/processed/challenge_coverage_summary.csv
data/processed/office_challenge_evidence.csv
data/processed/wbs_challenge_summary.csv
data/processed/review_summary.csv
```

## 6. Dashboard

Report output:

```text
outputs/reports/classification_dashboard.html
```

Published outputs:

```text
outputs/publish/genesis_dashboard_latest.html
outputs/publish/genesis_dashboard_YYYYMMDD_HHMMSS.html
outputs/publish/genesis_dashboard_YYYYMMDD_HHMMSS.zip
```

The dashboard is static and self-contained. It includes combined, FYNSP 2028,
and FYNSP 2027 views.

See [Dashboard Publishing](05_dashboard_publishing.md) for refresh commands.
