# Overview

Programming Ontology helps analysts review how NNSA FYNSP programming rows
support Genesis Mission Challenges. It combines source programming extracts,
classifies each funding row against a challenge, applies SME corrections, and
publishes a static dashboard for review and sharing.

## Audience

This documentation is for:

- Analysts refreshing the dashboard after data or SME review changes.
- Engineers maintaining the Python package or tests.
- Reviewers who need to understand where a dashboard number came from.
- New project members who did not participate in the original build.

## What The Package Answers

The package is built around a few practical questions:

- Which programming rows support each Genesis Mission Challenge?
- Which program offices are actually funding work mapped to a challenge?
- Do offices listed in `ProgramOffice_Mapping.csv` have positive funding
  evidence for their claimed challenges?
- Which rows are direct AI/ML work, AI-adjacent infrastructure, or weaker AI
  links?
- Which row-level classifications were accepted, corrected, or flagged by SME
  review?

## Data Lifecycle

The lifecycle is intentionally linear:

```text
raw CSVs
-> combined_programming_data.csv
-> programming_challenge_classifications.csv
-> sme_review_overrides.csv applied
-> reviewed_programming_challenge_classifications.csv
-> validation summary CSVs
-> classification_dashboard.html
-> genesis_dashboard_latest.html + timestamped zip
```

The raw files are the source inputs. `combined_programming_data.csv` preserves
where each programming row came from. Classification outputs are generated.
`sme_review_overrides.csv` is the human review file. The reviewed
classification table is the source for validation summaries, ontology evidence
edges, and the dashboard.

## Funding Treatment

Dashboard and validation views treat these funding levels as positive evidence:

```text
BASELINE
UFR
ROT
```

`DECREMENT` rows are retained in row-level classification outputs but excluded
from positive funding evidence, validation totals, and executive dashboard
metrics.

## Primary Workflow

For normal dashboard refreshes after SME edits, use Ask Sage cache and then
publish:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m programming_ontology.classify_programming --classifier asksage --resume-cache
.\.venv\Scripts\python.exe -m programming_ontology.publish_dashboard --zip
```

For details, see:

- [Data Inputs](02_data_inputs.md)
- [Pipeline Architecture](03_pipeline_architecture.md)
- [Classification And SME Review](04_classification_and_sme_review.md)
- [Dashboard Publishing](05_dashboard_publishing.md)
