# Outputs Data Dictionary

This page describes the generated outputs most often used for review,
debugging, and publishing.

## Processed Data Outputs

### `combined_programming_data.csv`

Path:

```text
data/processed/combined_programming_data.csv
```

Purpose: combined FYNSP 2028 and FYNSP 2027 programming data with provenance.

Important columns:

| Column | Meaning |
| --- | --- |
| `programming_record_id` | Stable row key used throughout the pipeline. |
| `source_file` | Original raw programming extract. |
| `source_row_number` | Source-file row number. |
| `fynsp_cycle` | `FYNSP 2028` or `FYNSP 2027`. |
| `Funding Levels` | Source funding category. |
| `Sub Office Number` | Funding owner from the source row. |
| `WBS`, `WBS Name` | Programming work breakdown context. |
| `Scope Description` | Primary evidence used for challenge classification. |
| `FY2027` through `FY2032` | Normalized fiscal-year amount columns. |

### `programming_challenge_classifications.csv`

Path:

```text
data/processed/programming_challenge_classifications.csv
```

Purpose: raw classifier output before SME overrides are applied.

Important columns:

| Column | Meaning |
| --- | --- |
| `programming_record_id` | Row key. |
| `source_file`, `source_row_number`, `fynsp_cycle` | Provenance fields. |
| `normalized_funding_office` | Office used for validation after parent rollup logic. |
| `funding_level` | Normalized funding category. |
| `five_year_total` | Sum of the row's relevant FYNSP fiscal-year amounts. |
| `validation_funding_level` | True for `BASELINE`, `UFR`, and `ROT`; false for `DECREMENT`. |
| `predicted_challenge` | Classifier-selected Genesis Mission Challenge. |
| `confidence` | Classifier confidence. |
| `review_flag` | Whether classifier output needs review. |
| `ai_directness_score` | 0-100 AI-directness score. |
| `ai_directness_label` | AI-directness category. |
| `top_candidate_challenges` | Candidate challenge traceability. |

### `reviewed_programming_challenge_classifications.csv`

Path:

```text
data/processed/reviewed_programming_challenge_classifications.csv
```

Purpose: classifier output after SME overrides are applied. This is the main
row-level source for validation and dashboard artifacts.

Additional important columns:

| Column | Meaning |
| --- | --- |
| `original_predicted_challenge` | Challenge before SME override. |
| `sme_review_decision` | Review decision from `sme_review_overrides.csv`. |
| `sme_override_applied` | True when a correction changed the classifier result. |
| `sme_notes`, `sme_reviewer`, `sme_review_date` | Human review metadata. |

### `office_claim_validation.csv`

Path:

```text
data/processed/office_claim_validation.csv
```

Purpose: validates whether each claimed office has positive funding evidence
for the claimed challenge.

Important columns:

| Column | Meaning |
| --- | --- |
| `challenge_name` | Genesis Mission Challenge. |
| `claimed_office` | Lead or partner office from `ProgramOffice_Mapping.csv`. |
| `claim_role` | `lead` or `partner`. |
| `validation_status` | `validated`, `needs_review`, or `unfunded`. |
| `evidence_row_count` | Matching positive evidence rows. |
| `evidence_wbs_count` | Matching WBS count. |
| `five_year_total` | Matching positive funding total. |
| `review_flag_count` | Matching rows still needing review. |

### `challenge_coverage_summary.csv`

Path:

```text
data/processed/challenge_coverage_summary.csv
```

Purpose: challenge-level rollup of claimed offices, validation statuses, and
actual funding evidence.

### `office_challenge_evidence.csv`

Path:

```text
data/processed/office_challenge_evidence.csv
```

Purpose: positive evidence grouped by challenge and actual funding office.
Useful for finding unclaimed contributors.

### `wbs_challenge_summary.csv`

Path:

```text
data/processed/wbs_challenge_summary.csv
```

Purpose: rollup by WBS, challenge, and funding office.

### Audit JSONL Files

Paths:

```text
data/processed/asksage_classification_audit.jsonl
data/processed/asksage_ai_directness_audit.jsonl
data/processed/openai_classification_audit.jsonl
data/processed/openai_ai_directness_audit.jsonl
```

Purpose: provider audit/cache records used by `--resume-cache`.

## Graph Outputs

```text
outputs/graphs/combined_ontology.graphml
outputs/graphs/classified_ontology.graphml
```

`combined_ontology.graphml` includes raw ontology structure. 
`classified_ontology.graphml` includes evidence classification edges and SME
corrections.

## Dashboard Outputs

```text
outputs/reports/classification_dashboard.html
outputs/publish/genesis_dashboard_latest.html
outputs/publish/genesis_dashboard_YYYYMMDD_HHMMSS.html
outputs/publish/genesis_dashboard_YYYYMMDD_HHMMSS.zip
```

Use the timestamped zip as the normal share artifact.
