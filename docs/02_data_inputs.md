# Data Inputs

The pipeline expects approved source CSVs under `data/raw/` and a human-editable
SME review file under `data/review/`.

## Required Raw Files

```text
data/raw/MSTC_data.csv
data/raw/ProgramOffice_Mapping.csv
data/raw/programming_data.csv
data/raw/programming_data_FYNSP27_wide.csv
```

`MSTC_data.csv` defines Genesis Mission Challenge context. The challenge names,
problem statements, proposed solutions, advantages, and outcomes are used as
classification context.

`ProgramOffice_Mapping.csv` lists claimed lead and partner program offices for
each challenge. These rows are claims to validate. They do not assign WBS rows
to offices.

`programming_data.csv` is the original FYNSP 2028 programming extract covering
FY2028 through FY2032.

`programming_data_FYNSP27_wide.csv` is the FYNSP 2027 programming extract
covering FY2027 through FY2031. It is read with encoding fallback because the
file may not be UTF-8.

## FYNSP 2027 vs FYNSP 2028

The preprocessing step combines the two programming extracts into:

```text
data/processed/combined_programming_data.csv
```

The combined file preserves both source cycles:

```text
FYNSP 2028: programming_data.csv, FY2028-FY2032
FYNSP 2027: programming_data_FYNSP27_wide.csv, FY2027-FY2031
```

The dashboard defaults to a combined view and includes toggles for FYNSP 2028
and FYNSP 2027 views.

## Provenance Columns

The combined programming file adds provenance columns so every row can be traced
back to the source extract:

| Column | Meaning |
| --- | --- |
| `programming_record_id` | Stable row key used by classification outputs and SME review. |
| `source_file` | Raw programming file that supplied the row. |
| `source_row_number` | One-based row number in the source programming file. |
| `fynsp_cycle` | Human-readable cycle label, such as `FYNSP 2028` or `FYNSP 2027`. |
| `fynsp_years` | Compact year range, such as `28-32` or `27-31`. |

FYNSP 2028 record IDs remain stable, for example:

```text
programming_record:programming_data_csv:1
```

FYNSP 2027 rows use their own source-specific IDs, for example:

```text
programming_record:programming_data_fynsp27_wide_csv:1
```

## Funding Levels

Positive funding evidence:

```text
BASELINE
UFR
ROT
```

Excluded from positive evidence:

```text
DECREMENT
```

`DECREMENT` rows remain useful for traceability and classification review, but
they do not validate a program-office claim or contribute to positive dashboard
funding totals.

## SME Review File

The review file is:

```text
data/review/sme_review_overrides.csv
```

It is keyed by `programming_record_id`. Existing review decisions are preserved
when the pipeline refreshes, and new programming rows are appended to the
review table.

Important editable fields include:

| Column | Purpose |
| --- | --- |
| `review_decision` | `accept`, `change`, `needs_discussion`, or `out_of_scope`. |
| `corrected_challenge` | SME-selected challenge. If populated, it overrides the classifier. |
| `sme_notes` | Human review notes. |
| `reviewer` | Reviewer name or initials. |
| `review_date` | Review date. |

See [Classification And SME Review](04_classification_and_sme_review.md) for
what SME overrides protect.
