# Dashboard Publishing

The dashboard is a static HTML artifact built from processed CSV outputs. It can
be opened directly in a browser and shared as a zip bundle.

## Normal Refresh After SME Edits

If you updated:

```text
data/review/sme_review_overrides.csv
```

run:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m programming_ontology.classify_programming --classifier asksage --resume-cache
.\.venv\Scripts\python.exe -m programming_ontology.publish_dashboard --zip
```

The first command:

- Rebuilds `combined_programming_data.csv`.
- Reclassifies rows with Ask Sage cache where possible.
- Applies SME overrides.
- Rebuilds reviewed classifications, validation summaries, and graphs.

The second command:

- Builds the dashboard HTML from processed artifacts.
- Updates `outputs/publish/genesis_dashboard_latest.html`.
- Creates a timestamped HTML file.
- Creates a timestamped zip bundle.

## Published Files

Latest dashboard:

```text
outputs/publish/genesis_dashboard_latest.html
```

Timestamped dashboard:

```text
outputs/publish/genesis_dashboard_YYYYMMDD_HHMMSS.html
```

Timestamped zip:

```text
outputs/publish/genesis_dashboard_YYYYMMDD_HHMMSS.zip
```

The zip is the recommended stakeholder share artifact.

## AskSage AI Insights

The dashboard can still be opened directly as static HTML, but AskSage insights
and Draft Report generation use a local proxy so API keys are not sent directly
from the browser to AskSage.

Run:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m programming_ontology.dashboard_ai_proxy --dashboard outputs\publish\genesis_dashboard_latest.html
```

Then open the local URL printed by the command and enter AskSage credentials in
the dashboard. Use `Get Insights` for chart/table-specific answers or
`Draft Report` to generate a downloadable Word executive summary for the
currently visible dashboard view.

## Report HTML vs Published HTML

Report output:

```text
outputs/reports/classification_dashboard.html
```

Published output:

```text
outputs/publish/genesis_dashboard_latest.html
```

The report output is useful for local inspection. The published output is the
team-shareable copy and includes publish metadata.

## Deterministic Publishing

For tests or reproducible local checks, pass a timestamp:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m programming_ontology.publish_dashboard --zip --timestamp 20260602_120000
```

## Metadata

You can add a custom dashboard metadata line:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m programming_ontology.publish_dashboard --zip --metadata "Generated from reviewed FYNSP snapshot."
```

## Publishing Checklist

Before sharing:

- Confirm `sme_review_overrides.csv` contains the intended review edits.
- Refresh with Ask Sage cache unless lexical replacement is intentional.
- Run `publish_dashboard --zip`.
- Open `genesis_dashboard_latest.html` locally and confirm the FYNSP toggle,
  ROT funding, and reviewed corrections appear as expected.
- Share the timestamped zip from `outputs/publish/`.
