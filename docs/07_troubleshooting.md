# Troubleshooting

This page lists common issues and the fastest checks.

## `No module named programming_ontology`

Set `PYTHONPATH` before running module commands:

```powershell
$env:PYTHONPATH = "src"
```

Then rerun the command:

```powershell
.\.venv\Scripts\python.exe -m programming_ontology.publish_dashboard --zip
```

If the package is installed in editable mode, `PYTHONPATH` may not be required,
but setting it is harmless for local repo work.

## `No module named pytest`

Use the repo virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

If pytest is still missing, install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Dashboard Did Not Change After SME Edits

Editing `sme_review_overrides.csv` alone does not rebuild generated outputs.
Run:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m programming_ontology.classify_programming --classifier asksage --resume-cache
.\.venv\Scripts\python.exe -m programming_ontology.publish_dashboard --zip
```

Then reopen:

```text
outputs/publish/genesis_dashboard_latest.html
```

If your browser shows the old page, hard refresh or open the new timestamped
HTML in `outputs/publish/`.

## Accidental Lexical Overwrite

This command can replace uncorrected LLM classifications:

```powershell
.\.venv\Scripts\python.exe -m programming_ontology.classify_programming --classifier lexical
```

Rows with `corrected_challenge` in `sme_review_overrides.csv` are still
protected. Rows without corrections can change.

To restore Ask Sage outputs, rerun:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m programming_ontology.classify_programming --classifier asksage --resume-cache
```

## Ask Sage Credentials Error

Ask Sage uses:

```text
credentials.json
```

The file should live at the repo root and should not be committed. It needs a
shape like:

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

If the OpenAI-compatible endpoint differs, set `openai_url` in
`credentials.json` or pass `--asksage-openai-url`.

## Stale LLM Cache

`--resume-cache` only reuses entries when the request hash matches. Cache misses
can happen when:

- The row text changed.
- The candidate challenge set changed.
- The model changed.
- The classifier payload changed.
- A new FYNSP source row was added.

Cache misses are normal after data or prompt changes. The provider may be called
again for those rows.

## FYNSP 2027 Rows Missing

Confirm the raw file exists:

```text
data/raw/programming_data_FYNSP27_wide.csv
```

Run preprocessing:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m programming_ontology.preprocess_programming
```

Then inspect:

```text
data/processed/combined_programming_data.csv
```

It should contain both `FYNSP 2028` and `FYNSP 2027` values in `fynsp_cycle`.

## ROT Missing From Dashboard

Confirm the full classification pipeline was rerun after the new data was
added:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m programming_ontology.classify_programming --classifier asksage --resume-cache
.\.venv\Scripts\python.exe -m programming_ontology.publish_dashboard --zip
```

The dashboard should include `BASELINE`, `UFR`, and `ROT` funding modes.

## Replacement Characters In Outputs

The FYNSP 2027 programming file is read with encoding fallback. If you see the
Unicode replacement character `U+FFFD` in outputs, rerun preprocessing and
tests:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m programming_ontology.preprocess_programming
.\.venv\Scripts\python.exe -m pytest
```

The tests include checks for replacement characters in generated artifacts.
