# Classification And SME Review

Classification predicts which Genesis Mission Challenge each programming row
supports. SME review records human decisions that can accept, flag, or correct
the prediction.

## Classifier Options

### Lexical

Command:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m programming_ontology.classify_programming --classifier lexical
```

The lexical classifier is deterministic and offline. It is useful for tests,
smoke checks, and debugging.

Important: lexical refreshes overwrite uncorrected LLM classifications. SME
rows with `corrected_challenge` still win, but accepted or discussion-only LLM
rows can be replaced by lexical predictions.

### Ask Sage

Command:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m programming_ontology.classify_programming --classifier asksage --resume-cache
```

Ask Sage uses `credentials.json` by default. That file is local, ignored by git,
and must not be committed.

Ask Sage writes audit/cache files:

```text
data/processed/asksage_classification_audit.jsonl
data/processed/asksage_ai_directness_audit.jsonl
```

### OpenAI

Command:

```powershell
$env:PYTHONPATH = "src"
$env:OPENAI_API_KEY = "..."
.\.venv\Scripts\python.exe -m programming_ontology.classify_programming --classifier openai --resume-cache
```

OpenAI writes audit/cache files:

```text
data/processed/openai_classification_audit.jsonl
data/processed/openai_ai_directness_audit.jsonl
```

## Cache Behavior

`--resume-cache` reuses prior provider decisions when the request hash matches.
The request hash includes the model and classification payload. If the row
content, candidate set, prompt payload, or model changes, the cache may not
match and the provider may be called again.

Running lexical first and then Ask Sage with cache is allowed. The second run
overwrites the first run's processed outputs with Ask Sage-based outputs plus
SME overrides.

## SME Review File

Review file:

```text
data/review/sme_review_overrides.csv
```

The key is:

```text
programming_record_id
```

Existing FYNSP 2028 IDs remain stable, such as:

```text
programming_record:programming_data_csv:1
```

FYNSP 2027 IDs are source-specific, such as:

```text
programming_record:programming_data_fynsp27_wide_csv:1
```

## What SME Overrides Protect

If `corrected_challenge` is populated:

- The SME challenge is treated as fact.
- The row's classifier challenge is overridden.
- The corrected challenge is used in validation, graph evidence, and dashboard
  outputs.
- This remains true even if the classifier backend changes.

If `review_decision` is `accept` but `corrected_challenge` is blank:

- The current classifier prediction is accepted for that run.
- A later run with a different classifier can change the underlying prediction.

If `review_decision` is `needs_discussion` and `corrected_challenge` is blank:

- The row remains flagged for review.
- A later run with a different classifier can change the underlying prediction.

If `review_decision` is `out_of_scope`:

- The row remains flagged and marked as out of scope in reviewed outputs.

## Safe Refresh After SME Edits

Use Ask Sage cache unless you intentionally want lexical replacement:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m programming_ontology.classify_programming --classifier asksage --resume-cache
.\.venv\Scripts\python.exe -m programming_ontology.publish_dashboard --zip
```

Then share the new zip from `outputs/publish/`.
