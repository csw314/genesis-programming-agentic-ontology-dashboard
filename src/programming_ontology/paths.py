"""Central project paths used by scripts and notebooks."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REVIEW_DATA_DIR = DATA_DIR / "review"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
GRAPH_OUTPUT_DIR = OUTPUT_DIR / "graphs"
REPORT_OUTPUT_DIR = OUTPUT_DIR / "reports"
PUBLISH_OUTPUT_DIR = OUTPUT_DIR / "publish"

NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"

MSTC_DATA_PATH = RAW_DATA_DIR / "MSTC_data.csv"
PROGRAM_OFFICE_MAPPING_PATH = RAW_DATA_DIR / "ProgramOffice_Mapping.csv"
PROGRAMMING_DATA_PATH = RAW_DATA_DIR / "programming_data.csv"

CLASSIFICATION_DF_PATH = PROCESSED_DATA_DIR / "classification_df.csv"
PROGRAMMING_CHALLENGE_CLASSIFICATIONS_PATH = (
    PROCESSED_DATA_DIR / "programming_challenge_classifications.csv"
)
REVIEWED_PROGRAMMING_CHALLENGE_CLASSIFICATIONS_PATH = (
    PROCESSED_DATA_DIR / "reviewed_programming_challenge_classifications.csv"
)
REVIEW_SUMMARY_PATH = PROCESSED_DATA_DIR / "review_summary.csv"
SME_REVIEW_OVERRIDES_PATH = REVIEW_DATA_DIR / "sme_review_overrides.csv"
OFFICE_CLAIM_VALIDATION_PATH = PROCESSED_DATA_DIR / "office_claim_validation.csv"
CHALLENGE_COVERAGE_SUMMARY_PATH = PROCESSED_DATA_DIR / "challenge_coverage_summary.csv"
OFFICE_CHALLENGE_EVIDENCE_PATH = PROCESSED_DATA_DIR / "office_challenge_evidence.csv"
WBS_CHALLENGE_SUMMARY_PATH = PROCESSED_DATA_DIR / "wbs_challenge_summary.csv"
OPENAI_CLASSIFICATION_AUDIT_PATH = PROCESSED_DATA_DIR / "openai_classification_audit.jsonl"
ASKSAGE_CLASSIFICATION_AUDIT_PATH = PROCESSED_DATA_DIR / "asksage_classification_audit.jsonl"
OPENAI_AI_DIRECTNESS_AUDIT_PATH = PROCESSED_DATA_DIR / "openai_ai_directness_audit.jsonl"
ASKSAGE_AI_DIRECTNESS_AUDIT_PATH = PROCESSED_DATA_DIR / "asksage_ai_directness_audit.jsonl"

COMBINED_ONTOLOGY_GRAPH_PATH = GRAPH_OUTPUT_DIR / "combined_ontology.graphml"
CLASSIFIED_ONTOLOGY_GRAPH_PATH = GRAPH_OUTPUT_DIR / "classified_ontology.graphml"
CLASSIFICATION_DASHBOARD_HTML_PATH = REPORT_OUTPUT_DIR / "classification_dashboard.html"


def ensure_output_dirs() -> None:
    """Create directories used for generated artifacts."""

    for directory in [
        PROCESSED_DATA_DIR,
        REVIEW_DATA_DIR,
        GRAPH_OUTPUT_DIR,
        REPORT_OUTPUT_DIR,
        PUBLISH_OUTPUT_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
