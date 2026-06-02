from __future__ import annotations

import os
import subprocess
import sys

import networkx as nx

from programming_ontology import paths
from programming_ontology.build_ontology import (
    build_combined_ontology,
    export_graph,
    read_csv,
)
from programming_ontology.preprocess_programming import combine_programming_data


def test_path_constants_resolve_to_project_structure() -> None:
    assert paths.PROJECT_ROOT.name == "ProgrammingOntology"
    assert paths.RAW_DATA_DIR == paths.PROJECT_ROOT / "data" / "raw"
    assert paths.PROCESSED_DATA_DIR == paths.PROJECT_ROOT / "data" / "processed"
    assert paths.REVIEW_DATA_DIR == paths.PROJECT_ROOT / "data" / "review"
    assert paths.GRAPH_OUTPUT_DIR == paths.PROJECT_ROOT / "outputs" / "graphs"
    assert paths.REPORT_OUTPUT_DIR == paths.PROJECT_ROOT / "outputs" / "reports"
    assert paths.PUBLISH_OUTPUT_DIR == paths.PROJECT_ROOT / "outputs" / "publish"
    assert paths.PROGRAMMING_DATA_FYNSP_2027_PATH.name == "programming_data_FYNSP27_wide.csv"
    assert paths.COMBINED_PROGRAMMING_DATA_PATH.name == "combined_programming_data.csv"
    assert paths.OFFICE_CLAIM_VALIDATION_PATH.name == "office_claim_validation.csv"
    assert paths.CHALLENGE_COVERAGE_SUMMARY_PATH.name == "challenge_coverage_summary.csv"
    assert paths.OFFICE_CHALLENGE_EVIDENCE_PATH.name == "office_challenge_evidence.csv"
    assert paths.WBS_CHALLENGE_SUMMARY_PATH.name == "wbs_challenge_summary.csv"
    assert paths.OPENAI_CLASSIFICATION_AUDIT_PATH.name == "openai_classification_audit.jsonl"
    assert paths.ASKSAGE_CLASSIFICATION_AUDIT_PATH.name == "asksage_classification_audit.jsonl"
    assert paths.OPENAI_AI_DIRECTNESS_AUDIT_PATH.name == "openai_ai_directness_audit.jsonl"
    assert paths.ASKSAGE_AI_DIRECTNESS_AUDIT_PATH.name == "asksage_ai_directness_audit.jsonl"


def test_raw_csv_files_are_loadable() -> None:
    programming = read_csv(paths.PROGRAMMING_DATA_PATH, schema="programming")
    programming_fynsp_2027 = read_csv(
        paths.PROGRAMMING_DATA_FYNSP_2027_PATH,
        schema="programming",
    )
    mstc = read_csv(paths.MSTC_DATA_PATH, schema="mstc")
    office_mapping = read_csv(
        paths.PROGRAM_OFFICE_MAPPING_PATH,
        schema="program_office",
    )

    assert not programming.empty
    assert not programming_fynsp_2027.empty
    assert "\ufffd" not in programming_fynsp_2027.to_string()
    assert not mstc.empty
    assert not office_mapping.empty


def test_combined_programming_preprocess_preserves_provenance() -> None:
    combined = combine_programming_data()

    assert len(combined) == 120
    assert combined["fynsp_cycle"].value_counts().to_dict() == {
        "FYNSP 2028": 61,
        "FYNSP 2027": 59,
    }
    first_fy28 = combined.loc[
        combined["programming_record_id"].eq("programming_record:programming_data_csv:1")
    ].iloc[0]
    first_fy27 = combined.loc[
        combined["programming_record_id"].eq(
            "programming_record:programming_data_fynsp27_wide_csv:1"
        )
    ].iloc[0]

    assert first_fy28["source_file"] == "programming_data.csv"
    assert first_fy28["source_row_number"] == 1
    assert first_fy28["fynsp_years"] == "28-32"
    assert first_fy27["source_file"] == "programming_data_FYNSP27_wide.csv"
    assert first_fy27["source_row_number"] == 1
    assert first_fy27["fynsp_years"] == "27-31"


def test_default_combined_ontology_uses_raw_data_dir() -> None:
    graph = build_combined_ontology()

    assert graph.number_of_nodes() > 0
    assert graph.number_of_edges() > 0
    assert "programming_data.csv" in graph.graph["source_files"]
    assert "programming_data_FYNSP27_wide.csv" in graph.graph["source_files"]


def test_exports_can_write_to_temporary_directory(tmp_path) -> None:
    graph = nx.MultiDiGraph()
    graph.add_node("source", label="Source", node_type="test")
    graph.add_node("target", label="Target", node_type="test")
    graph.add_edge("source", "target", relation="relates_to")

    graphml_path = export_graph(graph, tmp_path / "graphs" / "test.graphml")
    csv_path = export_graph(graph, tmp_path / "tables" / "test.csv")

    assert graphml_path.exists()
    assert csv_path.exists()


def test_package_imports_from_outside_project_root() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(paths.PROJECT_ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from programming_ontology import paths; print(paths.PROJECT_ROOT.name)",
        ],
        cwd=paths.PROJECT_ROOT.parent,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.stdout.strip() == "ProgrammingOntology"
