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


def test_path_constants_resolve_to_project_structure() -> None:
    assert paths.PROJECT_ROOT.name == "ProgrammingOntology"
    assert paths.RAW_DATA_DIR == paths.PROJECT_ROOT / "data" / "raw"
    assert paths.PROCESSED_DATA_DIR == paths.PROJECT_ROOT / "data" / "processed"
    assert paths.REVIEW_DATA_DIR == paths.PROJECT_ROOT / "data" / "review"
    assert paths.GRAPH_OUTPUT_DIR == paths.PROJECT_ROOT / "outputs" / "graphs"
    assert paths.REPORT_OUTPUT_DIR == paths.PROJECT_ROOT / "outputs" / "reports"
    assert paths.PUBLISH_OUTPUT_DIR == paths.PROJECT_ROOT / "outputs" / "publish"
    assert paths.OFFICE_CLAIM_VALIDATION_PATH.name == "office_claim_validation.csv"
    assert paths.CHALLENGE_COVERAGE_SUMMARY_PATH.name == "challenge_coverage_summary.csv"
    assert paths.OFFICE_CHALLENGE_EVIDENCE_PATH.name == "office_challenge_evidence.csv"
    assert paths.WBS_CHALLENGE_SUMMARY_PATH.name == "wbs_challenge_summary.csv"
    assert paths.OPENAI_CLASSIFICATION_AUDIT_PATH.name == "openai_classification_audit.jsonl"
    assert paths.ASKSAGE_CLASSIFICATION_AUDIT_PATH.name == "asksage_classification_audit.jsonl"


def test_raw_csv_files_are_loadable() -> None:
    programming = read_csv(paths.PROGRAMMING_DATA_PATH, schema="programming")
    mstc = read_csv(paths.MSTC_DATA_PATH, schema="mstc")
    office_mapping = read_csv(
        paths.PROGRAM_OFFICE_MAPPING_PATH,
        schema="program_office",
    )

    assert not programming.empty
    assert not mstc.empty
    assert not office_mapping.empty


def test_default_combined_ontology_uses_raw_data_dir() -> None:
    graph = build_combined_ontology()

    assert graph.number_of_nodes() > 0
    assert graph.number_of_edges() > 0
    assert "programming_data.csv" in graph.graph["source_files"]


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
