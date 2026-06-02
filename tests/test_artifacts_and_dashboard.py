from __future__ import annotations

from html import escape
import json
import re
from zipfile import ZipFile

import matplotlib
import networkx as nx
import pandas as pd
import pytest

from programming_ontology import paths
from programming_ontology._dashboard_report import build_classification_dashboard_html
from programming_ontology.publish_dashboard import publish_dashboard


FUNDING_LEVELS = ("BASELINE", "UFR", "ROT")


def _render_dashboard_html() -> str:
    return build_classification_dashboard_html(
        paths_module=paths,
        reviewed_classifications=pd.read_csv(
            paths.REVIEWED_PROGRAMMING_CHALLENGE_CLASSIFICATIONS_PATH
        ),
        office_claim_validation=pd.read_csv(paths.OFFICE_CLAIM_VALIDATION_PATH),
        challenge_coverage=pd.read_csv(paths.CHALLENGE_COVERAGE_SUMMARY_PATH),
        office_challenge_evidence=pd.read_csv(paths.OFFICE_CHALLENGE_EVIDENCE_PATH),
    )


def test_processed_artifacts_have_expected_row_level_schema() -> None:
    classifications = pd.read_csv(paths.PROGRAMMING_CHALLENGE_CLASSIFICATIONS_PATH)
    reviewed = pd.read_csv(paths.REVIEWED_PROGRAMMING_CHALLENGE_CLASSIFICATIONS_PATH)
    claims = pd.read_csv(paths.OFFICE_CLAIM_VALIDATION_PATH)
    coverage = pd.read_csv(paths.CHALLENGE_COVERAGE_SUMMARY_PATH)
    evidence = pd.read_csv(paths.OFFICE_CHALLENGE_EVIDENCE_PATH)

    required_classification_columns = {
        "programming_record_id",
        "source_file",
        "source_row_number",
        "fynsp_cycle",
        "source_sub_office",
        "normalized_funding_office",
        "funding_level",
        "scope_description",
        "five_year_total",
        "predicted_challenge",
        "ai_directness_score",
        "ai_directness_label",
        "ai_directness_confidence",
        "ai_directness_rationale",
        "fy2027",
        "fy2032",
    }
    assert required_classification_columns.issubset(classifications.columns)
    assert len(classifications) == 120
    assert len(classifications) == len(reviewed)
    assert classifications["fynsp_cycle"].value_counts().to_dict() == {
        "FYNSP 2028": 61,
        "FYNSP 2027": 59,
    }
    validation_by_level = (
        classifications.groupby("funding_level")["validation_funding_level"]
        .first()
        .astype(str)
        .str.lower()
        .to_dict()
    )
    assert validation_by_level["ROT"] == "true"
    assert validation_by_level["DECREMENT"] == "false"
    assert "lead_program_office" not in classifications.columns
    assert "support_program_offices" not in classifications.columns
    assert set(claims["validation_status"]).issubset(
        {"validated", "needs_review", "unfunded"}
    )
    assert int(coverage["claim_count"].sum()) == len(claims)
    assert "five_year_total" in evidence.columns
    claim_pairs = {
        (row.challenge_name, row.claimed_office, row.claim_role)
        for row in claims.itertuples(index=False)
    }
    assert "NAPAS" not in set(claims["claimed_office"])
    assert "NAMB" not in set(claims["claimed_office"])
    assert (
        "Accelerating Business and Management Processes for National Security Missions",
        "NA-MB",
        "lead",
    ) in claim_pairs
    assert (
        "Accelerating Nuclear Enterprise Acquisition",
        "NA-PAS",
        "lead",
    ) in claim_pairs


def test_artifacts_do_not_contain_replacement_characters() -> None:
    artifact_paths = [
        *paths.PROCESSED_DATA_DIR.glob("*.csv"),
        paths.CLASSIFICATION_DASHBOARD_HTML_PATH,
    ]

    for artifact_path in artifact_paths:
        text = artifact_path.read_text(encoding="utf-8")
        assert "\ufffd" not in text, f"replacement character found in {artifact_path}"


def test_classified_graph_uses_evidence_edges_only() -> None:
    graph = nx.read_graphml(paths.CLASSIFIED_ONTOLOGY_GRAPH_PATH)
    relations = {data.get("relation") for _, _, data in graph.edges(data=True)}

    assert "classified_as_challenge" in relations
    assert "funds_challenge_evidence" in relations
    assert "assigned_lead_program_office" not in relations
    assert "assigned_support_program_office" not in relations


def test_classified_graph_uses_sme_corrected_challenges() -> None:
    graph = nx.read_graphml(paths.CLASSIFIED_ONTOLOGY_GRAPH_PATH)
    review = pd.read_csv(paths.SME_REVIEW_OVERRIDES_PATH, dtype=str).fillna("")
    corrections = review.loc[review["corrected_challenge"].str.strip().ne("")]
    assert not corrections.empty

    graph_challenges_by_record = {
        source: graph.nodes[target].get("label", target)
        for source, target, data in graph.edges(data=True)
        if data.get("relation") == "classified_as_challenge"
    }
    mismatches = [
        (
            row.programming_record_id,
            row.source_row_number,
            row.corrected_challenge,
            graph_challenges_by_record.get(row.programming_record_id),
        )
        for row in corrections.itertuples(index=False)
        if graph_challenges_by_record.get(row.programming_record_id)
        != row.corrected_challenge
    ]

    assert mismatches == []


def test_classification_dashboard_code_cells_execute() -> None:
    matplotlib.use("Agg", force=True)
    notebook = json.loads(paths.NOTEBOOK_DIR.joinpath("classification_dashboard.ipynb").read_text(encoding="utf-8"))
    namespace: dict[str, object] = {
        "__name__": "__dashboard_smoke_test__",
        "display": lambda *args, **kwargs: None,
    }

    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        exec(compile(source, f"classification_dashboard.ipynb cell {index}", "exec"), namespace)
        namespace["display"] = lambda *args, **kwargs: None
        if "plt" in namespace:
            namespace["plt"].show = lambda *args, **kwargs: None

    html = paths.CLASSIFICATION_DASHBOARD_HTML_PATH.read_text(encoding="utf-8")
    assert "Genesis Mission Funding and Collaboration Dashboard" in html
    assert "Mission Funding Mix" in html
    assert "Challenge-by-Office Contribution Matrix" in html
    assert "Project Funding Disruptiveness" in html
    assert "Office Portfolios" in html
    assert "Appendix: Validation and Traceability" in html


def test_dashboard_html_contains_executive_sections_and_citations() -> None:
    html = _render_dashboard_html()

    for expected in [
        "Executive Mission Funding Snapshot",
        "Mission Funding Mix",
        "Project Funding Disruptiveness",
        "Challenge Collaboration Groups",
        "Lead Office Funding by Claimed Mission",
        "Challenge Coverage",
        "Reviewed Row-Level Classifications",
    ]:
        assert expected in html
    assert "programming_data.csv row" in html
    assert "programming_data_FYNSP27_wide.csv row" in html
    assert "Executive views use BASELINE, UFR, and ROT rows only" in html
    assert 'data-fynsp-view-button="combined"' in html
    assert 'data-fynsp-view-button="fynsp-2028"' in html
    assert 'data-fynsp-view-button="fynsp-2027"' in html
    assert 'data-fynsp-view="combined"' in html
    assert 'data-office="NA-MB"' in html
    assert 'data-office="NA-PAS"' in html
    assert "NA-MB</td><td>Accelerating Business and Management Processes for National Security Missions" in html
    assert "NA-PAS</td><td>Accelerating Nuclear Enterprise Acquisition" in html
    assert html.find("Project Funding Disruptiveness") < html.find("Appendix: Validation and Traceability")
    assert html.find("Office Pair Collaboration Opportunities") > html.find("Appendix: Validation and Traceability")
    assert "data-disruptiveness-data" in html
    assert 'data-tile-layout="packed-grid"' in html
    assert 'data-x-axis="ai-directness-reversed"' in html
    assert "AI-directness score (more directly AI to the left)" in html
    assert "nearestOpenCell" in html
    assert "targetCell" in html
    assert 'data-disruptiveness-mode="BASELINE"' in html
    assert 'data-disruptiveness-mode="UFR"' in html
    assert 'data-disruptiveness-mode="ROT"' in html
    assert "tile-rot" in html
    assert '<text x="78" y="510" text-anchor="middle">100</text>' in html
    assert '<text x="974" y="510" text-anchor="middle">0</text>' in html


def test_dashboard_challenge_totals_match_positive_funding_rows() -> None:
    html = _render_dashboard_html()
    reviewed = pd.read_csv(paths.REVIEWED_PROGRAMMING_CHALLENGE_CLASSIFICATIONS_PATH)
    positive = reviewed.loc[
        reviewed["funding_level"].astype(str).str.upper().isin(FUNDING_LEVELS)
    ].copy()
    positive["five_year_total"] = pd.to_numeric(
        positive["five_year_total"], errors="coerce"
    ).fillna(0.0)
    totals = positive.pivot_table(
        index="predicted_challenge",
        columns="funding_level",
        values="five_year_total",
        aggfunc="sum",
        fill_value=0.0,
    )

    for challenge, row in totals.iterrows():
        challenge_attr = escape(str(challenge), quote=True)
        assert f'data-challenge="{challenge_attr}"' in html
        for level in FUNDING_LEVELS:
            assert f'data-{level.lower()}-total="{float(row.get(level, 0.0)):.2f}"' in html


def test_dashboard_office_counts_match_funded_challenges() -> None:
    html = _render_dashboard_html()
    reviewed = pd.read_csv(paths.REVIEWED_PROGRAMMING_CHALLENGE_CLASSIFICATIONS_PATH)
    positive = reviewed.loc[
        reviewed["funding_level"].astype(str).str.upper().isin(FUNDING_LEVELS)
    ].copy()
    expected_counts = positive.groupby("normalized_funding_office")[
        "predicted_challenge"
    ].nunique()

    for office, challenge_count in expected_counts.items():
        office_attr = escape(str(office), quote=True)
        assert (
            f'data-office="{office_attr}" '
            f'data-funded-challenge-count="{int(challenge_count)}"'
        ) in html


def test_dashboard_disruptiveness_payload_matches_positive_funding_rows() -> None:
    html = _render_dashboard_html()
    match = re.search(
        r'<script type="application/json" data-disruptiveness-data>(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    assert match is not None
    payload = json.loads(match.group(1).replace("<\\/", "</"))
    rows = payload["rows"]
    reviewed = pd.read_csv(paths.REVIEWED_PROGRAMMING_CHALLENGE_CLASSIFICATIONS_PATH)
    eligible = reviewed.loc[
        reviewed["funding_level"].astype(str).str.upper().isin(FUNDING_LEVELS)
    ].copy()

    assert len(rows) == len(eligible)
    assert {row["funding_level"] for row in rows}.issubset(set(FUNDING_LEVELS))

    eligible["five_year_total"] = pd.to_numeric(
        eligible["five_year_total"], errors="coerce"
    ).fillna(0.0)
    office_totals = eligible.pivot_table(
        index="normalized_funding_office",
        columns="funding_level",
        values="five_year_total",
        aggfunc="sum",
        fill_value=0.0,
    )
    for level in FUNDING_LEVELS:
        if level not in office_totals.columns:
            office_totals[level] = 0.0
    office_totals["ALL"] = office_totals[list(FUNDING_LEVELS)].sum(axis=1)

    by_id = {row["id"]: row for row in rows}
    for source_row in eligible.itertuples(index=False):
        payload_row = by_id[source_row.programming_record_id]
        office_totals_row = office_totals.loc[source_row.normalized_funding_office]
        amount = float(source_row.five_year_total)
        all_total = float(office_totals_row["ALL"])
        assert payload_row["office_share_all"] == pytest.approx(
            0.0 if all_total <= 0 else amount / all_total
        )
        for level in FUNDING_LEVELS:
            level_total = float(office_totals_row[level])
            expected = (
                amount / level_total
                if source_row.funding_level == level and level_total > 0
                else 0.0
            )
            assert payload_row[f"office_share_{level.lower()}"] == pytest.approx(expected)


def test_publish_dashboard_writes_sanitized_shareable_html(tmp_path) -> None:
    published = publish_dashboard(
        output_dir=tmp_path,
        timestamp="20260527_134500",
    )

    assert published.timestamp == "20260527_134500"
    assert published.timestamped_html_path == tmp_path / "genesis_dashboard_20260527_134500.html"
    assert published.latest_html_path == tmp_path / "genesis_dashboard_latest.html"
    assert published.timestamped_html_path.exists()
    assert published.latest_html_path.exists()

    html = published.timestamped_html_path.read_text(encoding="utf-8")
    assert "Genesis Mission Funding and Collaboration Dashboard" in html
    assert "Generated from Programming Ontology data snapshot on 2026-05-27 13:45" in html
    assert str(paths.PROCESSED_DATA_DIR) not in html
    assert "C:\\Users\\Cameron.Webster" not in html
    assert html == published.latest_html_path.read_text(encoding="utf-8")


def test_publish_dashboard_zip_contains_html_and_readme(tmp_path) -> None:
    published = publish_dashboard(
        output_dir=tmp_path,
        create_zip=True,
        timestamp="20260527_134500",
        metadata="Generated from approved dashboard snapshot.",
    )

    assert published.zip_path == tmp_path / "genesis_dashboard_20260527_134500.zip"
    assert published.zip_path.exists()
    with ZipFile(published.zip_path) as archive:
        names = set(archive.namelist())
        assert "genesis_dashboard_20260527_134500.html" in names
        assert "README.txt" in names
        readme = archive.read("README.txt").decode("utf-8")
        html = archive.read("genesis_dashboard_20260527_134500.html").decode("utf-8")

    assert "self-contained" in readme
    assert "Generated from approved dashboard snapshot." in readme
    assert "Generated from approved dashboard snapshot." in html
    assert str(paths.PROCESSED_DATA_DIR) not in html
