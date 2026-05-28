from __future__ import annotations

import pandas as pd
import pytest

from programming_ontology.build_ontology import build_combined_ontology
from programming_ontology.classify_programming import (
    ChallengeProfile,
    apply_sme_overrides,
    build_challenge_profiles,
    build_office_claims,
    build_programming_evidence_profiles,
    build_validation_outputs,
    classify_programming_records,
    classification_results_to_dataframe,
    enrich_graph_with_reviewed_classifications,
    enrich_graph_with_classifications,
    is_valid_claim_office,
    normalize_office_code,
    rollup_office_to_claimed_parent,
)


def test_office_normalization_and_rollups() -> None:
    claimed = {"NA20", "NA80", "NAIM", "NAESH", "NA-PAS", "NA-MB"}

    assert normalize_office_code("NA-IM") == "NAIM"
    assert normalize_office_code("NA-ESH") == "NAESH"
    assert normalize_office_code("NA-PAS") == "NA-PAS"
    assert normalize_office_code("NAPAS") == "NA-PAS"
    assert normalize_office_code("NA-MB") == "NA-MB"
    assert normalize_office_code("NAMB") == "NA-MB"
    assert rollup_office_to_claimed_parent("NA22", claimed) == "NA20"
    assert rollup_office_to_claimed_parent("NA84", claimed) == "NA80"
    assert rollup_office_to_claimed_parent("NA30", claimed) == "NA30"
    assert is_valid_claim_office("NA-PAS")
    assert is_valid_claim_office("NA-MB")
    assert not is_valid_claim_office("NNSA Field Offices")


def test_claims_include_lead_and_partners_but_exclude_field_offices() -> None:
    graph = build_combined_ontology()
    challenges = build_challenge_profiles(graph)
    claims = build_office_claims(graph, challenges)

    claim_pairs = {(claim.challenge_name, claim.claimed_office, claim.claim_role) for claim in claims}
    assert (
        "Streamlining Production, Removing Red Tape, and Ensuring Safety in the Nuclear Enterprise",
        "NA90",
        "lead",
    ) in claim_pairs
    assert (
        "Streamlining Production, Removing Red Tape, and Ensuring Safety in the Nuclear Enterprise",
        "NA70",
        "partner",
    ) in claim_pairs
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
    assert all(claim.claimed_office != "NNSA FIELD OFFICES" for claim in claims)


def test_row_level_classification_preserves_source_funding_office() -> None:
    graph = build_combined_ontology()
    challenges = build_challenge_profiles(graph)
    claims = build_office_claims(graph, challenges)
    claimed_offices = {claim.claimed_office for claim in claims}
    evidence = build_programming_evidence_profiles(graph, claimed_offices=claimed_offices)
    results = classify_programming_records(evidence[:1], challenges)
    frame = classification_results_to_dataframe(results)

    assert "lead_program_office" not in frame.columns
    assert "support_program_offices" not in frame.columns
    assert frame.loc[0, "programming_record_id"].startswith("programming_record:")
    assert frame.loc[0, "source_sub_office"] == "NA11"
    assert frame.loc[0, "normalized_funding_office"] == "NA11"
    assert frame.loc[0, "funding_level"] == "BASELINE"
    assert frame.loc[0, "scope_description"]
    assert frame.loc[0, "five_year_total"] > 0


def test_graph_enrichment_uses_evidence_edges_not_assignment_edges() -> None:
    graph = build_combined_ontology()
    challenges = build_challenge_profiles(graph)
    claims = build_office_claims(graph, challenges)
    evidence = build_programming_evidence_profiles(
        graph,
        claimed_offices={claim.claimed_office for claim in claims},
    )
    results = classify_programming_records(evidence[:5], challenges)
    enriched = enrich_graph_with_classifications(graph, results)
    relations = {
        data.get("relation")
        for _, _, _, data in enriched.edges(keys=True, data=True)
    }

    assert "classified_as_challenge" in relations
    assert "funds_challenge_evidence" in relations
    assert "assigned_lead_program_office" not in relations
    assert "assigned_support_program_office" not in relations


def test_claim_validation_statuses() -> None:
    claims = pd.DataFrame(
        [
            {
                "challenge_name": "Challenge A",
                "claimed_office": "NA20",
                "claim_role": "lead",
                "challenge_has_mstc_context": True,
            },
            {
                "challenge_name": "Challenge B",
                "claimed_office": "NA80",
                "claim_role": "partner",
                "challenge_has_mstc_context": True,
            },
            {
                "challenge_name": "Challenge C",
                "claimed_office": "NA11",
                "claim_role": "partner",
                "challenge_has_mstc_context": True,
            },
        ]
    )
    classifications = pd.DataFrame(
        [
            {
                "programming_record_id": "row-1",
                "wbs_name": "WBS 1",
                "predicted_challenge": "Challenge A",
                "normalized_funding_office": "NA20",
                "funding_level": "BASELINE",
                "validation_funding_level": True,
                "five_year_total": 10.0,
                "confidence": "medium",
                "review_flag": False,
            },
            {
                "programming_record_id": "row-2",
                "wbs_name": "WBS 2",
                "predicted_challenge": "Challenge B",
                "normalized_funding_office": "NA80",
                "funding_level": "UFR",
                "validation_funding_level": True,
                "five_year_total": 20.0,
                "confidence": "low",
                "review_flag": True,
            },
        ]
    )

    validation = build_validation_outputs(classifications, claims)["office_claim_validation"]
    statuses = validation.set_index("challenge_name")["validation_status"].to_dict()

    assert statuses["Challenge A"] == "validated"
    assert statuses["Challenge B"] == "needs_review"
    assert statuses["Challenge C"] == "unfunded"


def test_sme_corrected_challenge_overrides_accept_decision_with_warning() -> None:
    classifications = _sample_classifications()
    review = pd.DataFrame(
        [
            {
                "programming_record_id": "row-1",
                "source_row_number": 1,
                "review_decision": "accept",
                "corrected_challenge": "Challenge B",
                "sme_notes": "SME correction",
                "reviewer": "Reviewer",
                "review_date": "2026-05-28",
            }
        ]
    )

    with pytest.warns(UserWarning, match="review_decision is not 'change'"):
        reviewed = apply_sme_overrides(classifications, review, _sample_challenges())

    row = reviewed.loc[reviewed["programming_record_id"].eq("row-1")].iloc[0]
    assert row["predicted_challenge"] == "Challenge B"
    assert row["challenge_node_id"] == "challenge-b"
    assert bool(row["sme_override_applied"]) is True
    assert bool(row["review_flag"]) is False


def test_sme_corrected_challenge_overrides_literal_review_decision_with_warning() -> None:
    review = pd.DataFrame(
        [
            {
                "programming_record_id": "row-1",
                "source_row_number": 1,
                "review_decision": "review_decision",
                "corrected_challenge": "Challenge B",
            }
        ]
    )

    with pytest.warns(UserWarning, match="row-1"):
        reviewed = apply_sme_overrides(
            _sample_classifications(),
            review,
            _sample_challenges(),
        )

    assert reviewed.loc[0, "predicted_challenge"] == "Challenge B"
    assert reviewed.loc[0, "sme_review_decision"] == "review_decision"


def test_sme_invalid_corrected_challenge_raises_without_change_decision() -> None:
    review = pd.DataFrame(
        [
            {
                "programming_record_id": "row-1",
                "source_row_number": 1,
                "review_decision": "accept",
                "corrected_challenge": "Not A Challenge",
            }
        ]
    )

    with pytest.raises(ValueError, match="invalid corrected_challenge"):
        apply_sme_overrides(_sample_classifications(), review, _sample_challenges())


def test_sme_review_decisions_without_correction_keep_existing_behavior() -> None:
    review = pd.DataFrame(
        [
            {
                "programming_record_id": "row-1",
                "source_row_number": 1,
                "review_decision": "needs_discussion",
                "corrected_challenge": "",
            },
            {
                "programming_record_id": "row-2",
                "source_row_number": 2,
                "review_decision": "accept",
                "corrected_challenge": "",
            },
        ]
    )

    reviewed = apply_sme_overrides(_sample_classifications(), review, _sample_challenges())

    by_id = reviewed.set_index("programming_record_id")
    assert by_id.loc["row-1", "predicted_challenge"] == "Challenge A"
    assert bool(by_id.loc["row-1", "review_flag"]) is True
    assert by_id.loc["row-2", "predicted_challenge"] == "Challenge B"
    assert bool(by_id.loc["row-2", "review_flag"]) is False


def test_reviewed_classification_graph_edges_use_corrected_challenges() -> None:
    graph = build_combined_ontology()
    reviewed = pd.DataFrame(
        [
            {
                "programming_record_id": "programming_record:test",
                "source_file": "programming_data.csv",
                "source_row_number": 999999,
                "predicted_challenge": "Challenge B",
                "challenge_node_id": "challenge-b",
                "validation_funding_level": True,
                "five_year_total": 100.0,
                "normalized_funding_office": "NA11",
                "confidence": "medium",
                "review_flag": False,
                "rationale": "SME correction",
                "funding_level": "UFR",
            }
        ]
    )

    enriched = enrich_graph_with_reviewed_classifications(graph, reviewed)
    classified_targets = [
        enriched.nodes[target].get("label")
        for source, target, _, data in enriched.edges(keys=True, data=True)
        if data.get("relation") == "classified_as_challenge"
        and data.get("source_rows") == "999999"
    ]

    assert classified_targets == ["Challenge B"]


def _sample_challenges() -> dict[str, ChallengeProfile]:
    return {
        "Challenge A": ChallengeProfile(
            challenge_node_id="challenge-a",
            challenge_name="Challenge A",
            problem_statements=("Problem A",),
            proposed_solutions=(),
            nnsa_advantages=(),
            expected_outcomes=(),
        ),
        "Challenge B": ChallengeProfile(
            challenge_node_id="challenge-b",
            challenge_name="Challenge B",
            problem_statements=("Problem B",),
            proposed_solutions=(),
            nnsa_advantages=(),
            expected_outcomes=(),
        ),
    }


def _sample_classifications() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "programming_record_id": "row-1",
                "predicted_challenge": "Challenge A",
                "challenge_node_id": "challenge-a",
                "challenge_has_mstc_context": True,
                "review_flag": True,
                "review_reasons": "low_confidence",
                "rationale": "Classifier rationale.",
            },
            {
                "programming_record_id": "row-2",
                "predicted_challenge": "Challenge B",
                "challenge_node_id": "challenge-b",
                "challenge_has_mstc_context": True,
                "review_flag": True,
                "review_reasons": "low_confidence",
                "rationale": "Classifier rationale.",
            },
        ]
    )
