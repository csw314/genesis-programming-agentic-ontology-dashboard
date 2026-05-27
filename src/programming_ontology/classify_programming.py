"""Classify programming funding rows and validate program office claims.

The core question for this package is whether offices that claimed equity in a
Genesis Mission Challenge have BASELINE or UFR funding rows that map to that
challenge. Program office mappings are treated as claims, not as assignments of
WBS items to offices.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha1
from pathlib import Path
import json
import re
from typing import Any, Callable, Mapping, Sequence

import networkx as nx
import pandas as pd

from .asksage_classifier import DEFAULT_ASKSAGE_MODEL
from .openai_classifier import DEFAULT_OPENAI_MODEL


LLMClassifier = Callable[[dict[str, Any]], Mapping[str, Any] | str]

CLASSIFICATION_SOURCE = "classification_pipeline"
CONFIDENCE_LEVELS = {"high", "medium", "low"}
VALIDATION_FUNDING_LEVELS = {"BASELINE", "UFR"}
VALID_REVIEW_DECISIONS = {"accept", "change", "needs_discussion", "out_of_scope"}
NON_VALIDATABLE_OFFICES = {"NNSA FIELD OFFICES"}
SPECIAL_OFFICE_CODES = {"NAIM", "NAESH", "NA-PAS", "NA-MB"}
OFFICE_CODE_ALIASES = {
    "NAIM": "NAIM",
    "NAESH": "NAESH",
    "NAPAS": "NA-PAS",
    "NAMB": "NA-MB",
}
CHALLENGE_LEAD_OFFICE_OVERRIDES = {
    "Accelerating Business and Management Processes for National Security Missions": "NA-MB",
    "Accelerating Nuclear Enterprise Acquisition": "NA-PAS",
}

CLASSIFICATION_CONTEXT_COLUMNS = {
    "Program Int. Area",
    "Process Imp. Area",
    "Program Request",
    "Acquisition Type",
    "Acquisition Name",
}

BUDGET_COLUMNS = ("FY2028", "FY2029", "FY2030", "FY2031", "FY2032")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "with",
}


@dataclass(frozen=True)
class ChallengeProfile:
    """Context describing a Genesis Mission Challenge."""

    challenge_node_id: str
    challenge_name: str
    problem_statements: tuple[str, ...]
    proposed_solutions: tuple[str, ...]
    nnsa_advantages: tuple[str, ...]
    expected_outcomes: tuple[str, ...]


@dataclass(frozen=True)
class OfficeMapping:
    """Challenge-to-office mapping from ProgramOffice_Mapping.csv."""

    challenge_name: str
    challenge_node_id: str
    lead_program_office: str
    lead_program_office_node_id: str
    support_program_offices: tuple[str, ...]
    support_program_office_node_ids: tuple[str, ...]


@dataclass(frozen=True)
class OfficeClaim:
    """A single office claim of equity in a Genesis Mission Challenge."""

    challenge_name: str
    challenge_node_id: str
    claimed_office: str
    claim_role: str
    challenge_has_mstc_context: bool


@dataclass(frozen=True)
class EvidenceProfile:
    """Row-level programming evidence to classify against a challenge."""

    programming_record_id: str
    source_file: str
    source_row_number: int
    wbs_name: str
    wbs_node_id: str
    wbs_code: str
    source_sub_office: str
    normalized_funding_office: str
    stat_l3: str
    stat_l4: str
    stat_l5: str
    funding_level: str
    scope_description: str
    program_interest_area: str
    process_improvement_area: str
    program_request: str
    acquisition_context: tuple[str, ...]
    fy2028: float
    fy2029: float
    fy2030: float
    fy2031: float
    fy2032: float
    five_year_total: float
    validation_funding_level: bool
    additional_context: tuple[str, ...]
    evidence_node_ids: tuple[str, ...]


@dataclass(frozen=True)
class CandidateChallenge:
    """Lexically ranked challenge candidate for one evidence row."""

    challenge_name: str
    challenge_node_id: str
    score: float
    matched_terms: tuple[str, ...]
    has_mstc_context: bool


@dataclass(frozen=True)
class ClassificationResult:
    """Final row-level challenge classification for one programming record."""

    programming_record_id: str
    source_file: str
    source_row_number: int
    wbs_name: str
    wbs_node_id: str
    wbs_code: str
    source_sub_office: str
    normalized_funding_office: str
    stat_l3: str
    stat_l4: str
    stat_l5: str
    funding_level: str
    scope_description: str
    program_interest_area: str
    process_improvement_area: str
    program_request: str
    acquisition_context: tuple[str, ...]
    fy2028: float
    fy2029: float
    fy2030: float
    fy2031: float
    fy2032: float
    five_year_total: float
    validation_funding_level: bool
    predicted_challenge: str
    challenge_node_id: str
    challenge_has_mstc_context: bool
    confidence: str
    review_flag: bool
    rationale: str
    top_candidate_challenges: tuple[str, ...]
    additional_programming_context: tuple[str, ...]
    evidence_node_ids: tuple[str, ...]
    review_reasons: tuple[str, ...]


def normalize_office_code(value: Any) -> str:
    """Normalize office labels without rolling child offices to parents."""

    if value is None or pd.isna(value):
        return ""

    label = re.sub(r"\s+", " ", str(value)).strip().upper()
    if not label:
        return ""

    compact = label.replace("-", "").replace(" ", "")
    if compact.startswith("NA") and compact[2:].isdigit():
        return f"NA{int(compact[2:]):02d}"
    if compact in OFFICE_CODE_ALIASES:
        return OFFICE_CODE_ALIASES[compact]
    return label.replace("-", "")


def is_valid_claim_office(value: Any) -> bool:
    """Return whether an office can be validated against programming data."""

    code = normalize_office_code(value)
    if code in NON_VALIDATABLE_OFFICES:
        return False
    return bool(re.fullmatch(r"NA\d{2}", code) or code in SPECIAL_OFFICE_CODES)


def rollup_office_to_claimed_parent(
    value: Any,
    claimed_offices: set[str] | frozenset[str],
) -> str:
    """Roll child offices to a claimed parent office when applicable."""

    code = normalize_office_code(value)
    if not code or code in claimed_offices:
        return code

    match = re.fullmatch(r"NA(\d{2})", code)
    if not match:
        return code

    number = int(match.group(1))
    parent = f"NA{number // 10 * 10:02d}"
    if parent != code and parent in claimed_offices:
        return parent
    return code


def build_challenge_profiles(graph: nx.MultiDiGraph) -> dict[str, ChallengeProfile]:
    """Build challenge profiles keyed by challenge name.

    Mapping-only challenges are included as title-only profiles because the
    Program Office mapping can contain claims that are absent from MSTC_data.csv.
    """

    profiles = {}
    for challenge_node_id, data in graph.nodes(data=True):
        if data.get("node_type") != "mstc_challenge":
            continue

        challenge_name = data.get("label", challenge_node_id)
        profiles[challenge_name] = ChallengeProfile(
            challenge_node_id=challenge_node_id,
            challenge_name=challenge_name,
            problem_statements=_labels_for_nodes(
                graph,
                _target_nodes(graph, challenge_node_id, "has_problem_statement"),
            ),
            proposed_solutions=_labels_for_nodes(
                graph,
                _target_nodes(graph, challenge_node_id, "has_proposed_solution"),
            ),
            nnsa_advantages=_labels_for_nodes(
                graph,
                _target_nodes(graph, challenge_node_id, "has_nnsa_advantage"),
            ),
            expected_outcomes=_labels_for_nodes(
                graph,
                _target_nodes(graph, challenge_node_id, "has_expected_outcome"),
            ),
        )

    return dict(sorted(profiles.items()))


def build_office_mappings(graph: nx.MultiDiGraph) -> dict[str, OfficeMapping]:
    """Build raw challenge-to-office mappings from ProgramOffice_Mapping.csv."""

    mappings = {}
    for challenge_node_id, data in graph.nodes(data=True):
        if data.get("node_type") != "mstc_challenge":
            continue

        lead_nodes = _target_nodes(graph, challenge_node_id, "led_by_program_office")
        support_nodes = _target_nodes(graph, challenge_node_id, "has_partner_organization")
        if not lead_nodes and not support_nodes:
            continue

        lead_node = lead_nodes[0] if lead_nodes else ""
        challenge_name = data.get("label", challenge_node_id)
        lead_office = (
            normalize_office_code(graph.nodes[lead_node].get("label", ""))
            if lead_node
            else ""
        )
        lead_office = CHALLENGE_LEAD_OFFICE_OVERRIDES.get(
            challenge_name,
            lead_office,
        )
        mappings[challenge_name] = OfficeMapping(
            challenge_name=challenge_name,
            challenge_node_id=challenge_node_id,
            lead_program_office=lead_office,
            lead_program_office_node_id=lead_node,
            support_program_offices=tuple(
                normalize_office_code(label)
                for label in _labels_for_nodes(graph, support_nodes)
            ),
            support_program_office_node_ids=tuple(support_nodes),
        )

    return dict(sorted(mappings.items()))


def build_office_claims(
    graph: nx.MultiDiGraph,
    challenges: Mapping[str, ChallengeProfile] | None = None,
) -> list[OfficeClaim]:
    """Build valid lead and partner office claims for challenge validation."""

    if challenges is None:
        challenges = build_challenge_profiles(graph)

    claims = []
    for mapping in build_office_mappings(graph).values():
        challenge = challenges.get(mapping.challenge_name)
        has_context = bool(challenge and _has_mstc_context(challenge))
        challenge_node_id = (
            challenge.challenge_node_id if challenge else mapping.challenge_node_id
        )

        if is_valid_claim_office(mapping.lead_program_office):
            claims.append(
                OfficeClaim(
                    challenge_name=mapping.challenge_name,
                    challenge_node_id=challenge_node_id,
                    claimed_office=normalize_office_code(mapping.lead_program_office),
                    claim_role="lead",
                    challenge_has_mstc_context=has_context,
                )
            )

        for office in mapping.support_program_offices:
            if not is_valid_claim_office(office):
                continue
            claims.append(
                OfficeClaim(
                    challenge_name=mapping.challenge_name,
                    challenge_node_id=challenge_node_id,
                    claimed_office=normalize_office_code(office),
                    claim_role="partner",
                    challenge_has_mstc_context=has_context,
                )
            )

    role_order = {"lead": 0, "partner": 1}
    return sorted(
        dict.fromkeys(claims),
        key=lambda claim: (
            claim.challenge_name.casefold(),
            role_order.get(claim.claim_role, 99),
            claim.claimed_office,
        ),
    )


def office_claims_to_dataframe(claims: Sequence[OfficeClaim]) -> pd.DataFrame:
    """Convert office claims to a DataFrame."""

    return pd.DataFrame([asdict(claim) for claim in claims])


def build_programming_evidence_profiles(
    graph: nx.MultiDiGraph,
    *,
    claimed_offices: set[str] | frozenset[str] | None = None,
) -> list[EvidenceProfile]:
    """Build one evidence profile for each programming source row."""

    claimed = frozenset(claimed_offices or set())
    profiles = []

    for record_id, data in graph.nodes(data=True):
        if data.get("node_type") != "programming_record":
            continue

        source_file = _first_delimited(data.get("source_files", ""))
        row_number = int(data.get("row_number", 0))
        source_sub_office = _record_target_label(graph, record_id, "belongs_to_sub_office")
        funding_level = _record_target_label(graph, record_id, "has_funding_level")
        scope_node = _record_target_node(graph, record_id, "has_scope_description")
        scope_description = graph.nodes[scope_node].get("label", "") if scope_node else ""
        wbs_node_id = _record_target_node(graph, record_id, "has_wbs")
        wbs_name = graph.nodes[wbs_node_id].get("label", "") if wbs_node_id else ""
        budget = {
            column.casefold(): _collect_single_programming_column_value(
                graph, record_id, column
            )
            for column in BUDGET_COLUMNS
        }
        year_values = {
            column.casefold(): _to_float(budget[column.casefold()]) or 0.0
            for column in BUDGET_COLUMNS
        }
        five_year_total = sum(year_values.values())
        additional_context, additional_context_nodes = (
            _collect_additional_programming_context(graph, [record_id])
        )

        profiles.append(
            EvidenceProfile(
                programming_record_id=record_id,
                source_file=source_file,
                source_row_number=row_number,
                wbs_name=wbs_name,
                wbs_node_id=wbs_node_id,
                wbs_code=_collect_single_programming_column_value(graph, record_id, "WBS"),
                source_sub_office=source_sub_office,
                normalized_funding_office=rollup_office_to_claimed_parent(
                    source_sub_office,
                    claimed,
                ),
                stat_l3=_record_target_label(graph, record_id, "classified_by_stat_l3"),
                stat_l4=_record_target_label(graph, record_id, "classified_by_stat_l4"),
                stat_l5=_record_target_label(graph, record_id, "classified_by_stat_l5"),
                funding_level=funding_level,
                scope_description=scope_description,
                program_interest_area=_collect_single_programming_column_value(
                    graph,
                    record_id,
                    "Program Int. Area",
                ),
                process_improvement_area=_collect_single_programming_column_value(
                    graph,
                    record_id,
                    "Process Imp. Area",
                ),
                program_request=_collect_single_programming_column_value(
                    graph,
                    record_id,
                    "Program Request",
                ),
                acquisition_context=_collect_programming_column_context(
                    graph,
                    [record_id],
                    {"Acquisition Type", "Acquisition Name"},
                ),
                fy2028=year_values["fy2028"],
                fy2029=year_values["fy2029"],
                fy2030=year_values["fy2030"],
                fy2031=year_values["fy2031"],
                fy2032=year_values["fy2032"],
                five_year_total=five_year_total,
                validation_funding_level=funding_level in VALIDATION_FUNDING_LEVELS,
                additional_context=additional_context,
                evidence_node_ids=tuple(
                    node
                    for node in [record_id, wbs_node_id, scope_node, *additional_context_nodes]
                    if node
                ),
            )
        )

    return sorted(profiles, key=lambda profile: profile.source_row_number)


def rank_challenge_candidates(
    evidence: EvidenceProfile,
    challenges: Mapping[str, ChallengeProfile],
    top_k: int = 5,
) -> list[CandidateChallenge]:
    """Rank candidate challenges using deterministic lexical overlap."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    evidence_text = _evidence_text(evidence)
    evidence_tokens = _tokens(evidence_text)
    candidates = []

    for challenge in challenges.values():
        challenge_text = _challenge_text(challenge)
        challenge_tokens = _tokens(challenge_text)
        matched = tuple(sorted(evidence_tokens & challenge_tokens))
        score = _candidate_score(
            evidence,
            challenge,
            evidence_tokens=evidence_tokens,
            challenge_tokens=challenge_tokens,
            matched_terms=set(matched),
        )
        candidates.append(
            CandidateChallenge(
                challenge_name=challenge.challenge_name,
                challenge_node_id=challenge.challenge_node_id,
                score=round(score, 4),
                matched_terms=matched[:20],
                has_mstc_context=_has_mstc_context(challenge),
            )
        )

    return sorted(
        candidates,
        key=lambda candidate: (-candidate.score, candidate.challenge_name.casefold()),
    )[:top_k]


def classify_programming_records(
    evidence_rows: list[EvidenceProfile],
    challenges: Mapping[str, ChallengeProfile],
    offices: Mapping[str, OfficeMapping] | None = None,
    llm_classifier: LLMClassifier | None = None,
    *,
    top_k: int = 5,
) -> list[ClassificationResult]:
    """Classify row-level programming evidence against Genesis challenges."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    eligible_challenges = _eligible_challenge_profiles(challenges, offices)
    results = []

    for evidence in evidence_rows:
        candidates = rank_challenge_candidates(evidence, eligible_challenges, top_k=top_k)
        if not candidates:
            raise ValueError(
                "No challenge candidates were returned for programming record "
                f"{evidence.programming_record_id!r}."
            )
        payload = _classification_payload(evidence, candidates, eligible_challenges)

        if llm_classifier is None:
            raw_decision = _lexical_classifier(payload)
        else:
            raw_decision = llm_classifier(payload)

        results.append(
            _build_result_from_decision(
                evidence,
                candidates,
                _parse_classifier_response(raw_decision),
                challenges=eligible_challenges,
            )
        )

    return results


def export_classifications(
    results: list[ClassificationResult],
    path: str | Path,
) -> Path:
    """Export row-level classification results to CSV and return the path."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    classification_results_to_dataframe(results).to_csv(output_path, index=False)
    return output_path


def classification_results_to_dataframe(
    results: list[ClassificationResult],
) -> pd.DataFrame:
    """Convert classification results to a flat DataFrame."""

    rows = []
    for result in results:
        row = asdict(result)
        for key, value in list(row.items()):
            if isinstance(value, tuple):
                row[key] = "; ".join(str(item) for item in value)
        rows.append(row)
    return pd.DataFrame(rows)


def candidate_rankings_to_dataframe(
    evidence_rows: list[EvidenceProfile],
    challenges: Mapping[str, ChallengeProfile],
    offices: Mapping[str, OfficeMapping] | None = None,
    *,
    top_k: int = 5,
) -> pd.DataFrame:
    """Return a preview table of deterministic candidate rankings."""

    eligible_challenges = _eligible_challenge_profiles(challenges, offices)
    rows = []
    for evidence in evidence_rows:
        for rank, candidate in enumerate(
            rank_challenge_candidates(evidence, eligible_challenges, top_k=top_k),
            start=1,
        ):
            rows.append(
                {
                    "programming_record_id": evidence.programming_record_id,
                    "source_row_number": evidence.source_row_number,
                    "wbs_name": evidence.wbs_name,
                    "rank": rank,
                    "challenge_name": candidate.challenge_name,
                    "score": candidate.score,
                    "matched_terms": "; ".join(candidate.matched_terms),
                    "challenge_has_mstc_context": candidate.has_mstc_context,
                }
            )
    return pd.DataFrame(rows)


def classify_graph(
    graph: nx.MultiDiGraph,
    *,
    classifier: str = "lexical",
    model: str | None = None,
    top_k: int = 5,
    reasoning_effort: str = "medium",
    limit: int | None = None,
    audit_cache_path: str | Path | None = None,
    resume_cache: bool = False,
    openai_client: Any | None = None,
    asksage_client: Any | None = None,
    asksage_base_url: str | None = None,
    asksage_openai_url: str | None = None,
) -> list[ClassificationResult]:
    """Classify programming rows by traversing an ontology graph."""

    challenges = build_challenge_profiles(graph)
    offices = build_office_mappings(graph)
    claims = build_office_claims(graph, challenges)
    evidence_rows = build_programming_evidence_profiles(
        graph,
        claimed_offices={claim.claimed_office for claim in claims},
    )
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be greater than zero when provided")
        evidence_rows = evidence_rows[:limit]

    llm_classifier: LLMClassifier | None = None
    if classifier == "openai":
        from .openai_classifier import OpenAIChallengeClassifier

        llm_classifier = OpenAIChallengeClassifier(
            model=model or DEFAULT_OPENAI_MODEL,
            reasoning_effort=reasoning_effort,
            client=openai_client,
            audit_cache_path=audit_cache_path,
            resume_cache=resume_cache,
        )
    elif classifier == "asksage":
        from .asksage_classifier import AskSageChallengeClassifier

        llm_classifier = AskSageChallengeClassifier(
            model=model,
            reasoning_effort=reasoning_effort,
            base_url=asksage_base_url,
            openai_url=asksage_openai_url,
            client=asksage_client,
            audit_cache_path=audit_cache_path,
            resume_cache=resume_cache,
        )
    elif classifier != "lexical":
        raise ValueError("classifier must be 'lexical', 'openai', or 'asksage'")

    return classify_programming_records(
        evidence_rows,
        challenges,
        offices=offices,
        llm_classifier=llm_classifier,
        top_k=top_k,
    )


def classify_graph_with_openai(
    graph: nx.MultiDiGraph,
    *,
    model: str | None = None,
    top_k: int = 5,
    reasoning_effort: str = "medium",
    limit: int | None = None,
    audit_cache_path: str | Path | None = None,
    resume_cache: bool = False,
    openai_client: Any | None = None,
) -> list[ClassificationResult]:
    """Classify graph evidence with the OpenAI Responses API."""

    return classify_graph(
        graph,
        classifier="openai",
        model=model,
        top_k=top_k,
        reasoning_effort=reasoning_effort,
        limit=limit,
        audit_cache_path=audit_cache_path,
        resume_cache=resume_cache,
        openai_client=openai_client,
    )


def classify_graph_with_asksage(
    graph: nx.MultiDiGraph,
    *,
    model: str | None = None,
    top_k: int = 5,
    reasoning_effort: str = "medium",
    limit: int | None = None,
    audit_cache_path: str | Path | None = None,
    resume_cache: bool = False,
    asksage_client: Any | None = None,
    asksage_base_url: str | None = None,
    asksage_openai_url: str | None = None,
) -> list[ClassificationResult]:
    """Classify graph evidence with Ask Sage's OpenAI-compatible API."""

    return classify_graph(
        graph,
        classifier="asksage",
        model=model,
        top_k=top_k,
        reasoning_effort=reasoning_effort,
        limit=limit,
        audit_cache_path=audit_cache_path,
        resume_cache=resume_cache,
        asksage_client=asksage_client,
        asksage_base_url=asksage_base_url,
        asksage_openai_url=asksage_openai_url,
    )


def enrich_graph_with_classifications(
    graph: nx.MultiDiGraph,
    results: list[ClassificationResult],
) -> nx.MultiDiGraph:
    """Return a copy of ``graph`` with row-level classification evidence edges."""

    enriched = graph.copy()

    for result in results:
        record_id = _ensure_node(
            enriched,
            result.programming_record_id,
            f"{result.source_file} row {result.source_row_number}",
            "programming_record",
        )
        challenge_node_id = _ensure_node(
            enriched,
            result.challenge_node_id,
            result.predicted_challenge,
            "mstc_challenge",
        )
        _add_classification_edge(
            enriched,
            record_id,
            challenge_node_id,
            "classified_as_challenge",
            result,
        )

        funding_office_node = _find_node(
            enriched,
            "sub_office",
            result.normalized_funding_office,
        )
        if result.validation_funding_level and result.five_year_total > 0:
            if not funding_office_node:
                funding_office_node = _ensure_node(
                    enriched,
                    "",
                    result.normalized_funding_office,
                    "sub_office",
                )
            _add_classification_edge(
                enriched,
                funding_office_node,
                challenge_node_id,
                "funds_challenge_evidence",
                result,
            )

    enriched.graph["classification_source"] = CLASSIFICATION_SOURCE
    return enriched


def build_review_template(classifications_df: pd.DataFrame) -> pd.DataFrame:
    """Build a row-level SME review template from classification results."""

    template = pd.DataFrame(
        {
            "programming_record_id": classifications_df["programming_record_id"],
            "source_row_number": classifications_df["source_row_number"],
            "wbs_name": classifications_df["wbs_name"],
            "source_sub_office": classifications_df["source_sub_office"],
            "normalized_funding_office": classifications_df["normalized_funding_office"],
            "funding_level": classifications_df["funding_level"],
            "current_predicted_challenge": classifications_df["predicted_challenge"],
            "current_confidence": classifications_df["confidence"],
            "current_review_flag": classifications_df["review_flag"].map(_bool_text),
            "review_decision": classifications_df["review_flag"].map(
                lambda value: "needs_discussion" if _truthy(value) else "accept"
            ),
            "corrected_challenge": "",
            "sme_notes": "",
            "reviewer": "",
            "review_date": "",
        }
    )
    return template


def load_or_initialize_sme_review(
    path: str | Path,
    classifications_df: pd.DataFrame,
) -> tuple[pd.DataFrame, bool]:
    """Load, migrate, or create the row-level SME review CSV."""

    review_path = Path(path)
    template = build_review_template(classifications_df)
    created = False

    if review_path.exists():
        existing = pd.read_csv(review_path, dtype=str).fillna("")
        editable = ["review_decision", "corrected_challenge", "sme_notes", "reviewer", "review_date"]
        if "programming_record_id" in existing.columns:
            review = template.set_index("programming_record_id")
            existing_indexed = existing.drop_duplicates(
                "programming_record_id",
                keep="last",
            ).set_index("programming_record_id")
            overlap = review.index.intersection(existing_indexed.index)
            for column in editable:
                if column in existing_indexed.columns:
                    review.loc[overlap, column] = existing_indexed.loc[overlap, column]
            review = review.reset_index()
        elif "wbs_name" in existing.columns:
            review = _migrate_wbs_review(existing, template, editable)
        else:
            review = template
            created = True
    else:
        review = template
        created = True

    review["review_decision"] = (
        review["review_decision"].fillna("").astype(str).str.strip().str.lower()
    )
    review.loc[
        ~review["review_decision"].isin(VALID_REVIEW_DECISIONS),
        "review_decision",
    ] = "needs_discussion"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(review_path, index=False)
    return review, created


def apply_sme_overrides(
    classifications_df: pd.DataFrame,
    review_df: pd.DataFrame,
    challenges: Mapping[str, ChallengeProfile],
) -> pd.DataFrame:
    """Apply row-level SME review decisions without changing funding offices."""

    invalid_decisions = review_df.loc[
        ~review_df["review_decision"].isin(VALID_REVIEW_DECISIONS)
    ]
    if not invalid_decisions.empty:
        raise ValueError("SME review file contains invalid review_decision values")

    valid_challenges = set(challenges)
    change_rows = review_df.loc[review_df["review_decision"] == "change"].copy()
    change_rows["corrected_challenge"] = (
        change_rows["corrected_challenge"].fillna("").astype(str).str.strip()
    )
    invalid_changes = change_rows.loc[
        ~change_rows["corrected_challenge"].isin(valid_challenges)
    ]
    if not invalid_changes.empty:
        raise ValueError("SME review file contains change rows with invalid corrected_challenge values")

    reviewed = classifications_df.copy()
    reviewed["original_predicted_challenge"] = reviewed["predicted_challenge"]
    reviewed["original_review_flag"] = reviewed["review_flag"].map(_truthy)
    reviewed["sme_review_decision"] = "needs_discussion"
    reviewed["sme_override_applied"] = False
    reviewed["sme_notes"] = ""
    reviewed["sme_reviewer"] = ""
    reviewed["sme_review_date"] = ""
    reviewed["review_flag"] = reviewed["review_flag"].map(_truthy)

    review_by_record = review_df.drop_duplicates(
        "programming_record_id",
        keep="last",
    ).set_index("programming_record_id")
    for record_id, review_row in review_by_record.iterrows():
        row_mask = reviewed["programming_record_id"] == record_id
        if not row_mask.any():
            continue

        decision = review_row["review_decision"]
        reviewed.loc[row_mask, "sme_review_decision"] = decision
        reviewed.loc[row_mask, "sme_notes"] = str(review_row.get("sme_notes", ""))
        reviewed.loc[row_mask, "sme_reviewer"] = str(review_row.get("reviewer", ""))
        reviewed.loc[row_mask, "sme_review_date"] = str(review_row.get("review_date", ""))

        if decision == "accept":
            reviewed.loc[row_mask, "review_flag"] = False
            reviewed.loc[row_mask, "review_reasons"] = reviewed.loc[
                row_mask,
                "review_reasons",
            ].map(lambda value: append_reason(value, "sme_accepted"))
        elif decision == "change":
            corrected = str(review_row["corrected_challenge"]).strip()
            challenge = challenges[corrected]
            notes = str(review_row.get("sme_notes", "")).strip()
            reviewed.loc[row_mask, "predicted_challenge"] = corrected
            reviewed.loc[row_mask, "challenge_node_id"] = challenge.challenge_node_id
            reviewed.loc[row_mask, "challenge_has_mstc_context"] = _has_mstc_context(
                challenge
            )
            reviewed.loc[row_mask, "review_flag"] = False
            reviewed.loc[row_mask, "review_reasons"] = "sme_corrected"
            reviewed.loc[row_mask, "sme_override_applied"] = True
            if notes:
                reviewed.loc[row_mask, "rationale"] = (
                    reviewed.loc[row_mask, "rationale"].astype(str)
                    + " SME correction: "
                    + notes
                )
        elif decision == "needs_discussion":
            reviewed.loc[row_mask, "review_flag"] = True
            reviewed.loc[row_mask, "review_reasons"] = reviewed.loc[
                row_mask,
                "review_reasons",
            ].map(lambda value: append_reason(value, "sme_needs_discussion"))
        elif decision == "out_of_scope":
            reviewed.loc[row_mask, "review_flag"] = True
            reviewed.loc[row_mask, "review_reasons"] = reviewed.loc[
                row_mask,
                "review_reasons",
            ].map(lambda value: append_reason(value, "sme_out_of_scope"))

    return reviewed


def build_validation_outputs(
    classifications_df: pd.DataFrame,
    claims: Sequence[OfficeClaim] | pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build validation and summary tables from reviewed classifications."""

    claims_df = (
        claims.copy()
        if isinstance(claims, pd.DataFrame)
        else office_claims_to_dataframe(claims)
    )
    evidence = classifications_df.copy()
    evidence["review_flag"] = evidence["review_flag"].map(_truthy)
    evidence["validation_funding_level"] = evidence["validation_funding_level"].map(_truthy)
    evidence["five_year_total"] = pd.to_numeric(
        evidence["five_year_total"],
        errors="coerce",
    ).fillna(0.0)
    for column in ["source_sub_office", "scope_description"]:
        if column not in evidence.columns:
            evidence[column] = ""

    positive = evidence.loc[
        evidence["validation_funding_level"] & (evidence["five_year_total"] > 0)
    ].copy()

    office_claim_validation = _build_office_claim_validation(claims_df, positive)
    office_challenge_evidence = _build_office_challenge_evidence(positive, claims_df)
    challenge_coverage_summary = _build_challenge_coverage_summary(
        office_claim_validation,
        office_challenge_evidence,
        claims_df,
    )
    wbs_challenge_summary = _build_wbs_challenge_summary(positive)

    return {
        "office_claim_validation": office_claim_validation,
        "challenge_coverage_summary": challenge_coverage_summary,
        "office_challenge_evidence": office_challenge_evidence,
        "wbs_challenge_summary": wbs_challenge_summary,
    }


def export_validation_outputs(
    outputs: Mapping[str, pd.DataFrame],
    paths_by_name: Mapping[str, str | Path],
) -> dict[str, Path]:
    """Export validation tables to their configured CSV paths."""

    exported = {}
    for name, frame in outputs.items():
        path = Path(paths_by_name[name])
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        exported[name] = path
    return exported


def run_classification_pipeline(
    *,
    classifier: str = "lexical",
    model: str | None = None,
    top_k: int = 5,
    reasoning_effort: str = "medium",
    limit: int | None = None,
    resume_cache: bool = False,
    audit_cache_path: str | Path | None = None,
    openai_client: Any | None = None,
    asksage_client: Any | None = None,
    asksage_base_url: str | None = None,
    asksage_openai_url: str | None = None,
    write_outputs: bool = True,
) -> dict[str, Any]:
    """Build the graph, classify row evidence, and produce validation outputs."""

    from . import paths
    from .build_ontology import build_combined_ontology, export_graph

    paths.ensure_output_dirs()
    graph = build_combined_ontology()
    challenges = build_challenge_profiles(graph)
    offices = build_office_mappings(graph)
    claims = build_office_claims(graph, challenges)
    evidence_rows = build_programming_evidence_profiles(
        graph,
        claimed_offices={claim.claimed_office for claim in claims},
    )
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be greater than zero when provided")
        evidence_rows = evidence_rows[:limit]

    llm_classifier: LLMClassifier | None = None
    resolved_audit_cache_path = audit_cache_path
    if classifier == "openai":
        from .openai_classifier import OpenAIChallengeClassifier

        if resolved_audit_cache_path is None:
            resolved_audit_cache_path = paths.OPENAI_CLASSIFICATION_AUDIT_PATH
        llm_classifier = OpenAIChallengeClassifier(
            model=model or DEFAULT_OPENAI_MODEL,
            reasoning_effort=reasoning_effort,
            client=openai_client,
            audit_cache_path=resolved_audit_cache_path,
            resume_cache=resume_cache,
        )
    elif classifier == "asksage":
        from .asksage_classifier import AskSageChallengeClassifier

        if resolved_audit_cache_path is None:
            resolved_audit_cache_path = paths.ASKSAGE_CLASSIFICATION_AUDIT_PATH
        llm_classifier = AskSageChallengeClassifier(
            model=model,
            reasoning_effort=reasoning_effort,
            base_url=asksage_base_url,
            openai_url=asksage_openai_url,
            client=asksage_client,
            audit_cache_path=resolved_audit_cache_path,
            resume_cache=resume_cache,
        )
    elif classifier != "lexical":
        raise ValueError("classifier must be 'lexical', 'openai', or 'asksage'")

    results = classify_programming_records(
        evidence_rows,
        challenges,
        offices=offices,
        llm_classifier=llm_classifier,
        top_k=top_k,
    )
    classification_df = classification_results_to_dataframe(results)

    review_df = build_review_template(classification_df)
    if write_outputs:
        export_classifications(results, paths.PROGRAMMING_CHALLENGE_CLASSIFICATIONS_PATH)
        classification_df.to_csv(paths.CLASSIFICATION_DF_PATH, index=False)
        review_df, _ = load_or_initialize_sme_review(
            paths.SME_REVIEW_OVERRIDES_PATH,
            classification_df,
        )

    reviewed_classifications = apply_sme_overrides(classification_df, review_df, challenges)
    validation_outputs = build_validation_outputs(reviewed_classifications, claims)
    classified_graph = enrich_graph_with_classifications(graph, results)

    if write_outputs:
        reviewed_classifications.to_csv(
            paths.REVIEWED_PROGRAMMING_CHALLENGE_CLASSIFICATIONS_PATH,
            index=False,
        )
        export_validation_outputs(
            validation_outputs,
            {
                "office_claim_validation": paths.OFFICE_CLAIM_VALIDATION_PATH,
                "challenge_coverage_summary": paths.CHALLENGE_COVERAGE_SUMMARY_PATH,
                "office_challenge_evidence": paths.OFFICE_CHALLENGE_EVIDENCE_PATH,
                "wbs_challenge_summary": paths.WBS_CHALLENGE_SUMMARY_PATH,
            },
        )
        validation_outputs["challenge_coverage_summary"].to_csv(
            paths.REVIEW_SUMMARY_PATH,
            index=False,
        )
        export_graph(graph, paths.COMBINED_ONTOLOGY_GRAPH_PATH)
        export_graph(classified_graph, paths.CLASSIFIED_ONTOLOGY_GRAPH_PATH)

    return {
        "graph": graph,
        "classified_graph": classified_graph,
        "challenges": challenges,
        "claims": claims,
        "evidence_rows": evidence_rows,
        "classification_results": results,
        "classifications": classification_df,
        "reviewed_classifications": reviewed_classifications,
        "validation_outputs": validation_outputs,
        "audit_cache_path": resolved_audit_cache_path,
    }


def append_reason(existing: Any, reason: str) -> str:
    """Append a semicolon-delimited review reason if not already present."""

    parts = []
    if not pd.isna(existing) and str(existing).strip() and str(existing).lower() != "nan":
        parts.extend(part.strip() for part in str(existing).split(";") if part.strip())
    if reason not in parts:
        parts.append(reason)
    return "; ".join(parts)


def _eligible_challenge_profiles(
    challenges: Mapping[str, ChallengeProfile],
    offices: Mapping[str, OfficeMapping] | None,
) -> dict[str, ChallengeProfile]:
    if offices:
        eligible = {name: challenges[name] for name in offices if name in challenges}
        if eligible:
            return eligible
    if challenges:
        return dict(challenges)
    raise ValueError("No Genesis Mission Challenge candidates were found.")


def _build_result_from_decision(
    evidence: EvidenceProfile,
    candidates: list[CandidateChallenge],
    decision: dict[str, Any],
    *,
    challenges: Mapping[str, ChallengeProfile],
) -> ClassificationResult:
    top_candidate = candidates[0]
    candidate_names = {candidate.challenge_name for candidate in candidates}
    selected_name = str(decision.get("predicted_challenge", "")).strip()
    review_reasons = []

    if selected_name not in candidate_names:
        selected_name = top_candidate.challenge_name
        review_reasons.append("classifier_selection_not_in_candidate_list")

    confidence = _normalize_confidence(decision.get("confidence", "low"))
    if confidence == "low":
        review_reasons.append("low_confidence")

    if _is_ambiguous(candidates):
        review_reasons.append("ambiguous_top_candidates")

    selected_challenge = challenges[selected_name]
    has_mstc_context = _has_mstc_context(selected_challenge)
    if not has_mstc_context:
        review_reasons.append("title_only_challenge_context")

    rationale = str(decision.get("rationale", "")).strip()
    if not rationale:
        rationale = _fallback_rationale(evidence, top_candidate)

    return ClassificationResult(
        programming_record_id=evidence.programming_record_id,
        source_file=evidence.source_file,
        source_row_number=evidence.source_row_number,
        wbs_name=evidence.wbs_name,
        wbs_node_id=evidence.wbs_node_id,
        wbs_code=evidence.wbs_code,
        source_sub_office=evidence.source_sub_office,
        normalized_funding_office=evidence.normalized_funding_office,
        stat_l3=evidence.stat_l3,
        stat_l4=evidence.stat_l4,
        stat_l5=evidence.stat_l5,
        funding_level=evidence.funding_level,
        scope_description=evidence.scope_description,
        program_interest_area=evidence.program_interest_area,
        process_improvement_area=evidence.process_improvement_area,
        program_request=evidence.program_request,
        acquisition_context=evidence.acquisition_context,
        fy2028=evidence.fy2028,
        fy2029=evidence.fy2029,
        fy2030=evidence.fy2030,
        fy2031=evidence.fy2031,
        fy2032=evidence.fy2032,
        five_year_total=evidence.five_year_total,
        validation_funding_level=evidence.validation_funding_level,
        predicted_challenge=selected_name,
        challenge_node_id=selected_challenge.challenge_node_id,
        challenge_has_mstc_context=has_mstc_context,
        confidence=confidence,
        review_flag=bool(review_reasons),
        rationale=rationale,
        top_candidate_challenges=tuple(
            f"{candidate.challenge_name} ({candidate.score:.2f})"
            for candidate in candidates
        ),
        additional_programming_context=evidence.additional_context,
        evidence_node_ids=evidence.evidence_node_ids,
        review_reasons=tuple(dict.fromkeys(review_reasons)),
    )


def _classification_payload(
    evidence: EvidenceProfile,
    candidates: list[CandidateChallenge],
    challenges: Mapping[str, ChallengeProfile],
) -> dict[str, Any]:
    return {
        "instructions": (
            "Choose exactly one predicted_challenge from candidate_challenges. "
            "Use Scope Description as the primary evidence for what Genesis "
            "Mission Challenge the programming row supports. Use WBS, STAT, "
            "program request, funding office, and program context only as "
            "supporting evidence. Do not assign or change program offices. "
            "Return predicted_challenge, confidence, and rationale. Confidence "
            "must be one of: high, medium, low."
        ),
        "programming_evidence": asdict(evidence),
        "candidate_challenges": [
            {
                **asdict(candidate),
                "context": asdict(challenges[candidate.challenge_name]),
            }
            for candidate in candidates
        ],
    }


def _lexical_classifier(payload: dict[str, Any]) -> dict[str, Any]:
    candidates = payload["candidate_challenges"]
    if not candidates:
        raise ValueError("The lexical classifier received no candidate challenges.")

    top = candidates[0]
    second_score = candidates[1]["score"] if len(candidates) > 1 else 0.0
    score_gap = top["score"] - second_score

    if top["score"] >= 0.11 and score_gap >= 0.02:
        confidence = "high"
    elif top["score"] >= 0.06 and score_gap >= 0.01:
        confidence = "medium"
    else:
        confidence = "low"

    matched_terms = ", ".join(top.get("matched_terms", [])[:8])
    rationale = (
        f"Selected by deterministic lexical retrieval from row-level scope evidence. "
        f"Top score {top['score']:.2f}; matched terms: {matched_terms or 'none'}."
    )
    return {
        "predicted_challenge": top["challenge_name"],
        "confidence": confidence,
        "rationale": rationale,
    }


def _parse_classifier_response(response: Mapping[str, Any] | str) -> dict[str, Any]:
    if isinstance(response, str):
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            return {"predicted_challenge": response, "confidence": "low", "rationale": ""}
        if isinstance(parsed, dict):
            return parsed
        return {"predicted_challenge": str(parsed), "confidence": "low", "rationale": ""}

    if response is None:
        return {"predicted_challenge": "", "confidence": "low", "rationale": ""}

    if not isinstance(response, Mapping):
        raise TypeError(
            "llm_classifier must return a mapping or JSON string with "
            "predicted_challenge, confidence, and rationale"
        )

    return dict(response)


def _candidate_score(
    evidence: EvidenceProfile,
    challenge: ChallengeProfile,
    *,
    evidence_tokens: set[str],
    challenge_tokens: set[str],
    matched_terms: set[str],
) -> float:
    if not evidence_tokens or not challenge_tokens:
        return 0.0

    weighted_overlap = sum(_term_weight(term) for term in matched_terms)
    weighted_evidence = sum(_term_weight(term) for term in evidence_tokens)
    weighted_challenge = sum(_term_weight(term) for term in challenge_tokens)
    denominator = max(weighted_evidence + weighted_challenge - weighted_overlap, 1.0)
    score = weighted_overlap / denominator

    scope_tokens = _tokens(evidence.scope_description)
    title_tokens = _tokens(challenge.challenge_name)
    if title_tokens:
        score += 0.18 * len(scope_tokens & title_tokens) / len(title_tokens)
    if challenge.challenge_name.casefold() in evidence.scope_description.casefold():
        score += 0.3
    if evidence.wbs_name and evidence.wbs_name.casefold() in challenge.challenge_name.casefold():
        score += 0.08

    return min(score, 1.0)


def _term_weight(term: str) -> float:
    if term in {"ai", "ml"}:
        return 0.4
    if term in {"nuclear", "nnsa", "mission"}:
        return 0.6
    if len(term) >= 10:
        return 1.4
    return 1.0


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", str(text).casefold())
        if len(token) >= 3 and token not in STOPWORDS
    }


def _evidence_text(evidence: EvidenceProfile) -> str:
    # Scope is repeated intentionally because it is the authoritative mapping evidence.
    parts = [
        evidence.scope_description,
        evidence.scope_description,
        evidence.wbs_name,
        evidence.stat_l3,
        evidence.stat_l4,
        evidence.stat_l5,
        evidence.program_interest_area,
        evidence.process_improvement_area,
        evidence.program_request,
        *evidence.acquisition_context,
        *evidence.additional_context,
    ]
    return " ".join(part for part in parts if part)


def _challenge_text(challenge: ChallengeProfile) -> str:
    parts = [
        challenge.challenge_name,
        *challenge.problem_statements,
        *challenge.proposed_solutions,
        *challenge.nnsa_advantages,
        *challenge.expected_outcomes,
    ]
    return " ".join(part for part in parts if part)


def _has_mstc_context(challenge: ChallengeProfile) -> bool:
    return any(
        [
            challenge.problem_statements,
            challenge.proposed_solutions,
            challenge.nnsa_advantages,
            challenge.expected_outcomes,
        ]
    )


def _is_ambiguous(candidates: list[CandidateChallenge]) -> bool:
    if len(candidates) < 2:
        return False
    return candidates[0].score - candidates[1].score < 0.01


def _fallback_rationale(
    evidence: EvidenceProfile,
    candidate: CandidateChallenge,
) -> str:
    matched_terms = ", ".join(candidate.matched_terms[:8])
    return (
        f"{evidence.source_file} row {evidence.source_row_number} was matched to "
        f"{candidate.challenge_name} with lexical score {candidate.score:.2f}; "
        f"matched terms: {matched_terms or 'none'}."
    )


def _normalize_confidence(value: Any) -> str:
    confidence = str(value).strip().casefold()
    if confidence in CONFIDENCE_LEVELS:
        return confidence
    return "low"


def _migrate_wbs_review(
    existing: pd.DataFrame,
    template: pd.DataFrame,
    editable_columns: list[str],
) -> pd.DataFrame:
    review = template.copy()
    existing_indexed = existing.drop_duplicates("wbs_name", keep="last").set_index("wbs_name")
    for index, row in review.iterrows():
        wbs_name = row["wbs_name"]
        if wbs_name not in existing_indexed.index:
            continue
        existing_row = existing_indexed.loc[wbs_name]
        for column in editable_columns:
            if column in existing_row.index:
                review.loc[index, column] = existing_row[column]
    return review


def _build_office_claim_validation(
    claims_df: pd.DataFrame,
    positive_evidence: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, claim in claims_df.iterrows():
        matching = positive_evidence.loc[
            (positive_evidence["predicted_challenge"] == claim["challenge_name"])
            & (positive_evidence["normalized_funding_office"] == claim["claimed_office"])
        ].copy()
        evidence_count = len(matching)
        non_low_ready = matching.loc[
            (matching["confidence"].astype(str).str.lower() != "low")
            & (~matching["review_flag"].map(_truthy))
        ]
        if non_low_ready.empty and evidence_count:
            status = "needs_review"
        elif evidence_count:
            status = "validated"
        else:
            status = "unfunded"

        rows.append(
            {
                "challenge_name": claim["challenge_name"],
                "claimed_office": claim["claimed_office"],
                "claim_role": claim["claim_role"],
                "challenge_has_mstc_context": _truthy(claim["challenge_has_mstc_context"]),
                "validation_status": status,
                "evidence_row_count": evidence_count,
                "evidence_wbs_count": matching["wbs_name"].nunique() if evidence_count else 0,
                "five_year_total": matching["five_year_total"].sum() if evidence_count else 0.0,
                "high_confidence_count": int((matching["confidence"] == "high").sum()),
                "medium_confidence_count": int((matching["confidence"] == "medium").sum()),
                "low_confidence_count": int((matching["confidence"] == "low").sum()),
                "review_flag_count": int(matching["review_flag"].map(_truthy).sum()),
                "evidence_wbs_names": "; ".join(
                    sorted(matching["wbs_name"].dropna().astype(str).unique(), key=str.casefold)
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["validation_status", "challenge_name", "claimed_office"],
        kind="stable",
    )


def _build_office_challenge_evidence(
    positive_evidence: pd.DataFrame,
    claims_df: pd.DataFrame,
) -> pd.DataFrame:
    if positive_evidence.empty:
        columns = [
            "challenge_name",
            "normalized_funding_office",
            "claim_role",
            "is_claimed",
            "evidence_row_count",
            "evidence_wbs_count",
            "five_year_total",
            "review_flag_count",
        ]
        return pd.DataFrame(columns=columns)

    grouped = (
        positive_evidence.groupby(["predicted_challenge", "normalized_funding_office"], dropna=False)
        .agg(
            evidence_row_count=("programming_record_id", "count"),
            evidence_wbs_count=("wbs_name", "nunique"),
            five_year_total=("five_year_total", "sum"),
            review_flag_count=("review_flag", lambda values: sum(_truthy(value) for value in values)),
            evidence_wbs_names=(
                "wbs_name",
                lambda values: "; ".join(
                    sorted(set(str(value) for value in values if str(value).strip()), key=str.casefold)
                ),
            ),
        )
        .reset_index()
        .rename(
            columns={
                "predicted_challenge": "challenge_name",
                "normalized_funding_office": "normalized_funding_office",
            }
        )
    )
    claim_roles = (
        claims_df.groupby(["challenge_name", "claimed_office"])["claim_role"]
        .apply(lambda values: "; ".join(sorted(set(values))))
        .reset_index()
        .rename(columns={"claimed_office": "normalized_funding_office"})
    )
    grouped = grouped.merge(
        claim_roles,
        on=["challenge_name", "normalized_funding_office"],
        how="left",
    )
    grouped["is_claimed"] = grouped["claim_role"].fillna("") != ""
    grouped["claim_role"] = grouped["claim_role"].fillna("unclaimed")
    return grouped.sort_values(
        ["challenge_name", "is_claimed", "normalized_funding_office"],
        ascending=[True, False, True],
    )


def _build_challenge_coverage_summary(
    office_claim_validation: pd.DataFrame,
    office_challenge_evidence: pd.DataFrame,
    claims_df: pd.DataFrame,
) -> pd.DataFrame:
    status_counts = office_claim_validation.pivot_table(
        index="challenge_name",
        columns="validation_status",
        values="claimed_office",
        aggfunc="count",
        fill_value=0,
    )
    for column in ["validated", "needs_review", "unfunded"]:
        if column not in status_counts.columns:
            status_counts[column] = 0
    status_counts = status_counts[["validated", "needs_review", "unfunded"]]

    claim_counts = (
        claims_df.groupby("challenge_name")
        .agg(
            claim_count=("claimed_office", "count"),
            claimed_offices=(
                "claimed_office",
                lambda values: "; ".join(sorted(set(values))),
            ),
            challenge_has_mstc_context=("challenge_has_mstc_context", "max"),
        )
    )

    evidence_totals = (
        office_challenge_evidence.groupby("challenge_name")
        .agg(
            evidence_row_count=("evidence_row_count", "sum"),
            evidence_wbs_count=("evidence_wbs_count", "sum"),
            five_year_total=("five_year_total", "sum"),
            actual_funding_offices=(
                "normalized_funding_office",
                lambda values: "; ".join(sorted(set(values))),
            ),
        )
        if not office_challenge_evidence.empty
        else pd.DataFrame()
    )

    summary = claim_counts.join(status_counts, how="left").join(evidence_totals, how="left")
    summary = summary.fillna(
        {
            "validated": 0,
            "needs_review": 0,
            "unfunded": 0,
            "evidence_row_count": 0,
            "evidence_wbs_count": 0,
            "five_year_total": 0.0,
            "actual_funding_offices": "",
        }
    )
    for column in ["claim_count", "validated", "needs_review", "unfunded", "evidence_row_count", "evidence_wbs_count"]:
        summary[column] = summary[column].astype(int)
    return summary.reset_index().sort_values("challenge_name")


def _build_wbs_challenge_summary(positive_evidence: pd.DataFrame) -> pd.DataFrame:
    if positive_evidence.empty:
        return pd.DataFrame()

    return (
        positive_evidence.groupby(
            ["wbs_name", "predicted_challenge", "normalized_funding_office"],
            dropna=False,
        )
        .agg(
            evidence_row_count=("programming_record_id", "count"),
            funding_levels=("funding_level", lambda values: "; ".join(sorted(set(values)))),
            source_sub_offices=(
                "source_sub_office",
                lambda values: "; ".join(sorted(set(values))),
            ),
            five_year_total=("five_year_total", "sum"),
            confidence=("confidence", lambda values: "; ".join(sorted(set(values)))),
            review_flag_count=("review_flag", lambda values: sum(_truthy(value) for value in values)),
            scope_evidence=(
                "scope_description",
                lambda values: " | ".join(dict.fromkeys(str(value)[:180] for value in values if str(value).strip())),
            ),
        )
        .reset_index()
        .sort_values(["predicted_challenge", "normalized_funding_office", "wbs_name"])
    )


def _collect_additional_programming_context(
    graph: nx.MultiDiGraph,
    record_ids: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    context = []
    nodes = []

    for record_id in record_ids:
        for _, target, edge_data in graph.out_edges(record_id, data=True):
            relation = edge_data.get("relation", "")
            if not relation.startswith("has_programming_"):
                continue

            node_data = graph.nodes[target]
            column_name = node_data.get("column_name") or relation.removeprefix(
                "has_programming_"
            )
            is_classification_context = (
                str(node_data.get("classification_context", "")).casefold() == "true"
                or column_name in CLASSIFICATION_CONTEXT_COLUMNS
            )
            if not is_classification_context:
                continue

            label = node_data.get("label", target)
            context.append(f"{column_name}: {label}")
            nodes.append(target)

    return (
        tuple(sorted(dict.fromkeys(context), key=str.casefold)),
        tuple(dict.fromkeys(nodes)),
    )


def _collect_single_programming_column_value(
    graph: nx.MultiDiGraph,
    record_id: str,
    column_name: str,
) -> str:
    values = []
    for _, node_data in _iter_programming_attribute_nodes(graph, [record_id]):
        if node_data.get("column_name") == column_name:
            label = node_data.get("label", "")
            if label:
                values.append(str(label))
    return "; ".join(dict.fromkeys(values))


def _collect_programming_column_context(
    graph: nx.MultiDiGraph,
    record_ids: Sequence[str],
    column_names: set[str],
) -> tuple[str, ...]:
    context = []
    for _, node_data in _iter_programming_attribute_nodes(graph, record_ids):
        column_name = node_data.get("column_name")
        if column_name in column_names:
            label = node_data.get("label", "")
            if label:
                context.append(f"{column_name}: {label}")
    return tuple(sorted(dict.fromkeys(context), key=str.casefold))


def _iter_programming_attribute_nodes(
    graph: nx.MultiDiGraph,
    record_ids: Sequence[str],
) -> tuple[tuple[str, dict[str, Any]], ...]:
    nodes = []
    for record_id in record_ids:
        for _, target, edge_data in graph.out_edges(record_id, data=True):
            relation = edge_data.get("relation", "")
            if relation.startswith("has_programming_"):
                nodes.append((target, graph.nodes[target]))
    return tuple(nodes)


def _record_target_label(graph: nx.MultiDiGraph, record_id: str, relation: str) -> str:
    node_id = _record_target_node(graph, record_id, relation)
    return str(graph.nodes[node_id].get("label", "")) if node_id else ""


def _record_target_node(graph: nx.MultiDiGraph, record_id: str, relation: str) -> str:
    targets = _target_nodes(graph, record_id, relation)
    return targets[0] if targets else ""


def _target_nodes(
    graph: nx.MultiDiGraph,
    source: str,
    relation: str,
) -> list[str]:
    targets = []
    for _, target, edge_data in graph.out_edges(source, data=True):
        if edge_data.get("relation") == relation:
            targets.append(target)
    return sorted(dict.fromkeys(targets), key=lambda node: graph.nodes[node].get("label", node))


def _labels_for_nodes(
    graph: nx.MultiDiGraph,
    node_ids: Sequence[str],
) -> tuple[str, ...]:
    labels = [str(graph.nodes[node_id].get("label", node_id)) for node_id in node_ids]
    return tuple(sorted(dict.fromkeys(labels), key=str.casefold))


def _to_float(value: Any) -> float | None:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _first_delimited(value: Any, delimiter: str = "; ") -> str:
    return str(value or "").split(delimiter)[0]


def _truthy(value: Any) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes", "y"}


def _bool_text(value: Any) -> str:
    return "true" if _truthy(value) else "false"


def _add_classification_edge(
    graph: nx.MultiDiGraph,
    source: str,
    target: str,
    relation: str,
    result: ClassificationResult,
) -> None:
    graph.add_edge(
        source,
        target,
        key=f"{relation}:{_slugify(result.programming_record_id)}",
        relation=relation,
        weight=1,
        source_files=CLASSIFICATION_SOURCE,
        source_rows=str(result.source_row_number),
        confidence=result.confidence,
        review_flag=str(result.review_flag),
        rationale=result.rationale,
        funding_level=result.funding_level,
        five_year_total=result.five_year_total,
        normalized_funding_office=result.normalized_funding_office,
    )


def _ensure_node(
    graph: nx.MultiDiGraph,
    node_id: str,
    label: str,
    node_type: str,
) -> str:
    if node_id and node_id in graph:
        return node_id

    existing = _find_node(graph, node_type, label)
    if existing:
        return existing

    new_node_id = _node_id(node_type, label)
    graph.add_node(
        new_node_id,
        label=label,
        node_type=node_type,
        source_files=CLASSIFICATION_SOURCE,
    )
    return new_node_id


def _find_node(graph: nx.MultiDiGraph, node_type: str, label: str) -> str:
    for node_id, data in graph.nodes(data=True):
        if data.get("node_type") == node_type and data.get("label") == label:
            return node_id
    return ""


def _node_id(node_type: str, label: str) -> str:
    digest = sha1(label.casefold().encode("utf-8")).hexdigest()[:10]
    slug = _slugify(label)[:48].strip("_")
    return f"{node_type}:{slug or digest}:{digest}"


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().casefold()).strip("_")


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for row-level classification exports."""

    parser = argparse.ArgumentParser(
        description="Classify programming rows against Genesis Mission Challenges.",
    )
    parser.add_argument(
        "--classifier",
        choices=["lexical", "openai", "asksage"],
        default="lexical",
        help=(
            "Classifier backend. OpenAI requires OPENAI_API_KEY; Ask Sage "
            "uses credentials.json by default."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "LLM model. OpenAI defaults to "
            f"{DEFAULT_OPENAI_MODEL}; Ask Sage uses this value, ASKSAGE_MODEL, "
            f"credentials.json, or {DEFAULT_ASKSAGE_MODEL}."
        ),
    )
    parser.add_argument("--top-k", type=int, default=5, help="Candidate count per row.")
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "xhigh"],
        default="medium",
        help="Reasoning effort for LLM-backed classifiers.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Classify only the first N evidence rows.",
    )
    parser.add_argument(
        "--resume-cache",
        action="store_true",
        help="Reuse matching decisions from the provider audit JSONL cache.",
    )
    parser.add_argument(
        "--audit-cache",
        type=Path,
        default=None,
        help="Path for provider classification audit/cache JSONL.",
    )
    parser.add_argument(
        "--asksage-base-url",
        default=None,
        help=(
            "Optional Ask Sage server API base URL override. Defaults to "
            "credentials.json or the built-in Ask Sage server URL."
        ),
    )
    parser.add_argument(
        "--asksage-openai-url",
        default=None,
        help=(
            "Optional Ask Sage OpenAI-compatible base URL override. Defaults "
            "to credentials.json openai_url or derives from the server URL."
        ),
    )
    args = parser.parse_args(argv)

    outputs = run_classification_pipeline(
        classifier=args.classifier,
        model=args.model,
        top_k=args.top_k,
        reasoning_effort=args.reasoning_effort,
        limit=args.limit,
        resume_cache=args.resume_cache,
        audit_cache_path=args.audit_cache,
        asksage_base_url=args.asksage_base_url,
        asksage_openai_url=args.asksage_openai_url,
    )
    classifications = outputs["classifications"]
    validation = outputs["validation_outputs"]["office_claim_validation"]
    print(
        f"Classified {len(classifications)} programming rows with "
        f"{args.classifier} classifier."
    )
    print(f"Office claims validated: {len(validation)}")
    if outputs["audit_cache_path"]:
        print(f"LLM audit cache: {outputs['audit_cache_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
