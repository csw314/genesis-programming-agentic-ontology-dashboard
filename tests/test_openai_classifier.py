from __future__ import annotations

import json

import pytest

from programming_ontology.build_ontology import build_combined_ontology
from programming_ontology.classify_programming import classify_graph_with_openai
from programming_ontology.openai_classifier import (
    DEFAULT_OPENAI_MODEL,
    OpenAIChallengeClassifier,
)


class FakeResponses:
    def __init__(self, challenge_name: str | None = None, *, raises: bool = False) -> None:
        self.challenge_name = challenge_name
        self.raises = raises
        self.calls: list[dict] = []

    def create(self, **kwargs):
        if self.raises:
            raise AssertionError("unexpected OpenAI API call")
        self.calls.append(kwargs)
        enum = kwargs["text"]["format"]["schema"]["properties"]["predicted_challenge"]["enum"]
        challenge_name = self.challenge_name or enum[0]
        return {
            "id": "resp_test_123",
            "output_text": json.dumps(
                {
                    "predicted_challenge": challenge_name,
                    "confidence": "medium",
                    "rationale": "The scope evidence aligns with the selected challenge.",
                }
            ),
        }


class FakeClient:
    def __init__(self, responses: FakeResponses) -> None:
        self.responses = responses


def test_openai_classifier_uses_responses_api_structured_outputs(tmp_path) -> None:
    responses = FakeResponses()
    classifier = OpenAIChallengeClassifier(
        client=FakeClient(responses),
        audit_cache_path=tmp_path / "audit.jsonl",
    )
    payload = {
        "instructions": "choose a challenge",
        "programming_evidence": {
            "programming_record_id": "programming_record:test",
            "source_row_number": 1,
            "wbs_name": "Test WBS",
        },
        "candidate_challenges": [
            {
                "challenge_name": "Challenge A",
                "score": 0.9,
                "matched_terms": ["test"],
                "has_mstc_context": True,
                "context": {"challenge_name": "Challenge A"},
            }
        ],
    }

    decision = classifier(payload)
    call = responses.calls[0]

    assert decision["predicted_challenge"] == "Challenge A"
    assert call["model"] == DEFAULT_OPENAI_MODEL
    assert call["store"] is False
    assert call["text"]["format"]["type"] == "json_schema"
    assert call["text"]["format"]["strict"] is True
    assert call["reasoning"]["effort"] == "medium"
    assert (tmp_path / "audit.jsonl").exists()


def test_openai_classifier_resume_cache_avoids_api_call(tmp_path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    payload = {
        "instructions": "choose a challenge",
        "programming_evidence": {
            "programming_record_id": "programming_record:test",
            "source_row_number": 1,
            "wbs_name": "Test WBS",
        },
        "candidate_challenges": [
            {
                "challenge_name": "Challenge A",
                "score": 0.9,
                "matched_terms": [],
                "has_mstc_context": True,
                "context": {"challenge_name": "Challenge A"},
            }
        ],
    }
    OpenAIChallengeClassifier(
        client=FakeClient(FakeResponses()),
        audit_cache_path=audit_path,
    )(payload)

    cached = OpenAIChallengeClassifier(
        client=FakeClient(FakeResponses(raises=True)),
        audit_cache_path=audit_path,
        resume_cache=True,
    )

    assert cached(payload)["predicted_challenge"] == "Challenge A"


def test_openai_classifier_requires_api_key_without_injected_client(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIChallengeClassifier()


def test_graph_openai_entry_point_preserves_row_level_outputs(tmp_path) -> None:
    graph = build_combined_ontology()
    results = classify_graph_with_openai(
        graph,
        limit=1,
        openai_client=FakeClient(FakeResponses()),
        audit_cache_path=tmp_path / "audit.jsonl",
    )

    assert len(results) == 1
    assert results[0].programming_record_id.startswith("programming_record:")
    assert results[0].source_sub_office == "NA11"
    assert results[0].normalized_funding_office == "NA11"
