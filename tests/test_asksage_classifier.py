from __future__ import annotations

import json

import pytest

from programming_ontology.asksage_classifier import (
    DEFAULT_ASKSAGE_MODEL,
    AskSageAIDirectnessClassifier,
    AskSageChallengeClassifier,
    build_asksage_client,
    build_asksage_openai_client,
    load_credentials,
    normalize_asksage_openai_base_url,
    parse_asksage_chat_completion_decision,
    parse_asksage_credentials,
    resolve_asksage_openai_base_url,
)
from programming_ontology.build_ontology import build_combined_ontology
from programming_ontology.classify_programming import classify_graph_with_asksage


ASKSAGE_MODEL = "test-asksage-model"
ASKSAGE_USER_BASE_URL = "https://asksage.example.test/user"
ASKSAGE_SERVER_BASE_URL = "https://asksage.example.test/server"
ASKSAGE_OPENAI_URL = "https://asksage-openai.example.test/v1"


class FakeCompletions:
    def __init__(self, challenge_name: str | None = None, *, raises: bool = False) -> None:
        self.challenge_name = challenge_name
        self.raises = raises
        self.calls: list[dict] = []

    def create(self, **kwargs):
        if self.raises:
            raise AssertionError("unexpected Ask Sage API call")
        self.calls.append(kwargs)
        properties = kwargs["response_format"]["json_schema"]["schema"]["properties"]
        if "ai_directness_score" in properties:
            return {
                "id": "asksage-ai-directness-test",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "ai_directness_score": 87,
                                    "ai_directness_label": "ai_enabled_workflow",
                                    "confidence": "high",
                                    "rationale": (
                                        "The row describes an AI-enabled workflow."
                                    ),
                                }
                            )
                        }
                    }
                ],
            }
        enum = properties["predicted_challenge"]["enum"]
        challenge_name = self.challenge_name or enum[0]
        return {
            "id": "asksage-chatcmpl-test",
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "predicted_challenge": challenge_name,
                                "confidence": "medium",
                                "rationale": (
                                    "The scope evidence aligns with the selected challenge."
                                ),
                            }
                        )
                    }
                }
            ],
        }


class FakeChat:
    def __init__(self, completions: FakeCompletions) -> None:
        self.completions = completions


class FakeOpenAIClient:
    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = FakeChat(completions)


class FakeAskSageClient:
    calls: list[dict] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.calls.append(kwargs)


def _credentials_payload(*, model: str | None = ASKSAGE_MODEL) -> dict:
    credentials = {
        "api_key": "test-token",
        "Ask_sage_user_info": {"username": "user@example.test"},
        "user_base_url": ASKSAGE_USER_BASE_URL,
        "server_base_url": ASKSAGE_SERVER_BASE_URL,
        "openai_url": ASKSAGE_OPENAI_URL,
    }
    if model is not None:
        credentials["model"] = model
    return {"credentials": credentials}


def _payload() -> dict:
    return {
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


def test_load_credentials_parses_project_json_shape(tmp_path) -> None:
    credentials_path = tmp_path / "credentials.json"
    credentials_path.write_text(json.dumps(_credentials_payload()), encoding="utf-8")

    loaded = load_credentials(credentials_path)
    credentials = parse_asksage_credentials(loaded, source=str(credentials_path))

    assert credentials.api_key == "test-token"
    assert credentials.email == "user@example.test"
    assert credentials.user_base_url == ASKSAGE_USER_BASE_URL
    assert credentials.server_base_url == ASKSAGE_SERVER_BASE_URL
    assert credentials.openai_url == ASKSAGE_OPENAI_URL
    assert credentials.model == ASKSAGE_MODEL


def test_build_asksage_client_uses_credentials_file(tmp_path) -> None:
    FakeAskSageClient.calls.clear()
    credentials_path = tmp_path / "credentials.json"
    credentials_path.write_text(json.dumps(_credentials_payload()), encoding="utf-8")

    client = build_asksage_client(
        credentials_path=credentials_path,
        client_class=FakeAskSageClient,
    )

    assert isinstance(client, FakeAskSageClient)
    assert FakeAskSageClient.calls == [
        {
            "email": "user@example.test",
            "api_key": "test-token",
            "user_base_url": ASKSAGE_USER_BASE_URL,
            "server_base_url": ASKSAGE_SERVER_BASE_URL,
            "path_to_CA_Bundle": None,
        }
    ]


def test_build_asksage_openai_client_uses_openai_base_url_and_header(tmp_path) -> None:
    credentials_path = tmp_path / "credentials.json"
    credentials_path.write_text(json.dumps(_credentials_payload()), encoding="utf-8")

    client = build_asksage_openai_client(
        credentials_path=credentials_path,
        use_truststore=False,
    )

    assert str(client.base_url) == f"{ASKSAGE_OPENAI_URL}/"
    assert client.default_headers["x-access-tokens"] == "test-token"


def test_build_asksage_openai_client_derives_url_from_server_when_needed(tmp_path) -> None:
    credentials_path = tmp_path / "credentials.json"
    credentials_path.write_text(
        json.dumps(_credentials_payload(model=ASKSAGE_MODEL) | {
            "credentials": {
                key: value
                for key, value in _credentials_payload()["credentials"].items()
                if key != "openai_url"
            }
        }),
        encoding="utf-8",
    )

    client = build_asksage_openai_client(
        credentials_path=credentials_path,
        use_truststore=False,
    )

    assert str(client.base_url) == f"{ASKSAGE_SERVER_BASE_URL}/openai/v1/"


def test_normalize_asksage_openai_base_url_accepts_chat_completions_url() -> None:
    assert (
        normalize_asksage_openai_base_url(
            "https://asksage-openai.example.test/v1/chat/completions"
        )
        == "https://asksage-openai.example.test/v1"
    )


def test_normalize_asksage_openai_base_url_handles_separate_openai_host() -> None:
    assert (
        normalize_asksage_openai_base_url(
            "https://asksage-openai.example.test",
            explicit_openai_url=True,
        )
        == "https://asksage-openai.example.test/v1"
    )


def test_resolve_asksage_openai_base_url_prefers_credentials_openai_url(
    tmp_path,
) -> None:
    credentials_path = tmp_path / "credentials.json"
    credentials_path.write_text(
        json.dumps(
            {
                "credentials": {
                    "api_key": "test-token",
                    "Ask_sage_user_info": {"username": "user@example.test"},
                    "server_base_url": "https://asksage.example.test/server",
                    "openai_url": "https://asksage-openai.example.test",
                }
            }
        ),
        encoding="utf-8",
    )

    assert (
        resolve_asksage_openai_base_url(credentials_path=credentials_path)
        == "https://asksage-openai.example.test/v1"
    )


def test_asksage_classifier_uses_chat_completions_structured_outputs(tmp_path) -> None:
    completions = FakeCompletions()
    classifier = AskSageChallengeClassifier(
        model=ASKSAGE_MODEL,
        client=FakeOpenAIClient(completions),
        audit_cache_path=tmp_path / "audit.jsonl",
    )

    decision = classifier(_payload())
    call = completions.calls[0]

    assert decision["predicted_challenge"] == "Challenge A"
    assert call["model"] == ASKSAGE_MODEL
    assert call["temperature"] == 0
    assert call["response_format"]["type"] == "json_schema"
    assert call["response_format"]["json_schema"]["strict"] is True
    assert (
        call["response_format"]["json_schema"]["schema"]["properties"][
            "predicted_challenge"
        ]["enum"]
        == ["Challenge A"]
    )
    assert (tmp_path / "audit.jsonl").exists()


def test_asksage_classifier_resume_cache_avoids_api_call(tmp_path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    payload = _payload()
    AskSageChallengeClassifier(
        model=ASKSAGE_MODEL,
        client=FakeOpenAIClient(FakeCompletions()),
        audit_cache_path=audit_path,
    )(payload)

    cached = AskSageChallengeClassifier(
        model=ASKSAGE_MODEL,
        client=FakeOpenAIClient(FakeCompletions(raises=True)),
        audit_cache_path=audit_path,
        resume_cache=True,
    )

    assert cached(payload)["predicted_challenge"] == "Challenge A"


def test_asksage_ai_directness_classifier_uses_structured_outputs_and_cache(tmp_path) -> None:
    audit_path = tmp_path / "ai-directness-audit.jsonl"
    completions = FakeCompletions()
    classifier = AskSageAIDirectnessClassifier(
        model=ASKSAGE_MODEL,
        client=FakeOpenAIClient(completions),
        audit_cache_path=audit_path,
    )
    payload = {
        "instructions": "score AI directness",
        "predicted_challenge": "Challenge A",
        "programming_evidence": {
            "programming_record_id": "programming_record:test",
            "source_row_number": 1,
            "wbs_name": "AI workflow",
            "scope_description": "Build an AI-enabled workflow.",
        },
    }

    decision = classifier(payload)
    call = completions.calls[0]

    assert decision["ai_directness_score"] == 87
    assert decision["ai_directness_label"] == "ai_enabled_workflow"
    assert call["response_format"]["json_schema"]["name"] == "programming_ai_directness"
    assert (
        "ai_directness_score"
        in call["response_format"]["json_schema"]["schema"]["properties"]
    )
    assert audit_path.exists()

    cached = AskSageAIDirectnessClassifier(
        model=ASKSAGE_MODEL,
        client=FakeOpenAIClient(FakeCompletions(raises=True)),
        audit_cache_path=audit_path,
        resume_cache=True,
    )

    assert cached(payload)["ai_directness_score"] == 87


def test_asksage_classifier_uses_credentials_model_when_present(tmp_path) -> None:
    credentials_path = tmp_path / "credentials.json"
    credentials_path.write_text(json.dumps(_credentials_payload()), encoding="utf-8")

    classifier = AskSageChallengeClassifier(
        credentials_path=credentials_path,
        client=FakeOpenAIClient(FakeCompletions()),
    )

    assert classifier.model == ASKSAGE_MODEL


def test_asksage_classifier_defaults_model_without_credentials(monkeypatch) -> None:
    monkeypatch.delenv("ASKSAGE_MODEL", raising=False)

    classifier = AskSageChallengeClassifier(
        credentials_path="missing-credentials.json",
        client=FakeOpenAIClient(FakeCompletions()),
    )

    assert classifier.model == DEFAULT_ASKSAGE_MODEL


def test_parse_asksage_chat_completion_decision_accepts_markdown_json() -> None:
    decision = parse_asksage_chat_completion_decision(
        {
            "choices": [
                {
                    "message": {
                        "content": (
                            "```json\n"
                            '{"predicted_challenge": "Challenge A", '
                            '"confidence": "high", "rationale": "Good fit."}'
                            "\n```"
                        )
                    }
                }
            ]
        }
    )

    assert decision["predicted_challenge"] == "Challenge A"
    assert decision["confidence"] == "high"


def test_parse_asksage_chat_completion_decision_reports_non_json_preview() -> None:
    with pytest.raises(ValueError, match="Response preview: I cannot comply"):
        parse_asksage_chat_completion_decision(
            {"choices": [{"message": {"content": "I cannot comply with that format."}}]}
        )


def test_graph_asksage_entry_point_preserves_row_level_outputs(tmp_path) -> None:
    graph = build_combined_ontology()
    results = classify_graph_with_asksage(
        graph,
        model=ASKSAGE_MODEL,
        limit=1,
        asksage_client=FakeOpenAIClient(FakeCompletions()),
        audit_cache_path=tmp_path / "audit.jsonl",
    )

    assert len(results) == 1
    assert results[0].programming_record_id.startswith("programming_record:")
    assert results[0].source_sub_office == "NA11"
    assert results[0].normalized_funding_office == "NA11"
    assert 0 <= results[0].ai_directness_score <= 100
