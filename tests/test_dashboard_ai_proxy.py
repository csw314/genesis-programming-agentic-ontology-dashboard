from __future__ import annotations

import json

import httpx
import pytest

from programming_ontology.dashboard_ai_proxy import (
    AskSageProxyError,
    list_asksage_dashboard_models,
    prepare_dashboard_context_text,
    request_asksage_dashboard_insight,
)


BASE_URL = "https://asksage.example.test/server/openai/v1"


def _credentials() -> dict:
    return {"email": "user@example.test", "apiKey": "test-token"}


def test_list_dashboard_models_filters_to_image_capable_models() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert str(request.url) == f"{BASE_URL}/models"
        assert request.headers["authorization"] == "Bearer test-token"
        assert request.headers["x-access-tokens"] == "test-token"
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "gpt-4.1", "name": "GPT 4.1", "input": ["text", "image"], "cost": "medium"},
                    {"id": "gpt-o3-mini", "name": "O3 Mini", "input": ["text"]},
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))

    result = list_asksage_dashboard_models(
        {"credentials": _credentials()},
        http_client=client,
        base_url=BASE_URL,
    )

    model_ids = {model["id"] for model in result["models"]}
    assert "gpt-4.1" in model_ids
    assert "gpt-o3-mini" not in model_ids
    assert result["models"][0]["supports_image"] is True


def test_list_dashboard_models_reports_invalid_credentials_without_secret() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                401,
                json={"error": "Bearer test-token was rejected"},
            )
        )
    )

    with pytest.raises(AskSageProxyError) as exc_info:
        list_asksage_dashboard_models(
            {"credentials": _credentials()},
            http_client=client,
            base_url=BASE_URL,
        )

    assert exc_info.value.code == "invalid_credentials"
    assert "test-token" not in exc_info.value.message


def test_dashboard_insight_sends_context_and_image_to_asksage() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        assert request.method == "POST"
        assert str(request.url) == f"{BASE_URL}/chat/completions"
        assert request.headers["authorization"] == "Bearer test-token"
        assert "test-token" not in request.content.decode("utf-8")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "The chart shows concentrated BASELINE funding."
                        }
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = request_asksage_dashboard_insight(
        {
            "credentials": _credentials(),
            "model": "gpt-4.1",
            "prompt": "Summarize this visualization.",
            "context": {
                "target": {"title": "Mission Funding Mix"},
                "raw_data": [{"challenge_name": "Challenge A", "BASELINE": 100.0}],
                "ontology": {"nodes": [{"id": "n1"}], "edges": []},
            },
            "image": "data:image/png;base64,abc123",
        },
        http_client=client,
        base_url=BASE_URL,
    )

    assert result["answer"] == "The chart shows concentrated BASELINE funding."
    user_content = captured["body"]["messages"][1]["content"]
    assert user_content[0]["type"] == "text"
    assert "Mission Funding Mix" in user_content[0]["text"]
    assert user_content[1]["type"] == "image_url"
    assert user_content[1]["image_url"]["url"] == "data:image/png;base64,abc123"


def test_dashboard_insight_rejects_image_for_text_only_model() -> None:
    with pytest.raises(AskSageProxyError) as exc_info:
        request_asksage_dashboard_insight(
            {
                "credentials": _credentials(),
                "model": "gpt-o3-mini",
                "prompt": "Summarize.",
                "context": {"raw_data": []},
                "image": "data:image/png;base64,abc123",
            },
            http_client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200))),
            base_url=BASE_URL,
        )

    assert exc_info.value.code == "unsupported_image_model"


def test_dashboard_insight_maps_context_limit_errors() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                400,
                json={"error": {"message": "maximum context length exceeded"}},
            )
        )
    )

    with pytest.raises(AskSageProxyError) as exc_info:
        request_asksage_dashboard_insight(
            {
                "credentials": _credentials(),
                "model": "gpt-4.1",
                "prompt": "Summarize.",
                "context": {"raw_data": [{"value": "x"}]},
            },
            http_client=client,
            base_url=BASE_URL,
        )

    assert exc_info.value.code == "context_limit"


def test_prepare_dashboard_context_text_truncates_oversized_context() -> None:
    text = prepare_dashboard_context_text(
        {
            "raw_data": [{"value": "x" * 500} for _ in range(200)],
            "ontology": {
                "nodes": [{"id": str(index), "label": "node"} for index in range(200)],
                "edges": [{"source": "a", "target": "b"} for _ in range(200)],
            },
        },
        max_chars=2_000,
    )

    assert len(text) <= 2_000
    assert "dashboard context truncated" in text
