from __future__ import annotations

from io import BytesIO
import json

from docx import Document
import httpx
import pytest

from programming_ontology.dashboard_ai_proxy import (
    AskSageProxyError,
    REPORT_DOCX_MIME,
    build_dashboard_report_docx,
    list_asksage_dashboard_models,
    prepare_dashboard_context_text,
    request_asksage_dashboard_insight,
    request_asksage_dashboard_report,
)


BASE_URL = "https://asksage.example.test/server/openai/v1"
TINY_PNG = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


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


def _report_payload() -> dict:
    return {
        "credentials": _credentials(),
        "model": "gpt-4.1",
        "dashboard": {
            "title": "Genesis Mission Funding and Collaboration Dashboard",
            "generated_at_client": "2026-06-08T18:00:00",
            "metadata": "Generated from test dashboard snapshot.",
            "current_view": "combined",
            "assumptions": ["Executive views use BASELINE, UFR, and ROT rows only."],
        },
        "targets": [
            {
                "context": {
                    "target": {
                        "id": "combined-mission-funding-mix",
                        "title": "Mission Funding Mix",
                        "type": "table",
                    },
                    "metadata": {"legend": ["BASELINE", "UFR", "ROT"]},
                    "raw_data": [
                        {
                            "challenge_name": "Challenge A",
                            "BASELINE": 100.0,
                            "UFR": 20.0,
                            "ROT": 0.0,
                        }
                    ],
                    "ontology": {
                        "summary": {"nodes": 10, "edges": 12},
                        "selection": {"node_count": 3, "edge_count": 2},
                        "nodes": [{"id": "n1", "label": "Challenge A"}],
                        "edges": [],
                    },
                },
                "image": TINY_PNG,
                "session_insight": "Challenge A has the largest funding concentration.",
                "text_snapshot": "Challenge A | $120M | BASELINE-heavy",
            }
        ],
    }


def test_dashboard_report_generates_docx_and_sends_consolidated_context() -> None:
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
                            "content": (
                                "# Executive Summary\n"
                                "Challenge A is the primary funding signal.\n\n"
                                "## Key Findings\n"
                                "- BASELINE funding is concentrated.\n\n"
                                "## Conclusion and Recommended Next Steps\n"
                                "Validate the funding signal with source rows."
                            )
                        }
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    report = request_asksage_dashboard_report(
        _report_payload(),
        http_client=client,
        base_url=BASE_URL,
    )

    assert report.content_type == REPORT_DOCX_MIME
    assert report.filename == "dashboard_executive_summary_2026-06-08.docx"
    assert report.body.startswith(b"PK")

    user_content = captured["body"]["messages"][1]["content"]
    assert user_content[0]["type"] == "text"
    assert "Mission Funding Mix" in user_content[0]["text"]
    assert "session_insight" in user_content[0]["text"]
    assert user_content[1]["type"] == "image_url"

    document = Document(BytesIO(report.body))
    paragraphs = "\n".join(paragraph.text for paragraph in document.paragraphs)
    assert "Genesis Mission Funding and Collaboration Dashboard" in paragraphs
    assert "Executive Narrative" in paragraphs
    assert "Supporting Dashboard Evidence" in paragraphs
    assert "Mission Funding Mix" in paragraphs
    assert "Conclusion and Recommended Next Steps" in paragraphs


def test_dashboard_report_rejects_missing_dashboard_context() -> None:
    with pytest.raises(AskSageProxyError) as exc_info:
        request_asksage_dashboard_report(
            {"credentials": _credentials(), "model": "gpt-4.1", "targets": []},
            http_client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200))),
            base_url=BASE_URL,
        )

    assert exc_info.value.code == "missing_dashboard_context"


def test_dashboard_report_rejects_image_for_text_only_model() -> None:
    payload = _report_payload()
    payload["model"] = "gpt-o3-mini"

    with pytest.raises(AskSageProxyError) as exc_info:
        request_asksage_dashboard_report(
            payload,
            http_client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200))),
            base_url=BASE_URL,
        )

    assert exc_info.value.code == "unsupported_image_model"


def test_build_dashboard_report_docx_includes_tables_without_images() -> None:
    context = {
        "generated_at": "2026-06-08T18:00:00",
        "dashboard": {
            "title": "Genesis Mission Funding and Collaboration Dashboard",
            "current_view": "combined",
            "assumptions": ["Test assumption."],
        },
        "targets": [
            {
                "id": "target-1",
                "title": "Challenge Coverage",
                "raw_data_summary": {
                    "row_count": 1,
                    "columns": ["challenge_name", "total_funding"],
                    "sample_rows": [{"challenge_name": "Challenge A", "total_funding": 100}],
                },
            }
        ],
    }

    body = build_dashboard_report_docx(
        context,
        "# Executive Summary\nThe dashboard shows a clear funding signal.",
    )
    document = Document(BytesIO(body))
    paragraphs = "\n".join(paragraph.text for paragraph in document.paragraphs)

    assert "Current Dashboard Selections and Assumptions" in paragraphs
    assert "Challenge Coverage" in paragraphs
    assert len(document.tables) >= 1
