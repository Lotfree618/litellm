from typing import Any

import pytest

from litellm.extract.cost_calculator import extract_provider_cost_per_request
from litellm.router import Router
from litellm.router_utils.extract_api_router import ExtractAPIRouter
from litellm.types.extract import ExtractContents, ExtractData, ExtractResponse


def test_extract_cost_uses_the_configured_per_request_price(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_get_model_info(*, model: str, custom_llm_provider: str) -> dict[str, float]:
        captured.update(model=model, custom_llm_provider=custom_llm_provider)
        return {"input_cost_per_query": 0.005}

    monkeypatch.setattr("litellm.extract.cost_calculator.get_model_info", fake_get_model_info)

    assert extract_provider_cost_per_request(model="web-extract", custom_llm_provider="firecrawl") == (0.005, 0.0)
    assert captured == {"model": "web-extract", "custom_llm_provider": "firecrawl"}


@pytest.mark.asyncio
async def test_router_extract_uses_the_selected_extract_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    router = Router(
        model_list=[],
        extract_tools=[
            {
                "extract_tool_name": "web-extract",
                "litellm_params": {"extract_provider": "firecrawl"},
            }
        ],
    )
    captured: dict[str, Any] = {}

    async def fake_async_extract(**kwargs: Any) -> ExtractResponse:
        captured.update(kwargs)
        return ExtractResponse(
            data=ExtractData(
                url=kwargs["request"].url,
                contents=ExtractContents(markdown="content"),
            )
        )

    monkeypatch.setattr(ExtractAPIRouter, "async_extract", fake_async_extract)

    response = await router.aextract(
        model="web-extract",
        url="https://example.com",
        formats=["markdown"],
        only_main_content=True,
    )

    assert response.data.contents.markdown == "content"
    assert captured["router_instance"] is router
    assert captured["extract_tool_name"] == "web-extract"
    assert captured["request"].url == "https://example.com"


@pytest.mark.asyncio
async def test_router_extract_preserves_proxy_logging_context(monkeypatch: pytest.MonkeyPatch) -> None:
    router = Router(
        model_list=[],
        extract_tools=[
            {
                "extract_tool_name": "web-extract",
                "litellm_params": {"extract_provider": "firecrawl"},
            }
        ],
    )
    captured: dict[str, Any] = {}

    async def fake_async_extract(**kwargs: Any) -> ExtractResponse:
        captured.update(kwargs)
        return ExtractResponse(data=ExtractData(url=kwargs["request"].url))

    monkeypatch.setattr(ExtractAPIRouter, "async_extract", fake_async_extract)
    marker = object()

    await router.aextract(
        model="web-extract",
        url="https://example.com",
        litellm_logging_obj=marker,
    )

    assert captured["litellm_logging_obj"] is marker
