from unittest.mock import AsyncMock, Mock

import httpx
import pytest

import litellm.extract.main as extract_main
from litellm.extract.main import aextract
from litellm.llms.firecrawl.extract.transformation import FirecrawlExtractConfig
from litellm.types.extract import ExtractRequest
from litellm.types.llms.custom_http import httpxSpecialProvider


def test_firecrawl_headers_and_cloud_url() -> None:
    config = FirecrawlExtractConfig()

    assert config.validate_environment(
        api_key="fc-test",
        api_base=None,
        headers={"X-Test": "1"},
    ) == {
        "X-Test": "1",
        "Authorization": "Bearer fc-test",
        "Content-Type": "application/json",
    }
    assert config.get_complete_url(api_base="https://api.firecrawl.dev/v2") == ("https://api.firecrawl.dev/v2/scrape")


def test_firecrawl_url_does_not_duplicate_scrape() -> None:
    config = FirecrawlExtractConfig()

    assert config.get_complete_url(api_base="http://firecrawl.internal/v2/scrape") == (
        "http://firecrawl.internal/v2/scrape"
    )


def test_firecrawl_request_and_response_transformation() -> None:
    config = FirecrawlExtractConfig()
    request = ExtractRequest(
        url="https://example.com/article",
        formats=["markdown", "raw_html", "links"],
        only_main_content=False,
        include_tags=["article"],
        exclude_tags=["nav"],
        max_age=0,
    )

    assert config.transform_request(request) == {
        "url": "https://example.com/article",
        "formats": ["markdown", "rawHtml", "links"],
        "onlyMainContent": False,
        "includeTags": ["article"],
        "excludeTags": ["nav"],
        "maxAge": 0,
    }

    response = config.transform_response(
        request=request,
        payload={
            "success": True,
            "data": {
                "markdown": "content",
                "rawHtml": "<article>content</article>",
                "links": ["https://example.com/next"],
                "metadata": {
                    "sourceURL": "https://example.com/article",
                    "title": "Example",
                },
            },
        },
    )

    assert response.data.title == "Example"
    assert response.data.contents.markdown == "content"
    assert response.data.contents.raw_html == "<article>content</article>"
    assert response.data.links == ["https://example.com/next"]


def test_firecrawl_unknown_provider_option_is_rejected() -> None:
    config = FirecrawlExtractConfig()

    with pytest.raises(ValueError, match="Unsupported Firecrawl provider_options"):
        config.transform_request(
            ExtractRequest(
                url="https://example.com",
                provider_options={"proxy": "auto"},
            )
        )


@pytest.mark.asyncio
async def test_extract_uses_cached_firecrawl_client(monkeypatch: pytest.MonkeyPatch) -> None:
    async_client = Mock()
    async_client.post = AsyncMock(
        return_value=httpx.Response(
            200,
            json={
                "success": True,
                "data": {
                    "markdown": "content",
                    "metadata": {"sourceURL": "https://example.com"},
                },
            },
            request=httpx.Request("POST", "https://api.firecrawl.dev/v2/scrape"),
        )
    )
    cached_handler = Mock(client=async_client)
    get_async_client = Mock(return_value=cached_handler)
    monkeypatch.setattr(extract_main, "get_async_httpx_client", get_async_client)

    response = await aextract(
        request=ExtractRequest(url="https://example.com"),
        extract_provider="firecrawl",
        api_key="fc-test",
    )

    assert response.data.contents.markdown == "content"
    get_async_client.assert_called_once_with(
        llm_provider=httpxSpecialProvider.Extract,
        params={"timeout": 50},
    )
    async_client.post.assert_awaited_once()
