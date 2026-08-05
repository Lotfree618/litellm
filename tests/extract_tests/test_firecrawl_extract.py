import pytest

from litellm.llms.firecrawl.extract.transformation import FirecrawlExtractConfig
from litellm.types.extract import ExtractRequest


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
        provider_options={
            "blockAds": True,
            "skipTlsVerification": True,
        },
    )

    assert config.transform_request(request) == {
        "url": "https://example.com/article",
        "formats": ["markdown", "rawHtml", "links"],
        "onlyMainContent": False,
        "includeTags": ["article"],
        "excludeTags": ["nav"],
        "maxAge": 0,
        "blockAds": True,
        "skipTlsVerification": True,
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
