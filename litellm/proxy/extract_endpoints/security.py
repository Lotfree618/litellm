from urllib.parse import urlparse

import httpx

from litellm.litellm_core_utils.url_utils import SSRFError, async_safe_get, validate_url
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.types.llms.custom_http import httpxSpecialProvider

UnsafeExtractURL = SSRFError


async def validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.username is not None or parsed.password is not None:
        raise UnsafeExtractURL("URLs containing credentials are not allowed")
    validate_url(url)


async def validate_public_redirect_chain(
    url: str,
    *,
    client: httpx.AsyncClient | None = None,
) -> None:
    async_client = client or get_async_httpx_client(
        llm_provider=httpxSpecialProvider.Extract,
        params={"timeout": 10},
    )
    try:
        await validate_public_url(url)
        response = await async_safe_get(
            async_client,
            url,
            headers={
                "Range": "bytes=0-0",
                "User-Agent": "LiteLLM-Extract-URL-Validator/1.0",
            },
        )
        await response.aclose()
    except httpx.HTTPError as error:
        raise UnsafeExtractURL("URL redirect validation failed") from error
