from typing import Optional
from urllib.parse import urlparse

import httpx

from litellm.litellm_core_utils.url_utils import SSRFError, async_safe_get, validate_url


UnsafeExtractURL = SSRFError


async def validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.username is not None or parsed.password is not None:
        raise UnsafeExtractURL("URLs containing credentials are not allowed")
    validate_url(url)


async def validate_public_redirect_chain(
    url: str,
    *,
    client: Optional[httpx.AsyncClient] = None,
) -> None:
    owns_client = client is None
    async_client = client or httpx.AsyncClient(timeout=10, follow_redirects=False)
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
    finally:
        if owns_client:
            await async_client.aclose()
