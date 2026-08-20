import asyncio
from typing import Any

import httpx

from litellm.llms.base_llm.extract.transformation import (
    BaseExtractConfig,
    ExtractProviderError,
)
from litellm.llms.firecrawl.extract.transformation import FirecrawlExtractConfig
from litellm.types.extract import ExtractRequest, ExtractResponse
from litellm.utils import client


def get_extract_provider_config(provider: str) -> BaseExtractConfig:
    if provider == "firecrawl":
        return FirecrawlExtractConfig()
    raise ValueError(f"Extract is not supported for provider: {provider}")


@client
async def aextract(
    *,
    request: ExtractRequest,
    extract_provider: str,
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: float = 50,
    extra_headers: dict[str, str] | None = None,
    client: httpx.AsyncClient | None = None,
    **_: Any,
) -> ExtractResponse:
    provider_config = get_extract_provider_config(extract_provider)
    headers = provider_config.validate_environment(
        api_key=api_key,
        api_base=api_base,
        headers=extra_headers or {},
    )
    url = provider_config.get_complete_url(api_base=api_base)
    payload = provider_config.transform_request(request)

    owns_client = client is None
    async_client = client or httpx.AsyncClient(timeout=timeout)
    try:
        response = await async_client.post(url, headers=headers, json=payload, timeout=timeout)
        if response.status_code >= 400:
            raise ExtractProviderError(
                f"{extract_provider} extract failed with status {response.status_code}",
                status_code=response.status_code,
                headers={str(key): str(value) for key, value in response.headers.items()},
            )
        response_payload = response.json()
        if not isinstance(response_payload, dict):
            raise ValueError(f"{extract_provider} returned a non-object response")
        return provider_config.transform_response(request=request, payload=response_payload)
    finally:
        if owns_client:
            await async_client.aclose()


def extract(**kwargs: Any) -> ExtractResponse:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(aextract(**kwargs))
    raise RuntimeError("extract() cannot run inside an active event loop; use aextract()")
