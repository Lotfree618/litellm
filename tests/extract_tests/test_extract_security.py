from unittest.mock import AsyncMock, Mock

import pytest

import litellm.proxy.extract_endpoints.security as extract_security
from litellm.proxy.extract_endpoints.security import UnsafeExtractURL, validate_public_url
from litellm.types.llms.custom_http import httpxSpecialProvider


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://user:pass@example.com",
        "http://localhost/admin",
    ],
)
async def test_rejects_unsafe_url_shapes(url: str) -> None:
    with pytest.raises(UnsafeExtractURL):
        await validate_public_url(url)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "169.254.169.254",
        "192.168.1.1",
        "::1",
        "fc00::1",
    ],
)
async def test_rejects_private_resolved_addresses(address: str) -> None:
    with pytest.raises(UnsafeExtractURL):
        await validate_public_url(f"http://{f'[{address}]' if ':' in address else address}")


@pytest.mark.asyncio
async def test_accepts_public_resolved_addresses() -> None:
    await validate_public_url("https://93.184.216.34/article")


@pytest.mark.asyncio
async def test_redirect_validation_uses_cached_firecrawl_client(monkeypatch: pytest.MonkeyPatch) -> None:
    response = Mock()
    response.aclose = AsyncMock()
    get_async_client = Mock(return_value=Mock())
    monkeypatch.setattr(extract_security, "get_async_httpx_client", get_async_client)
    monkeypatch.setattr(extract_security, "validate_public_url", AsyncMock())
    monkeypatch.setattr(extract_security, "async_safe_get", AsyncMock(return_value=response))

    await extract_security.validate_public_redirect_chain("https://example.com")

    get_async_client.assert_called_once_with(
        llm_provider=httpxSpecialProvider.Extract,
        params={"timeout": 10},
    )
    response.aclose.assert_awaited_once()
