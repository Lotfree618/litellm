import pytest

from litellm.proxy.extract_endpoints.security import UnsafeExtractURL, validate_public_url


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
