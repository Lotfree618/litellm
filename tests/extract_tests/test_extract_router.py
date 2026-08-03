import asyncio
from typing import Any

import pytest

from litellm.llms.base_llm.extract.transformation import ExtractProviderError
from litellm.router_utils.extract_api_router import ExtractAPIRouter
from litellm.types.extract import ExtractData, ExtractRequest, ExtractResponse


class _Router:
    def __init__(self, tools: list[dict[str, Any]]) -> None:
        self.extract_tools = tools


def _tool(identifier: str, *, weight: float = 1, max_parallel: int = 2) -> dict[str, Any]:
    return {
        "extract_tool_id": identifier,
        "extract_tool_name": "web-extract",
        "litellm_params": {
            "extract_provider": "firecrawl",
            "api_key": f"key-{identifier}",
            "failure_domain": identifier,
            "weight": weight,
            "max_parallel_requests": max_parallel,
            "timeout": 50,
        },
    }


def _response() -> ExtractResponse:
    return ExtractResponse(data=ExtractData(url="https://example.com"))


@pytest.fixture(autouse=True)
def reset_router_state() -> None:
    ExtractAPIRouter.reset_state()


@pytest.mark.asyncio
async def test_router_is_deterministic_when_scores_match() -> None:
    selected_keys: list[str] = []

    async def fake_extract(**kwargs: Any) -> ExtractResponse:
        selected_keys.append(kwargs["api_key"])
        return _response()

    await ExtractAPIRouter.async_extract(
        router_instance=_Router([_tool("b"), _tool("a")]),
        extract_tool_name="web-extract",
        request=ExtractRequest(url="https://example.com"),
        original_function=fake_extract,
    )

    assert selected_keys == ["key-a"]


@pytest.mark.asyncio
async def test_router_uses_least_busy_deployment() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    selected_keys: list[str] = []

    async def fake_extract(**kwargs: Any) -> ExtractResponse:
        selected_keys.append(kwargs["api_key"])
        if kwargs["api_key"] == "key-a":
            started.set()
            await release.wait()
        return _response()

    first = asyncio.create_task(
        ExtractAPIRouter.async_extract(
            router_instance=_Router([_tool("a"), _tool("b")]),
            extract_tool_name="web-extract",
            request=ExtractRequest(url="https://example.com"),
            original_function=fake_extract,
        )
    )
    await started.wait()
    second = await ExtractAPIRouter.async_extract(
        router_instance=_Router([_tool("a"), _tool("b")]),
        extract_tool_name="web-extract",
        request=ExtractRequest(url="https://example.com"),
        original_function=fake_extract,
    )
    release.set()
    await first

    assert second.data.url == "https://example.com"
    assert selected_keys == ["key-a", "key-b"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [401, 402, 429])
async def test_router_retries_explicit_account_statuses(status_code: int) -> None:
    selected_keys: list[str] = []

    async def fake_extract(**kwargs: Any) -> ExtractResponse:
        selected_keys.append(kwargs["api_key"])
        if len(selected_keys) == 1:
            raise ExtractProviderError(
                "account error",
                status_code=status_code,
                headers={"Retry-After": "1"},
            )
        return _response()

    await ExtractAPIRouter.async_extract(
        router_instance=_Router([_tool("a"), _tool("b")]),
        extract_tool_name="web-extract",
        request=ExtractRequest(url="https://example.com"),
        original_function=fake_extract,
    )

    assert selected_keys == ["key-a", "key-b"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [400, 422, 500])
async def test_router_does_not_retry_request_or_provider_errors(status_code: int) -> None:
    selected_keys: list[str] = []

    async def fake_extract(**kwargs: Any) -> ExtractResponse:
        selected_keys.append(kwargs["api_key"])
        raise ExtractProviderError("failed", status_code=status_code)

    with pytest.raises(ExtractProviderError):
        await ExtractAPIRouter.async_extract(
            router_instance=_Router([_tool("a"), _tool("b")]),
            extract_tool_name="web-extract",
            request=ExtractRequest(url="https://example.com"),
            original_function=fake_extract,
        )

    assert selected_keys == ["key-a"]
    assert ExtractAPIRouter._credential_states["a"].inflight == 0


@pytest.mark.asyncio
async def test_router_releases_inflight_on_cancel() -> None:
    started = asyncio.Event()

    async def fake_extract(**_: Any) -> ExtractResponse:
        started.set()
        await asyncio.Event().wait()
        return _response()

    task = asyncio.create_task(
        ExtractAPIRouter.async_extract(
            router_instance=_Router([_tool("a")]),
            extract_tool_name="web-extract",
            request=ExtractRequest(url="https://example.com"),
            original_function=fake_extract,
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert ExtractAPIRouter._credential_states["a"].inflight == 0


@pytest.mark.asyncio
async def test_router_preserves_original_error_without_alternate_deployment() -> None:
    async def fake_extract(**_: Any) -> ExtractResponse:
        raise ExtractProviderError("rate limited", status_code=429, headers={"Retry-After": "1"})

    with pytest.raises(ExtractProviderError) as exc:
        await ExtractAPIRouter.async_extract(
            router_instance=_Router([_tool("a")]),
            extract_tool_name="web-extract",
            request=ExtractRequest(url="https://example.com"),
            original_function=fake_extract,
        )

    assert exc.value.status_code == 429
