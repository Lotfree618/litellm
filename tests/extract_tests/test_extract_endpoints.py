from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.extract_endpoints import endpoints
from litellm.proxy.extract_endpoints.security import (
    UnsafeExtractURL,
    validate_public_redirect_chain,
)
from litellm.types.extract import ExtractContents, ExtractData, ExtractResponse


def _user(*, allowed: bool) -> UserAPIKeyAuth:
    permission = SimpleNamespace(extract_tools=["web-extract"] if allowed else [])
    return UserAPIKeyAuth.model_construct(
        user_role=LitellmUserRoles.INTERNAL_USER,
        object_permission=permission,
        team_object_permission=None,
    )


def test_extract_permission_requires_explicit_tool() -> None:
    with pytest.raises(HTTPException) as exc:
        endpoints._authorize_extract_tool(
            extract_tool_name="web-extract",
            user_api_key_dict=_user(allowed=False),
        )

    assert exc.value.status_code == 403


def test_extract_permission_allows_explicit_tool_and_admin() -> None:
    endpoints._authorize_extract_tool(
        extract_tool_name="web-extract",
        user_api_key_dict=_user(allowed=True),
    )
    endpoints._authorize_extract_tool(
        extract_tool_name="web-extract",
        user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
    )


@pytest.mark.asyncio
async def test_firecrawl_compatibility_response(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_execute_extract(**_: Any) -> ExtractResponse:
        return ExtractResponse(
            data=ExtractData(
                url="https://example.com",
                title="Example",
                contents=ExtractContents(markdown="content"),
                metadata={"title": "Example"},
            )
        )

    monkeypatch.setattr(endpoints, "_execute_extract", fake_execute_extract)
    response = await endpoints.firecrawl_scrape_compatibility_endpoint(
        request=None,  # type: ignore[arg-type]
        scrape_request=endpoints.FirecrawlScrapeRequest(
            url="https://example.com",
            formats=["markdown"],
        ),
        user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
    )

    assert response == {
        "success": True,
        "data": {
            "metadata": {"title": "Example"},
            "markdown": "content",
        },
    }


@pytest.mark.asyncio
async def test_redirect_chain_revalidates_redirect_target() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": "http://internal.example/admin"}, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(UnsafeExtractURL):
            await validate_public_redirect_chain("https://93.184.216.34/start", client=client)
