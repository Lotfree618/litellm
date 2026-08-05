from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, ConfigDict, Field

from litellm.extract import aextract
from litellm.llms.base_llm.extract.transformation import ExtractProviderError
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.extract_endpoints.security import (
    UnsafeExtractURL,
    validate_public_redirect_chain,
)
from litellm.router_utils.extract_api_router import ExtractAPIRouter
from litellm.types.extract import ExtractRequest, ExtractResponse


router = APIRouter()


class FirecrawlScrapeRequest(BaseModel):
    url: str
    formats: List[str] = Field(default_factory=lambda: ["markdown"])
    only_main_content: bool = Field(default=True, alias="onlyMainContent")
    include_tags: List[str] = Field(default_factory=list, alias="includeTags")
    exclude_tags: List[str] = Field(default_factory=list, alias="excludeTags")
    max_age: Optional[int] = Field(default=None, alias="maxAge", ge=0)
    skip_tls_verification: Optional[bool] = Field(
        default=None, alias="skipTlsVerification"
    )
    remove_base64_images: Optional[bool] = Field(
        default=None, alias="removeBase64Images"
    )
    fast_mode: Optional[bool] = Field(default=None, alias="fastMode")
    block_ads: Optional[bool] = Field(default=None, alias="blockAds")
    store_in_cache: Optional[bool] = Field(default=None, alias="storeInCache")
    mobile: Optional[bool] = None
    # firecrawl-py v2 attaches its SDK identity to every request. This is
    # transport metadata, not an extract option, so LiteLLM deliberately
    # accepts it without forwarding it downstream.
    origin: Optional[str] = Field(default=None, max_length=256)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


def _is_proxy_admin(user_api_key_dict: UserAPIKeyAuth) -> bool:
    role = user_api_key_dict.user_role
    return role in {LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY}


def _permission_extract_tools(permission: Any) -> List[str]:
    if permission is None:
        return []
    value = getattr(permission, "extract_tools", None)
    if value is None and isinstance(permission, dict):
        value = permission.get("extract_tools")
    return list(value or [])


def _authorize_extract_tool(*, extract_tool_name: str, user_api_key_dict: UserAPIKeyAuth) -> None:
    if _is_proxy_admin(user_api_key_dict):
        return
    if extract_tool_name not in _permission_extract_tools(user_api_key_dict.object_permission):
        raise HTTPException(status_code=403, detail={"error": "extract_tool_access_denied"})
    if user_api_key_dict.team_object_permission is not None and extract_tool_name not in _permission_extract_tools(
        user_api_key_dict.team_object_permission
    ):
        raise HTTPException(status_code=403, detail={"error": "extract_tool_access_denied"})


async def _execute_extract(
    *,
    extract_tool_name: str,
    extract_request: ExtractRequest,
) -> ExtractResponse:
    from litellm.proxy.proxy_server import llm_router

    if llm_router is None:
        raise HTTPException(status_code=503, detail={"error": "extract_router_unavailable"})
    try:
        await validate_public_redirect_chain(extract_request.url)
        return await ExtractAPIRouter.async_extract(
            router_instance=llm_router,
            extract_tool_name=extract_tool_name,
            request=extract_request,
            original_function=aextract,
        )
    except UnsafeExtractURL as error:
        raise HTTPException(status_code=422, detail={"error": "unsafe_extract_url", "message": str(error)}) from error
    except ExtractProviderError as error:
        raise HTTPException(
            status_code=error.status_code,
            detail={"error": "extract_provider_error", "message": str(error)},
            headers={key: value for key, value in error.headers.items() if key.lower() == "retry-after"},
        ) from error
    except ValueError as error:
        raise HTTPException(
            status_code=422, detail={"error": "invalid_extract_request", "message": str(error)}
        ) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail={"error": "extract_deployment_unavailable"}) from error


@router.post(
    "/v1/extract/{extract_tool_name}",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["extract"],
)
@router.post(
    "/extract/{extract_tool_name}",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["extract"],
)
async def extract_endpoint(
    extract_tool_name: str,
    extract_request: ExtractRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Dict[str, Any]:
    _authorize_extract_tool(extract_tool_name=extract_tool_name, user_api_key_dict=user_api_key_dict)
    response = await _execute_extract(extract_tool_name=extract_tool_name, extract_request=extract_request)
    return response.model_dump(mode="json")


@router.post(
    "/v2/scrape",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["extract"],
)
async def firecrawl_scrape_compatibility_endpoint(
    scrape_request: FirecrawlScrapeRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Dict[str, Any]:
    extract_tool_name = "web-extract"
    _authorize_extract_tool(extract_tool_name=extract_tool_name, user_api_key_dict=user_api_key_dict)
    try:
        provider_options = {
            key: value
            for key, value in {
                "skipTlsVerification": scrape_request.skip_tls_verification,
                "removeBase64Images": scrape_request.remove_base64_images,
                "fastMode": scrape_request.fast_mode,
                "blockAds": scrape_request.block_ads,
                "storeInCache": scrape_request.store_in_cache,
                "mobile": scrape_request.mobile,
            }.items()
            if value is not None
        }
        extract_request = ExtractRequest(
            url=scrape_request.url,
            formats=["raw_html" if item == "rawHtml" else item for item in scrape_request.formats],
            only_main_content=scrape_request.only_main_content,
            include_tags=scrape_request.include_tags,
            exclude_tags=scrape_request.exclude_tags,
            max_age=scrape_request.max_age,
            provider_options=provider_options,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=422, detail={"error": "invalid_extract_request", "message": str(error)}
        ) from error

    response = await _execute_extract(extract_tool_name=extract_tool_name, extract_request=extract_request)
    data = response.data
    firecrawl_data: Dict[str, Any] = {
        "metadata": data.metadata,
    }
    if data.contents.markdown is not None:
        firecrawl_data["markdown"] = data.contents.markdown
    if data.contents.html is not None:
        firecrawl_data["html"] = data.contents.html
    if data.contents.raw_html is not None:
        firecrawl_data["rawHtml"] = data.contents.raw_html
    if data.links:
        firecrawl_data["links"] = data.links
    return {"success": True, "data": firecrawl_data}
