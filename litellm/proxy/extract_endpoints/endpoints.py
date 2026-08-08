from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, ConfigDict, Field

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.extract import ExtractRequest, ExtractResponse

router = APIRouter()


class FirecrawlScrapeRequest(BaseModel):
    url: str
    formats: List[str] = Field(default_factory=lambda: ["markdown"])
    only_main_content: bool = Field(default=True, alias="onlyMainContent")
    include_tags: List[str] = Field(default_factory=list, alias="includeTags")
    exclude_tags: List[str] = Field(default_factory=list, alias="excludeTags")
    max_age: Optional[int] = Field(default=None, alias="maxAge", ge=0)

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


async def _execute_extract_via_proxy(
    *,
    request: Request,
    fastapi_response: Response,
    extract_tool_name: str,
    extract_request: ExtractRequest,
    user_api_key_dict: UserAPIKeyAuth,
) -> ExtractResponse:
    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    if llm_router is None:
        raise HTTPException(status_code=503, detail={"error": "extract_router_unavailable"})

    matching_tools = [
        tool for tool in getattr(llm_router, "extract_tools", []) if tool.get("extract_tool_name") == extract_tool_name
    ]
    if not matching_tools:
        raise HTTPException(status_code=404, detail={"error": "extract_tool_not_found"})

    providers = {str(tool.get("litellm_params", {}).get("extract_provider") or "") for tool in matching_tools}
    if len(providers) != 1 or not next(iter(providers)):
        raise HTTPException(status_code=500, detail={"error": "extract_tool_provider_ambiguous"})

    data = extract_request.model_dump(mode="json")
    data["model"] = extract_tool_name
    data["extract_tool_name"] = extract_tool_name
    data["custom_llm_provider"] = next(iter(providers))
    data["metadata"] = {"model_group": extract_tool_name}
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="aextract",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as error:  # noqa: BLE001
        raise await processor._handle_llm_api_exception(
            e=error,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


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
    request: Request,
    fastapi_response: Response,
    extract_tool_name: str,
    extract_request: ExtractRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Dict[str, Any]:
    _authorize_extract_tool(extract_tool_name=extract_tool_name, user_api_key_dict=user_api_key_dict)
    response = await _execute_extract_via_proxy(
        request=request,
        fastapi_response=fastapi_response,
        extract_tool_name=extract_tool_name,
        extract_request=extract_request,
        user_api_key_dict=user_api_key_dict,
    )
    return response.model_dump(mode="json")


@router.post(
    "/firecrawl/v2/scrape",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["extract"],
)
async def firecrawl_scrape_compatibility_endpoint(
    request: Request,
    fastapi_response: Response,
    scrape_request: FirecrawlScrapeRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Dict[str, Any]:
    extract_tool_name = "web-extract"
    _authorize_extract_tool(extract_tool_name=extract_tool_name, user_api_key_dict=user_api_key_dict)
    try:
        extract_request = ExtractRequest(
            url=scrape_request.url,
            formats=["raw_html" if item == "rawHtml" else item for item in scrape_request.formats],
            only_main_content=scrape_request.only_main_content,
            include_tags=scrape_request.include_tags,
            exclude_tags=scrape_request.exclude_tags,
            max_age=scrape_request.max_age,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=422, detail={"error": "invalid_extract_request", "message": str(error)}
        ) from error

    response = await _execute_extract_via_proxy(
        request=request,
        fastapi_response=fastapi_response,
        extract_tool_name=extract_tool_name,
        extract_request=extract_request,
        user_api_key_dict=user_api_key_dict,
    )
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
