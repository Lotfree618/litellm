from typing import Any

from litellm.llms.base_llm.extract.transformation import BaseExtractConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.extract import ExtractContents, ExtractData, ExtractRequest, ExtractResponse


class FirecrawlExtractConfig(BaseExtractConfig):
    FIRECRAWL_API_BASE = "https://api.firecrawl.dev/v2"
    ALLOWED_PROVIDER_OPTIONS = frozenset()

    @staticmethod
    def ui_friendly_name() -> str:
        return "Firecrawl"

    def validate_environment(
        self,
        *,
        api_key: str | None,
        api_base: str | None,
        headers: dict[str, str],
    ) -> dict[str, str]:
        resolved_api_key = api_key or get_secret_str("FIRECRAWL_API_KEY")
        if not resolved_api_key:
            raise ValueError("FIRECRAWL_API_KEY is not set")
        return {
            **headers,
            "Authorization": f"Bearer {resolved_api_key}",
            "Content-Type": "application/json",
        }

    def get_complete_url(self, *, api_base: str | None) -> str:
        resolved_api_base = (api_base or get_secret_str("FIRECRAWL_API_BASE") or self.FIRECRAWL_API_BASE).rstrip("/")
        if resolved_api_base.endswith("/scrape"):
            return resolved_api_base
        return f"{resolved_api_base}/scrape"

    def transform_request(self, request: ExtractRequest) -> dict[str, Any]:
        unsupported_options = set(request.provider_options) - self.ALLOWED_PROVIDER_OPTIONS
        if unsupported_options:
            unsupported = ", ".join(sorted(unsupported_options))
            raise ValueError(f"Unsupported Firecrawl provider_options: {unsupported}")

        formats = ["rawHtml" if item == "raw_html" else item for item in request.formats]
        payload: dict[str, Any] = {
            "url": request.url,
            "formats": formats,
            "onlyMainContent": request.only_main_content,
        }
        if request.include_tags:
            payload["includeTags"] = request.include_tags
        if request.exclude_tags:
            payload["excludeTags"] = request.exclude_tags
        if request.max_age is not None:
            payload["maxAge"] = request.max_age
        return payload

    def transform_response(self, *, request: ExtractRequest, payload: dict[str, Any]) -> ExtractResponse:
        if payload.get("success") is False:
            raise ValueError("Firecrawl reported an unsuccessful scrape")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise ValueError("Firecrawl response is missing data")

        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        source_url = metadata.get("sourceURL") or metadata.get("url") or request.url
        title = metadata.get("title") or data.get("title")
        links = data.get("links") if isinstance(data.get("links"), list) else []

        return ExtractResponse(
            data=ExtractData(
                url=str(source_url),
                title=str(title) if title is not None else None,
                contents=ExtractContents(
                    markdown=data.get("markdown") if isinstance(data.get("markdown"), str) else None,
                    html=data.get("html") if isinstance(data.get("html"), str) else None,
                    raw_html=data.get("rawHtml") if isinstance(data.get("rawHtml"), str) else None,
                ),
                links=[str(link) for link in links],
                metadata=metadata,
            )
        )
