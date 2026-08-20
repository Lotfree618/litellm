from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing_extensions import Required, TypedDict

ExtractFormat = Literal["markdown", "html", "raw_html", "links"]


class ExtractRequest(BaseModel):
    url: str
    formats: list[ExtractFormat] = Field(default_factory=lambda: ["markdown"])
    only_main_content: bool = True
    include_tags: list[str] = Field(default_factory=list)
    exclude_tags: list[str] = Field(default_factory=list)
    max_age: int | None = Field(default=None, ge=0)
    provider_options: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    @field_validator("formats")
    @classmethod
    def validate_formats(cls, value: list[ExtractFormat]) -> list[ExtractFormat]:
        if not value:
            raise ValueError("formats must contain at least one item")
        return list(dict.fromkeys(value))


class ExtractContents(BaseModel):
    markdown: str | None = None
    html: str | None = None
    raw_html: str | None = None


class ExtractData(BaseModel):
    url: str
    title: str | None = None
    contents: ExtractContents = Field(default_factory=ExtractContents)
    links: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractResponse(BaseModel):
    object: Literal["web_extract"] = "web_extract"
    data: ExtractData


class ExtractToolLiteLLMParams(TypedDict, total=False):
    extract_provider: Required[str]
    api_key: str | None
    api_base: str | None
    failure_domain: str | None
    weight: float | None
    max_parallel_requests: int | None
    timeout: float | None
    num_retries: int | None


class ExtractToolTypedDict(TypedDict, total=False):
    extract_tool_name: Required[str]
    litellm_params: Required[ExtractToolLiteLLMParams]
    extract_tool_info: dict[str, Any] | None
    extract_tool_id: str | None
