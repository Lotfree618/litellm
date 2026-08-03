from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing_extensions import Required, TypedDict


ExtractFormat = Literal["markdown", "html", "raw_html", "links"]


class ExtractRequest(BaseModel):
    url: str
    formats: List[ExtractFormat] = Field(default_factory=lambda: ["markdown"])
    only_main_content: bool = True
    include_tags: List[str] = Field(default_factory=list)
    exclude_tags: List[str] = Field(default_factory=list)
    max_age: Optional[int] = Field(default=None, ge=0)
    provider_options: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    @field_validator("formats")
    @classmethod
    def validate_formats(cls, value: List[ExtractFormat]) -> List[ExtractFormat]:
        if not value:
            raise ValueError("formats must contain at least one item")
        return list(dict.fromkeys(value))


class ExtractContents(BaseModel):
    markdown: Optional[str] = None
    html: Optional[str] = None
    raw_html: Optional[str] = None


class ExtractData(BaseModel):
    url: str
    title: Optional[str] = None
    contents: ExtractContents = Field(default_factory=ExtractContents)
    links: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExtractResponse(BaseModel):
    object: Literal["web_extract"] = "web_extract"
    data: ExtractData


class ExtractToolLiteLLMParams(TypedDict, total=False):
    extract_provider: Required[str]
    api_key: Optional[str]
    api_base: Optional[str]
    failure_domain: Optional[str]
    weight: Optional[float]
    max_parallel_requests: Optional[int]
    timeout: Optional[float]
    num_retries: Optional[int]


class ExtractToolTypedDict(TypedDict, total=False):
    extract_tool_name: Required[str]
    litellm_params: Required[ExtractToolLiteLLMParams]
    extract_tool_info: Optional[Dict[str, Any]]
    extract_tool_id: Optional[str]
