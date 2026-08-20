from abc import ABC, abstractmethod
from typing import Any

from litellm.types.extract import ExtractRequest, ExtractResponse


class ExtractProviderError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.headers = headers or {}


class BaseExtractConfig(ABC):
    @staticmethod
    @abstractmethod
    def ui_friendly_name() -> str:
        raise NotImplementedError

    @abstractmethod
    def validate_environment(
        self,
        *,
        api_key: str | None,
        api_base: str | None,
        headers: dict[str, str],
    ) -> dict[str, str]:
        raise NotImplementedError

    @abstractmethod
    def get_complete_url(self, *, api_base: str | None) -> str:
        raise NotImplementedError

    @abstractmethod
    def transform_request(self, request: ExtractRequest) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def transform_response(self, *, request: ExtractRequest, payload: dict[str, Any]) -> ExtractResponse:
        raise NotImplementedError
