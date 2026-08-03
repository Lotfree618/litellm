from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from litellm.types.extract import ExtractRequest, ExtractResponse


class ExtractProviderError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        headers: Optional[Dict[str, str]] = None,
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
        api_key: Optional[str],
        api_base: Optional[str],
        headers: Dict[str, str],
    ) -> Dict[str, str]:
        raise NotImplementedError

    @abstractmethod
    def get_complete_url(self, *, api_base: Optional[str]) -> str:
        raise NotImplementedError

    @abstractmethod
    def transform_request(self, request: ExtractRequest) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def transform_response(self, *, request: ExtractRequest, payload: Dict[str, Any]) -> ExtractResponse:
        raise NotImplementedError
