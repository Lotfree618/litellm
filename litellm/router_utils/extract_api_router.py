import asyncio
import email.utils
from dataclasses import dataclass
from datetime import datetime, timezone
from time import monotonic
from typing import Any, Callable, Dict, List, Optional, Set

from litellm._logging import verbose_router_logger
from litellm.llms.base_llm.extract.transformation import ExtractProviderError
from litellm.types.extract import ExtractRequest, ExtractResponse


@dataclass
class _CredentialState:
    inflight: int = 0
    disabled: bool = False


@dataclass
class _FailureDomainState:
    cooldown_until: float = 0.0
    credits_exhausted: bool = False


class ExtractAPIRouter:
    _lock = asyncio.Lock()
    _credential_states: Dict[str, _CredentialState] = {}
    _domain_states: Dict[str, _FailureDomainState] = {}

    @classmethod
    def reset_state(cls) -> None:
        cls._credential_states.clear()
        cls._domain_states.clear()

    @staticmethod
    def _credential_id(tool: Dict[str, Any], index: int) -> str:
        return str(tool.get("extract_tool_id") or f"extract-{index}")

    @staticmethod
    def _failure_domain(tool: Dict[str, Any], credential_id: str) -> str:
        return str(tool.get("litellm_params", {}).get("failure_domain") or credential_id)

    @staticmethod
    def _retry_after_seconds(headers: Dict[str, str], default_seconds: float = 45.0) -> float:
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if not retry_after:
            return default_seconds
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            try:
                retry_at = email.utils.parsedate_to_datetime(retry_after)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError):
                return default_seconds

    @classmethod
    async def _acquire_tool(
        cls,
        *,
        tools: List[Dict[str, Any]],
        attempted_credentials: Set[str],
        attempted_domains: Set[str],
    ) -> tuple[Dict[str, Any], str, str]:
        async with cls._lock:
            now = monotonic()
            candidates: List[tuple[float, str, Dict[str, Any], str]] = []
            for index, tool in enumerate(tools):
                credential_id = cls._credential_id(tool, index)
                failure_domain = cls._failure_domain(tool, credential_id)
                if credential_id in attempted_credentials or failure_domain in attempted_domains:
                    continue
                credential_state = cls._credential_states.setdefault(credential_id, _CredentialState())
                domain_state = cls._domain_states.setdefault(failure_domain, _FailureDomainState())
                if credential_state.disabled or domain_state.credits_exhausted:
                    continue
                if domain_state.cooldown_until > now:
                    continue
                params = tool.get("litellm_params", {})
                max_parallel = int(params.get("max_parallel_requests") or 0)
                if max_parallel > 0 and credential_state.inflight >= max_parallel:
                    continue
                weight = float(params.get("weight") or 1.0)
                if weight <= 0:
                    continue
                candidates.append((credential_state.inflight / weight, credential_id, tool, failure_domain))

            if not candidates:
                raise RuntimeError("No healthy extract deployment is available")

            _, credential_id, tool, failure_domain = min(candidates, key=lambda item: (item[0], item[1]))
            cls._credential_states[credential_id].inflight += 1
            return tool, credential_id, failure_domain

    @classmethod
    async def _release_credential(cls, credential_id: str) -> None:
        async with cls._lock:
            state = cls._credential_states.setdefault(credential_id, _CredentialState())
            state.inflight = max(0, state.inflight - 1)

    @classmethod
    async def _record_provider_error(
        cls,
        *,
        credential_id: str,
        failure_domain: str,
        error: ExtractProviderError,
    ) -> None:
        async with cls._lock:
            credential_state = cls._credential_states.setdefault(credential_id, _CredentialState())
            domain_state = cls._domain_states.setdefault(failure_domain, _FailureDomainState())
            if error.status_code == 401:
                credential_state.disabled = True
            elif error.status_code == 402:
                domain_state.credits_exhausted = True
            elif error.status_code == 429:
                domain_state.cooldown_until = monotonic() + cls._retry_after_seconds(error.headers)

    @classmethod
    async def async_extract(
        cls,
        *,
        router_instance: Any,
        extract_tool_name: str,
        request: ExtractRequest,
        original_function: Callable[..., Any],
        max_attempts: int = 2,
    ) -> ExtractResponse:
        matching_tools = [
            tool
            for tool in getattr(router_instance, "extract_tools", [])
            if tool.get("extract_tool_name") == extract_tool_name
        ]
        if not matching_tools:
            raise ValueError(f"Extract tool '{extract_tool_name}' not found")

        attempted_credentials: Set[str] = set()
        attempted_domains: Set[str] = set()
        last_error: Optional[BaseException] = None

        for attempt in range(1, max_attempts + 1):
            try:
                tool, credential_id, failure_domain = await cls._acquire_tool(
                    tools=matching_tools,
                    attempted_credentials=attempted_credentials,
                    attempted_domains=attempted_domains,
                )
            except RuntimeError:
                if last_error is not None:
                    raise last_error
                raise
            params = dict(tool.get("litellm_params", {}))
            attempted_credentials.add(credential_id)
            verbose_router_logger.info(
                "Extract deployment selected tool=%s provider=%s credential_id=%s failure_domain=%s attempt=%s",
                extract_tool_name,
                params.get("extract_provider"),
                credential_id,
                failure_domain,
                attempt,
            )
            try:
                return await original_function(
                    request=request,
                    extract_provider=params.get("extract_provider"),
                    api_key=params.get("api_key"),
                    api_base=params.get("api_base"),
                    timeout=float(params.get("timeout") or 50),
                )
            except ExtractProviderError as error:
                last_error = error
                await cls._record_provider_error(
                    credential_id=credential_id,
                    failure_domain=failure_domain,
                    error=error,
                )
                if error.status_code not in {401, 402, 429} or attempt >= max_attempts:
                    raise
                if error.status_code in {402, 429}:
                    attempted_domains.add(failure_domain)
            finally:
                await cls._release_credential(credential_id)

        if last_error is not None:
            raise last_error
        raise RuntimeError("Extract request failed without an upstream error")
