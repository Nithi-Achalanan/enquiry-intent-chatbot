"""Bounded, observable retry handling for the external model provider."""

from __future__ import annotations

import logging
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, TypeVar


logger = logging.getLogger(__name__)
Result = TypeVar("Result")
RETRYABLE_STATUS_CODES = {408, 429}


@dataclass(frozen=True)
class InvocationDiagnostic:
    agent: str
    attempts: int
    retry_count: int
    retryable: bool
    error_type: str
    status_code: int | None
    request_id: str | None


class ModelInvocationError(RuntimeError):
    def __init__(self, diagnostic: InvocationDiagnostic) -> None:
        self.diagnostic = diagnostic
        super().__init__(f"{diagnostic.agent} model invocation failed after {diagnostic.attempts} attempt(s).")

    def artifact(self) -> dict[str, Any]:
        return asdict(self.diagnostic)


def _status_code(error: Exception) -> int | None:
    for value in (getattr(error, "status_code", None), getattr(getattr(error, "response", None), "status_code", None)):
        if isinstance(value, int):
            return value
    return None


def _request_id(error: Exception) -> str | None:
    headers = getattr(getattr(error, "response", None), "headers", None)
    if headers is None:
        return None
    value = headers.get("x-request-id")
    return str(value) if value else None


def _error_chain(error: Exception) -> list[Exception]:
    chain = []
    current: Exception | None = error
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def is_retryable_model_error(error: Exception) -> bool:
    for candidate in _error_chain(error):
        status_code = _status_code(candidate)
        if status_code is not None:
            return status_code in RETRYABLE_STATUS_CODES or status_code >= 500
        if isinstance(candidate, (TimeoutError, ConnectionError)) or candidate.__class__.__name__ in {
            "APITimeoutError",
            "APIConnectionError",
            "ConnectTimeout",
            "ReadTimeout",
            "TimeoutException",
            "NetworkError",
        }:
            return True
    return False


def invoke_with_retry(
    operation: Callable[[], Result],
    *,
    agent: str,
    max_retries: int,
    base_delay_seconds: float = 0.25,
) -> Result:
    for attempt in range(1, max_retries + 2):
        try:
            return operation()
        except Exception as error:
            retryable = is_retryable_model_error(error)
            diagnostic = InvocationDiagnostic(
                agent=agent,
                attempts=attempt,
                retry_count=attempt - 1,
                retryable=retryable,
                error_type=type(error).__name__,
                status_code=_status_code(error),
                request_id=_request_id(error),
            )
            logger.warning(
                "model_invocation_failed agent=%s attempt=%d retryable=%s error_type=%s status_code=%s",
                diagnostic.agent,
                diagnostic.attempts,
                diagnostic.retryable,
                diagnostic.error_type,
                diagnostic.status_code,
            )
            if not retryable or attempt > max_retries:
                raise ModelInvocationError(diagnostic) from error
            delay = min(4.0, base_delay_seconds * (2 ** (attempt - 1))) + random.uniform(0, base_delay_seconds)
            time.sleep(delay)

    raise AssertionError("Retry loop must return or raise.")
