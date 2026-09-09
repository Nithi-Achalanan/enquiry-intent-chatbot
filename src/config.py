"""Application configuration and startup validation."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


class ModelConfigurationError(RuntimeError):
    """Raised when the AI provider configuration cannot start the application."""


@dataclass(frozen=True)
class ModelConfiguration:
    api_key: str
    model: str
    timeout_seconds: float
    retry_attempts: int


def _environment_value(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else None


def _positive_number(name: str, default: float, value_type: type[float] | type[int], *, allow_zero: bool = False) -> float | int:
    raw_value = _environment_value(name)
    if raw_value is None:
        return default
    try:
        value = value_type(raw_value)
    except ValueError as error:
        raise ModelConfigurationError(f"{name} must be a positive number.") from error
    if value < 0 or (value == 0 and not allow_zero):
        raise ModelConfigurationError(f"{name} must be a {'non-negative' if allow_zero else 'positive'} number.")
    return value


def get_model_configuration() -> ModelConfiguration:
    """Return validated AI configuration without exposing secret values."""
    api_key = _environment_value("GROQ_API_KEY")
    model = _environment_value("GROQ_MODEL")
    missing = [
        name
        for name, value in (("GROQ_API_KEY", api_key), ("GROQ_MODEL", model))
        if value is None
    ]
    if missing:
        variables = ", ".join(missing)
        raise ModelConfigurationError(
            f"Missing required AI configuration: {variables}. "
            "Set the values in the environment or .env file using .env.example; the application will not start."
        )
    return ModelConfiguration(
        api_key=api_key,
        model=model,
        timeout_seconds=float(_positive_number("GROQ_TIMEOUT_SECONDS", 30.0, float)),
        retry_attempts=int(_positive_number("GROQ_RETRY_ATTEMPTS", 2, int, allow_zero=True)),
    )
