"""Tests for required model configuration."""

import os
import unittest
from unittest.mock import patch

from src.config import ModelConfigurationError, get_model_configuration


class ModelConfigurationTests(unittest.TestCase):
    def _environment(self, **values: str) -> dict[str, str]:
        environment = {
            key: value
            for key, value in os.environ.items()
            if key not in {"OPENAI_API_KEY", "PRIMARY_MODEL", "FALLBACK_MODEL"}
        }
        environment.update(values)
        return environment

    def test_missing_api_key_is_rejected(self) -> None:
        with patch.dict(
            os.environ,
            self._environment(PRIMARY_MODEL="gpt-4o-mini"),
            clear=True,
        ):
            with self.assertRaisesRegex(ModelConfigurationError, "OPENAI_API_KEY"):
                get_model_configuration()

    def test_missing_primary_model_is_rejected(self) -> None:
        with patch.dict(
            os.environ,
            self._environment(OPENAI_API_KEY="test-key"),
            clear=True,
        ):
            with self.assertRaisesRegex(ModelConfigurationError, "PRIMARY_MODEL"):
                get_model_configuration()

    def test_model_names_are_trimmed_and_blank_fallback_is_optional(self) -> None:
        with patch.dict(
            os.environ,
            self._environment(
                OPENAI_API_KEY="test-key",
                PRIMARY_MODEL="  gpt-4o-mini  ",
                FALLBACK_MODEL="   ",
            ),
            clear=True,
        ):
            configuration = get_model_configuration()

        self.assertEqual(configuration.primary_model, "gpt-4o-mini")
        self.assertIsNone(configuration.fallback_model)

    def test_fallback_model_is_trimmed_when_supplied(self) -> None:
        with patch.dict(
            os.environ,
            self._environment(
                OPENAI_API_KEY="test-key",
                PRIMARY_MODEL="gpt-4o-mini",
                FALLBACK_MODEL="  gpt-4.1-mini  ",
            ),
            clear=True,
        ):
            configuration = get_model_configuration()

        self.assertEqual(configuration.primary_model, "gpt-4o-mini")
        self.assertEqual(configuration.fallback_model, "gpt-4.1-mini")


if __name__ == "__main__":
    unittest.main()
