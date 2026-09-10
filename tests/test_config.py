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
            if key not in {"API_KEY", "MODEL", "OPENAI_TIMEOUT_SECONDS", "OPENAI_RETRY_ATTEMPTS"}
        }
        environment.update(values)
        return environment

    def test_missing_api_key_is_rejected(self) -> None:
        with patch.dict(
            os.environ,
            self._environment(MODEL="gpt-4.1-mini"),
            clear=True,
        ):
            with self.assertRaisesRegex(ModelConfigurationError, "API_KEY"):
                get_model_configuration()

    def test_missing_model_is_rejected(self) -> None:
        with patch.dict(
            os.environ,
            self._environment(API_KEY="test-key"),
            clear=True,
        ):
            with self.assertRaisesRegex(ModelConfigurationError, "MODEL"):
                get_model_configuration()

    def test_model_name_is_trimmed(self) -> None:
        with patch.dict(
            os.environ,
            self._environment(
                API_KEY="test-key",
                MODEL="  gpt-4.1-mini  ",
            ),
            clear=True,
        ):
            configuration = get_model_configuration()

        self.assertEqual(configuration.model, "gpt-4.1-mini")

    def test_timeout_and_retry_settings_are_configurable(self) -> None:
        with patch.dict(
            os.environ,
            self._environment(
                API_KEY="test-key",
                MODEL="gpt-4.1-mini",
                OPENAI_TIMEOUT_SECONDS="12.5",
                OPENAI_RETRY_ATTEMPTS="1",
            ),
            clear=True,
        ):
            configuration = get_model_configuration()

        self.assertEqual(configuration.timeout_seconds, 12.5)
        self.assertEqual(configuration.retry_attempts, 1)


if __name__ == "__main__":
    unittest.main()
