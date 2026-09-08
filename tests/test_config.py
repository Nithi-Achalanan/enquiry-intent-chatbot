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
            if key not in {"GROQ_API_KEY", "GROQ_MODEL"}
        }
        environment.update(values)
        return environment

    def test_missing_api_key_is_rejected(self) -> None:
        with patch.dict(
            os.environ,
            self._environment(GROQ_MODEL="openai/gpt-oss-20b"),
            clear=True,
        ):
            with self.assertRaisesRegex(ModelConfigurationError, "GROQ_API_KEY"):
                get_model_configuration()

    def test_missing_model_is_rejected(self) -> None:
        with patch.dict(
            os.environ,
            self._environment(GROQ_API_KEY="test-key"),
            clear=True,
        ):
            with self.assertRaisesRegex(ModelConfigurationError, "GROQ_MODEL"):
                get_model_configuration()

    def test_model_name_is_trimmed(self) -> None:
        with patch.dict(
            os.environ,
            self._environment(
                GROQ_API_KEY="test-key",
                GROQ_MODEL="  openai/gpt-oss-20b  ",
            ),
            clear=True,
        ):
            configuration = get_model_configuration()

        self.assertEqual(configuration.model, "openai/gpt-oss-20b")


if __name__ == "__main__":
    unittest.main()
