"""Deterministic mock personal-data retrieval for the POC."""

import json
from pathlib import Path

from langchain_core.tools import tool


DATA_PATH = Path(__file__).resolve().parents[2] / "local_data" / "personal_data.json"


def load_personal_data() -> dict:
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


@tool("personal_data", response_format="content_and_artifact")
def personal_data_tool() -> tuple[str, dict]:
    """Retrieve the mock learner profile for personalized course enquiries."""
    profile = load_personal_data()
    return json.dumps(profile, ensure_ascii=False), profile
