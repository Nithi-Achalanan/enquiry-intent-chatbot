"""Regression checks for the LLM-led enquiry and evidence contracts."""

import inspect
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage
from pydantic import ValidationError

from src.agents import template_design
from src.agents.data_retriever import (
    FinalAnswerResult,
    FinalResponseMode,
    SemanticGroundingResult,
    TOOL_SCHEMAS,
    _finalize,
    _parse_contract_response,
    validate_final_result,
)
from src.agents.template_design import GuidePlan, IntentFamily, PlannedResponseMode
from src.graph import course_catalog_node
from src.main import extract_related_courses
from src.state import DialogueState, merge_dialogue_state
from src.tools.course_catalog import load_course_catalog
from src.tools.personal_data import load_personal_data


def guide_plan(**overrides):
    value = {
        "semantic_intent": "ผู้ใช้ต้องการคำตอบเชิงความหมายที่ยืดหยุ่น",
        "intent_family": "recommend_course",
        "planned_response_mode": "recommend_one",
        "decision_summary": "เลือกหนึ่งคอร์สตามเป้าหมาย",
        "answer_instruction": "ตอบจากหลักฐานที่ค้นได้",
        "answer_template": "ชื่อคอร์สและเหตุผลสั้น ๆ",
        "retrieval_direction": "ค้น catalogue ก่อนเลือก",
    }
    value.update(overrides)
    return value


def tool_message(name="course_catalog", args=None):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args or {}, "id": "call-1"}])


class LlmOnlyDesignTests(unittest.TestCase):
    def test_catalogue_contains_substantive_facts_without_keyword_metadata(self):
        courses = load_course_catalog()
        self.assertTrue(courses)
        self.assertTrue(all("keywords" not in course for course in courses))
        self.assertTrue(all(len(course["description"]) >= 180 for course in courses))

    def test_tool_contract_has_catalogue_exact_id_and_personal_data(self):
        names = {schema["function"]["name"] for schema in TOOL_SCHEMAS}
        self.assertEqual(names, {"course_catalog", "course_id", "personal_data"})

    def test_intent_family_is_fixed_but_semantic_intent_is_free_text(self):
        self.assertEqual(
            {item.value for item in IntentFamily},
            {"recommend_course", "recommend_with_details", "compare_courses", "explore_direction", "free_style"},
        )
        plan = GuidePlan.model_validate(guide_plan(semantic_intent="ความต้องการเฉพาะที่ไม่ใช่ label ตายตัว"))
        self.assertEqual(plan.semantic_intent, "ความต้องการเฉพาะที่ไม่ใช่ label ตายตัว")
        with self.assertRaises(ValidationError):
            GuidePlan.model_validate(guide_plan(intent_family="personal_data"))

    def test_planned_mode_contract_is_enforced(self):
        self.assertEqual(
            {item.value for item in PlannedResponseMode},
            {"recommend_one", "recommend_one_with_details", "compare", "course_info", "explore", "clarify", "clarify_with_suggestion", "refuse"},
        )
        with self.assertRaises(ValidationError):
            GuidePlan.model_validate(guide_plan(planned_response_mode="compare"))

    def test_no_python_semantic_classifier_is_present(self):
        source = inspect.getsource(template_design)
        self.assertNotIn("import re", source)
        self.assertNotIn("keyword_to_intent", source)
        self.assertNotIn("classify_intent", source)
        self.assertIn("structured_llm.invoke", source)

    def test_optional_examples_can_be_absent(self):
        missing = Path(tempfile.gettempdir()) / "missing-intent-examples-for-test.json"
        with patch.dict(os.environ, {"GUIDE_EXAMPLE_SET_PATH": str(missing)}):
            self.assertEqual(template_design.load_intent_examples(), [])

    def test_dialogue_constraints_accumulate_and_new_value_replaces_old(self):
        prior = DialogueState(active_constraints={"topic": "Data", "level": "beginner"})
        updated = merge_dialogue_state(prior, active_constraints={"duration": "short"})
        replaced = merge_dialogue_state(updated, active_constraints={"level": "intermediate"})
        self.assertEqual(replaced.active_constraints, {"topic": "Data", "level": "intermediate", "duration": "short"})

    def test_reference_state_preserves_primary_and_related_order(self):
        state = merge_dialogue_state(None, primary_course_id="ai201", related_course_ids=["AI101", "AI301"])
        self.assertEqual(state.last_primary_course_id, "AI201")
        self.assertEqual(state.last_related_course_ids, ["AI101", "AI301"])
        self.assertEqual(state.resolved_course_ids, ["AI201", "AI101", "AI301"])

    def test_catalogue_is_one_shared_evidence_artifact(self):
        state = {"search_agent_state_memory": [tool_message()], "retrieved_context_raw": [], "tool_call_count": 0, "max_tool_calls": 5}
        result = course_catalog_node(state)
        self.assertEqual(len(result["retrieved_context_raw"]), 1)
        artifact = result["retrieved_context_raw"][0]
        self.assertEqual(artifact["tool_name"], "course_catalog")
        self.assertEqual(len(artifact["courses"]), len(load_course_catalog()))

    def test_related_courses_use_only_selected_evidence_backed_ids(self):
        courses = load_course_catalog()
        evidence = [{"tool_name": "course_catalog", "courses": courses}]
        final = {"final_response_mode": "course_info", "related_course_ids": ["AI201", "MADEUP", "AI201"]}
        related = extract_related_courses(final, evidence)
        self.assertEqual([course["course_id"] for course in related], ["AI201"])
        self.assertEqual(extract_related_courses({**final, "final_response_mode": "no_result"}, evidence), [])

    def test_final_validation_rejects_unsupported_ids_and_compare_downgrade(self):
        evidence = [{"tool_name": "course_id", "course": load_course_catalog()[0]}]
        result, issues = validate_final_result(
            FinalAnswerResult(final_response_mode="compare", answer="เปรียบเทียบ", referenced_course_ids=["CS101", "FAKE"]),
            evidence,
            "compare",
        )
        self.assertEqual(result.final_response_mode, FinalResponseMode.NO_RESULT)
        self.assertEqual(result.referenced_course_ids, ["CS101"])
        self.assertTrue(issues)

    def test_grounding_correction_runs_once(self):
        first = FinalAnswerResult(final_response_mode="course_info", answer="unsupported", referenced_course_ids=["CS101"])
        corrected = FinalAnswerResult(final_response_mode="course_info", answer="แก้แล้ว", referenced_course_ids=["CS101"])
        evidence = [{"tool_name": "course_id", "course": load_course_catalog()[0]}]
        with patch("src.agents.data_retriever._structured_final", side_effect=[first, corrected]) as finalizer, patch(
            "src.agents.data_retriever._semantic_grounding",
            side_effect=[SemanticGroundingResult(grounded=False, unsupported_claims=["unsupported"]), SemanticGroundingResult(grounded=True)],
        ) as verifier:
            result = _finalize({"retrieved_context_raw": evidence}, {"planned_response_mode": "course_info"}, "candidate")
        self.assertEqual(finalizer.call_count, 2)
        self.assertEqual(verifier.call_count, 2)
        self.assertEqual(result["grounding_status"], "corrected")
        self.assertEqual(result["final_answer"], "แก้แล้ว")

    def test_provider_prefixed_structured_tool_name_still_parses(self):
        raw = AIMessage(content="", tool_calls=[{
            "name": "functions.FinalAnswerResult",
            "args": {"final_response_mode": "course_info", "answer": "คำตอบ"},
            "id": "call-structured",
        }])
        parsed = _parse_contract_response({"raw": raw, "parsed": None, "parsing_error": KeyError()}, FinalAnswerResult)
        self.assertEqual(parsed.final_response_mode, FinalResponseMode.COURSE_INFO)
        self.assertEqual(parsed.answer, "คำตอบ")

    def test_personal_data_tool_reads_authoritative_mock_profile(self):
        profile = load_personal_data()
        self.assertEqual(profile["user_id"], "USER-001")
        self.assertIn("Python", profile["skills"])
        self.assertEqual(profile["preferences"]["budget"], 5000)


if __name__ == "__main__":
    unittest.main()
