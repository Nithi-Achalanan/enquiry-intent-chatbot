"""Regression checks for the LLM-led course-selection design."""

import unittest

from src.agents.data_retriever import TOOL_SCHEMAS
from src.agents.template_design import GuidePlan
from src.tools.course_catalog import load_course_catalog


class LlmOnlyDesignTests(unittest.TestCase):
    def test_catalogue_contains_course_facts_without_keyword_metadata(self) -> None:
        courses = load_course_catalog()

        self.assertTrue(courses)
        self.assertTrue(all("keywords" not in course for course in courses))

    def test_catalogue_descriptions_are_substantive_retrieval_data(self) -> None:
        courses = load_course_catalog()

        self.assertTrue(all(len(course["description"]) >= 180 for course in courses))

    def test_only_non_ranking_retrieval_tools_are_exposed(self) -> None:
        tool_names = {schema["function"]["name"] for schema in TOOL_SCHEMAS}

        self.assertEqual(tool_names, {"course_catalog", "course_id"})

    def test_guide_intent_is_not_limited_to_predefined_labels(self) -> None:
        plan = GuidePlan.model_validate({
            "intent": "ขอข้อมูลคอร์สเพื่อวางแผนเรียน",
            "intent_label": "ข้อมูลคอร์ส",
            "decision_summary": "ผู้ใช้ต้องการข้อมูลเพื่อวางแผน",
            "answer_instruction": "ตอบจากข้อมูลคอร์สที่ตรวจสอบได้",
            "answer_template": "คำตอบกระชับ",
            "retrieval_direction": "ดึง catalogue หากต้องการข้อมูลคอร์ส",
        })

        self.assertEqual(plan.intent, "ขอข้อมูลคอร์สเพื่อวางแผนเรียน")
