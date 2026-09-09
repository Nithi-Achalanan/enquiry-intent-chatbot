"""Contract tests for multi-turn clarification tracking and grounded options."""

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from src.agents.data_retriever import (
    ClarificationOption,
    FinalAnswerResult,
    FinalResponseMode,
    SemanticGroundingResult,
    _finalize,
    search_agent,
    validate_final_result,
)
from src.agents.template_design import GuidePlan, template_agent
from src.main import app
from src.state import DialogueState, PendingClarification, merge_dialogue_state
from src.tools.course_catalog import load_course_catalog


def guide_plan(**overrides):
    value = {
        "semantic_intent": "ต้องการหาแนวทางเรียนที่เหมาะสม",
        "intent_family": "explore_direction",
        "planned_response_mode": "clarify_with_suggestion",
        "decision_summary": "ต้องทราบแนวทางที่สนใจ",
        "answer_instruction": "ค้นหลักสูตรจริงก่อนเสนอทางเลือก",
        "answer_template": "เสนอข้อมูลที่ยืนยันได้แล้วถามหนึ่งคำถาม",
        "retrieval_direction": "เรียก course_catalog ก่อนสร้างตัวเลือก",
        "clarification_needed": True,
        "clarification_strategy": "ask_optional",
        "clarification_target": "learning_direction",
        "clarification_requires_retrieval": True,
        "clarification_option_goal": "หาสองแนวทางที่มีหลักสูตรรองรับจริง",
    }
    value.update(overrides)
    return value


def catalogue_evidence():
    return [{"tool_name": "course_catalog", "courses": load_course_catalog()}]


class ClarificationStateTests(unittest.TestCase):
    def test_pending_clarification_defaults_are_safe_and_independent(self):
        first = PendingClarification(target="learning_goal", question="เป้าหมายคืออะไร?")
        second = PendingClarification(target="experience_level", question="มีพื้นฐานแค่ไหน?")

        first.options.append("งานปัจจุบัน")

        self.assertEqual(first.supporting_course_ids, [])
        self.assertFalse(first.retrieval_required)
        self.assertEqual(first.attempt, 1)
        self.assertEqual(second.options, [])

    def test_dialogue_state_new_fields_have_backwards_compatible_defaults(self):
        state = DialogueState.model_validate({"current_goal": "learn AI"})

        self.assertIsNone(state.pending_clarification)
        self.assertEqual(state.clarification_count, 0)
        self.assertIsNone(state.last_intent_family)
        self.assertIsNone(state.last_response_mode)

    def test_merge_can_create_partially_update_and_clear_pending_state(self):
        created = merge_dialogue_state(
            DialogueState(active_constraints={"level": "beginner"}),
            pending_clarification={
                "target": "learning_direction",
                "question": "สนใจแนวไหน?",
                "options": ["Machine Learning"],
                "supporting_course_ids": ["ai201"],
                "retrieval_required": True,
            },
            clarification_asked=True,
            last_intent_family="explore_direction",
            last_response_mode="clarify_with_suggestion",
        )
        partial = merge_dialogue_state(
            created,
            active_constraints={"preferred_use": "automation"},
            pending_clarification={
                "target": "learning_direction",
                "reason": "need one remaining preference",
                "question": "อยากนำไปใช้กับงานแบบใด?",
                "retrieval_required": False,
                "attempt": 2,
            },
            clarification_asked=True,
        )
        cleared = merge_dialogue_state(partial, clear_pending_clarification=True)

        self.assertEqual(created.pending_clarification.supporting_course_ids, ["AI201"])
        self.assertEqual(created.clarification_count, 1)
        self.assertEqual(partial.pending_clarification.attempt, 2)
        self.assertEqual(partial.clarification_count, 2)
        self.assertEqual(partial.active_constraints, {"level": "beginner", "preferred_use": "automation"})
        self.assertIsNone(cleared.pending_clarification)
        self.assertEqual(cleared.clarification_count, 2)

    def test_merge_changes_count_only_for_a_real_question_and_can_reset(self):
        prior = DialogueState(
            clarification_count=2,
            pending_clarification=PendingClarification(
                target="learning_goal",
                question="เป้าหมายคืออะไร?",
                attempt=2,
            ),
        )

        unchanged = merge_dialogue_state(prior, active_constraints={"level": "beginner"})
        reset = merge_dialogue_state(
            unchanged,
            clear_pending_clarification=True,
            reset_clarification_count=True,
        )

        self.assertEqual(unchanged.clarification_count, 2)
        self.assertEqual(reset.clarification_count, 0)


class GuideClarificationContractTests(unittest.TestCase):
    def test_topic_change_does_not_inherit_exhausted_same_target_limit(self):
        prior = DialogueState(
            pending_clarification=PendingClarification(
                target="learning_goal",
                question="เป้าหมายเดิมคืออะไร?",
                attempt=2,
            ),
            clarification_count=2,
        )
        changed_topic_plan = GuidePlan.model_validate(
            guide_plan(
                planned_response_mode="clarify",
                clarification_strategy="ask_required",
                clarification_question="เป้าหมายของหัวข้อใหม่คืออะไร?",
                clarification_requires_retrieval=False,
                clarification_option_goal=None,
                pending_clarification_resolution="topic_changed",
            )
        ).model_dump(mode="json")
        with patch("src.agents.template_design._model_plan", return_value=changed_topic_plan):
            result = template_agent({
                "query": "ถามหัวข้อใหม่",
                "conversation": [],
                "dialogue_state": prior.model_dump(mode="json"),
            })

        self.assertFalse(result["guide_plan"]["clarification_limit_reached"])
        self.assertEqual(result["guide_plan"]["planned_response_mode"], "clarify")
        self.assertIsNone(result["dialogue_state"]["pending_clarification"])
        self.assertEqual(result["dialogue_state"]["clarification_count"], 0)

    def test_retrieval_backed_plan_can_omit_user_facing_question(self):
        plan = GuidePlan.model_validate(guide_plan())

        self.assertIsNone(plan.clarification_question)
        self.assertTrue(plan.clarification_requires_retrieval)
        self.assertEqual(plan.clarification_target, "learning_direction")
        self.assertTrue(plan.clarification_option_goal)

    def test_generic_clarification_still_requires_a_question(self):
        with self.assertRaises(ValidationError):
            GuidePlan.model_validate(
                guide_plan(
                    planned_response_mode="clarify",
                    clarification_strategy="ask_required",
                    clarification_requires_retrieval=False,
                    clarification_option_goal=None,
                )
            )

    def test_pending_resolution_values_and_limit_flag_round_trip(self):
        for resolution in ("none", "answered", "partially_answered", "rejected", "topic_changed"):
            plan = GuidePlan.model_validate(
                guide_plan(
                    pending_clarification_resolution=resolution,
                    clarification_limit_reached=True,
                )
            )
            self.assertEqual(plan.pending_clarification_resolution, resolution)
            self.assertTrue(plan.clarification_limit_reached)

        with self.assertRaises(ValidationError):
            GuidePlan.model_validate(guide_plan(pending_clarification_resolution="guessed"))


class GroundedClarificationTests(unittest.TestCase):
    def test_final_option_contract_has_independent_default_ids(self):
        first = ClarificationOption(label="Machine Learning")
        second = ClarificationOption(label="Generative AI")

        first.supporting_course_ids.append("AI201")

        self.assertEqual(second.supporting_course_ids, [])

    def test_validation_normalizes_ids_and_removes_unsupported_options(self):
        courses = load_course_catalog()
        supported_id = courses[0]["course_id"]
        final, issues = validate_final_result(
            FinalAnswerResult(
                final_response_mode="clarify_with_suggestion",
                answer="มีตัวเลือกจากหลักสูตรที่ค้นพบ แล้วคุณสนใจแนวไหน?",
                clarification_target="learning_direction",
                clarification_question="คุณสนใจแนวไหน?",
                clarification_options=[
                    ClarificationOption(label="supported", supporting_course_ids=[supported_id.lower()]),
                    ClarificationOption(label="unsupported", supporting_course_ids=["madeup"]),
                ],
            ),
            catalogue_evidence(),
            "clarify_with_suggestion",
            clarification_requires_retrieval=True,
        )

        flattened = [course_id for option in final.clarification_options for course_id in option.supporting_course_ids]
        self.assertEqual(flattened, [supported_id])
        self.assertNotIn("unsupported", [option.label for option in final.clarification_options])
        self.assertTrue(any("MADEUP" in issue for issue in issues))

    def test_only_unsupported_options_are_not_left_displayable(self):
        final, issues = validate_final_result(
            FinalAnswerResult(
                final_response_mode="clarify_with_suggestion",
                answer="กรุณาบอกแนวทางที่สนใจเพิ่มเติม",
                clarification_target="learning_direction",
                clarification_question="open question",
                clarification_options=[
                    ClarificationOption(label="invented", supporting_course_ids=["FAKE101"]),
                ],
            ),
            catalogue_evidence(),
            "clarify_with_suggestion",
            clarification_requires_retrieval=True,
        )

        self.assertEqual(final.clarification_options, [])
        self.assertTrue(issues)

    def test_retrieval_required_plan_forces_catalogue_before_final_answer(self):
        plan = GuidePlan.model_validate(guide_plan()).model_dump(mode="json")
        state = {
            "query": "ยังไม่รู้ว่าจะเรียนทางไหน",
            "guide_plan": plan,
            "dialogue_state": {},
            "conversation": [],
            "search_agent_state_memory": [],
            "retrieved_context_raw": [],
        }
        with patch("src.agents.data_retriever._get_models") as models:
            tool_model = models.return_value[1]
            tool_model.invoke.return_value = AIMessage(content="ควรตอบทันที")
            result = search_agent(state)

        calls = result["search_agent_state_memory"][0].tool_calls
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "course_catalog")

    def test_finalize_builds_pending_state_from_actual_user_facing_output(self):
        supported_id = load_course_catalog()[0]["course_id"]
        actual = FinalAnswerResult(
            final_response_mode="clarify_with_suggestion",
            answer="ข้อมูลจากหลักสูตรจริง พร้อมคำถามสุดท้าย",
            related_course_ids=[supported_id],
            clarification_target="learning_direction",
            clarification_question="คำถามที่ผู้ใช้เห็นจริง?",
            clarification_options=[
                ClarificationOption(label="เส้นทางที่รองรับ", supporting_course_ids=[supported_id]),
            ],
        )
        prior = DialogueState(
            clarification_count=1,
            pending_clarification=PendingClarification(
                target="learning_direction",
                question="คำถามเดิม?",
                attempt=1,
            ),
        )
        with patch("src.agents.data_retriever._structured_final", return_value=actual), patch(
            "src.agents.data_retriever._semantic_grounding",
            return_value=SemanticGroundingResult(grounded=True),
        ):
            result = _finalize(
                {"retrieved_context_raw": catalogue_evidence(), "dialogue_state": prior.model_dump()},
                GuidePlan.model_validate(guide_plan()).model_dump(mode="json"),
                "draft question",
            )

        pending = result["dialogue_state"]["pending_clarification"]
        self.assertEqual(pending["question"], actual.clarification_question)
        self.assertEqual(pending["options"], ["เส้นทางที่รองรับ"])
        self.assertEqual(pending["supporting_course_ids"], [supported_id])
        self.assertEqual(pending["attempt"], 2)
        self.assertEqual(result["dialogue_state"]["clarification_count"], 2)

    def test_same_target_clarification_is_bounded_after_two_attempts(self):
        actual = FinalAnswerResult(
            final_response_mode="clarify",
            answer="ขอถามซ้ำอีกครั้ง",
            clarification_target="learning_goal",
            clarification_question="เป้าหมายคืออะไร?",
        )
        prior = DialogueState(
            clarification_count=2,
            pending_clarification=PendingClarification(
                target="learning_goal",
                question="เป้าหมายคืออะไร?",
                attempt=2,
            ),
        )
        generic_plan = GuidePlan.model_validate(
            guide_plan(
                planned_response_mode="clarify",
                clarification_strategy="ask_required",
                clarification_question="เป้าหมายคืออะไร?",
                clarification_target="learning_goal",
                clarification_requires_retrieval=False,
                clarification_option_goal=None,
            )
        ).model_dump(mode="json")
        with patch("src.agents.data_retriever._structured_final", return_value=actual), patch(
            "src.agents.data_retriever._semantic_grounding",
            return_value=SemanticGroundingResult(grounded=True),
        ):
            result = _finalize(
                {"retrieved_context_raw": [], "dialogue_state": prior.model_dump()},
                generic_plan,
                "draft question",
            )

        self.assertEqual(result["final_response_mode"], FinalResponseMode.NO_RESULT)
        self.assertIsNone(result["dialogue_state"]["pending_clarification"])
        self.assertEqual(result["dialogue_state"]["clarification_count"], 0)
        self.assertTrue(any("maximum" in issue for issue in result["grounding_issues"]))

    def test_exact_repeated_question_is_corrected_then_replaced_with_safe_open_question(self):
        repeated = FinalAnswerResult(
            final_response_mode="clarify",
            answer="เป้าหมายคืออะไร?",
            clarification_target="learning_goal",
            clarification_question="  เป้าหมายคืออะไร?  ",
        )
        prior = DialogueState(
            clarification_count=1,
            pending_clarification=PendingClarification(
                target="learning_goal",
                question="เป้าหมายคืออะไร?",
                attempt=1,
            ),
        )
        generic_plan = GuidePlan.model_validate(
            guide_plan(
                planned_response_mode="clarify",
                clarification_strategy="ask_required",
                clarification_question="เป้าหมายคืออะไร?",
                clarification_target="learning_goal",
                clarification_requires_retrieval=False,
                clarification_option_goal=None,
                pending_clarification_resolution="rejected",
            )
        ).model_dump(mode="json")
        with patch("src.agents.data_retriever._structured_final", return_value=repeated) as finalizer, patch(
            "src.agents.data_retriever._semantic_grounding",
            return_value=SemanticGroundingResult(grounded=True),
        ):
            result = _finalize(
                {"retrieved_context_raw": [], "dialogue_state": prior.model_dump()},
                generic_plan,
                "draft question",
            )

        self.assertEqual(finalizer.call_count, 2)
        self.assertEqual(result["final_response_mode"], FinalResponseMode.CLARIFY)
        self.assertNotEqual(
            result["final_result"]["clarification_question"].strip(),
            prior.pending_clarification.question,
        )
        self.assertEqual(result["dialogue_state"]["pending_clarification"]["attempt"], 2)
        self.assertTrue(any("repeats" in issue for issue in result["grounding_issues"]))


class ClarificationApiTests(unittest.TestCase):
    def test_partial_graph_state_preserves_unrelated_previous_fields(self):
        pending = PendingClarification(
            target="learning_goal",
            question="เป้าหมายของคุณคืออะไร?",
        )
        previous = DialogueState(
            resolved_course_ids=["AI201"],
            active_constraints={"topic": "AI"},
            pending_clarification=pending,
            clarification_count=1,
        )
        graph_result = {
            "final_answer": "ตอบแล้ว",
            "final_result": {"final_response_mode": "course_info", "answer": "ตอบแล้ว"},
            "retrieved_context_raw": [],
            "dialogue_state": {"pending_clarification": None, "clarification_count": 0},
        }
        with patch("src.main.graph.invoke", return_value=graph_result):
            response = TestClient(app).post(
                "/api/chat",
                json={"query": "ตอบ", "dialogue_state": previous.model_dump(mode="json")},
            )

        state = response.json()["dialogue_state"]
        self.assertEqual(state["resolved_course_ids"], ["AI201"])
        self.assertEqual(state["active_constraints"], {"topic": "AI"})
        self.assertIsNone(state["pending_clarification"])
        self.assertEqual(state["clarification_count"], 0)

    def test_api_accepts_and_returns_new_dialogue_fields(self):
        pending = {
            "target": "learning_goal",
            "question": "เป้าหมายของคุณคืออะไร?",
            "options": [],
            "supporting_course_ids": [],
            "retrieval_required": False,
            "attempt": 1,
        }
        graph_result = {
            "final_answer": "ตอบแล้ว",
            "final_result": {"final_response_mode": "course_info", "answer": "ตอบแล้ว"},
            "retrieved_context_raw": [],
            "dialogue_state": {
                "pending_clarification": None,
                "clarification_count": 0,
                "last_intent_family": "free_style",
                "last_response_mode": "course_info",
            },
        }
        with patch("src.main.graph.invoke", return_value=graph_result):
            response = TestClient(app).post(
                "/api/chat",
                json={
                    "query": "คำตอบของผม",
                    "dialogue_state": {
                        "pending_clarification": pending,
                        "clarification_count": 1,
                        "last_intent_family": "explore_direction",
                        "last_response_mode": "clarify",
                    },
                },
            )

        self.assertEqual(response.status_code, 200)
        state = response.json()["dialogue_state"]
        self.assertIsNone(state["pending_clarification"])
        self.assertEqual(state["clarification_count"], 0)
        self.assertEqual(state["last_response_mode"], "course_info")


if __name__ == "__main__":
    unittest.main()
