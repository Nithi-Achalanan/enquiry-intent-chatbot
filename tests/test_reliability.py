import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.main import app
from src.reliability import InvocationDiagnostic, ModelInvocationError, invoke_with_retry, is_retryable_model_error
from scripts import run_baseline_evaluation as evaluation


class HttpError(Exception):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class ReliabilityTests(unittest.TestCase):
    def test_compare_retrieval_flag_does_not_require_clarification_options(self) -> None:
        observed = {
            "plan": {"clarification_requires_retrieval": True},
            "final_response_mode": "compare",
            "final_result": {"clarification_options": []},
            "retrieved_context_raw": [],
        }

        self.assertEqual(evaluation._clarification_option_score(observed), evaluation.NA)

    def test_evaluation_fails_missing_required_clarification_options(self) -> None:
        observed = {
            "plan": {"clarification_requires_retrieval": True},
            "final_response_mode": "clarify_with_suggestion",
            "final_result": {"clarification_options": []},
            "retrieved_context_raw": [],
        }

        self.assertEqual(evaluation._clarification_option_score(observed), evaluation.FAIL)

    def test_intent_evolution_allows_recommendation_after_clarification_answer(self) -> None:
        turn = {
            "expected": evaluation.expected(
                ["explore_direction", "recommend_course"],
                ["recommend_one"],
            ),
            "observed": {
                "plan": {"intent_family": "recommend_course"},
                "final_result": {
                    "final_response_mode": "recommend_one",
                    "primary_course_id": "AI101",
                    "referenced_course_ids": ["AI101"],
                    "related_course_ids": ["AI101"],
                    "evidence_course_ids": ["AI101"],
                },
                "final_response_mode": "recommend_one",
                "final_answer": "แนะนำคอร์ส AI101 ครับ",
                "grounding_status": "grounded",
                "retrieved_context_raw": [{
                    "tool_name": "course_id",
                    "course": {"course_id": "AI101"},
                }],
                "related_courses": [{"course_id": "AI101"}],
                "dialogue_state": {},
            },
        }

        self.assertEqual(evaluation.score_turn(turn)["intent_family"], evaluation.PASS)

    def test_recommendation_requires_primary_related_card(self) -> None:
        turn = {
            "expected": evaluation.expected("recommend_course", ["recommend_one"], related="nonempty"),
            "observed": {
                "plan": {"intent_family": "recommend_course"},
                "final_result": {
                    "final_response_mode": "recommend_one",
                    "primary_course_id": "AI201",
                    "referenced_course_ids": ["AI201"],
                    "related_course_ids": ["AI101"],
                    "evidence_course_ids": ["AI101", "AI201"],
                },
                "final_response_mode": "recommend_one",
                "final_answer": "แนะนำ AI201 ครับ",
                "grounding_status": "grounded",
                "retrieved_context_raw": [{"courses": [{"course_id": "AI101"}, {"course_id": "AI201"}]}],
                "related_courses": [{"course_id": "AI101"}],
                "dialogue_state": {},
            },
        }

        self.assertEqual(evaluation.score_turn(turn)["related_courses"], evaluation.FAIL)

    def test_compare_requires_two_compared_course_cards(self) -> None:
        turn = {
            "expected": evaluation.expected("compare_courses", ["compare"], related="nonempty"),
            "observed": {
                "plan": {"intent_family": "compare_courses"},
                "final_result": {
                    "final_response_mode": "compare",
                    "referenced_course_ids": ["AI101", "AI201"],
                    "related_course_ids": ["AI101"],
                    "evidence_course_ids": ["AI101", "AI201"],
                },
                "final_response_mode": "compare",
                "final_answer": "เปรียบเทียบ AI101 กับ AI201 ครับ",
                "grounding_status": "grounded",
                "retrieved_context_raw": [{"courses": [{"course_id": "AI101"}, {"course_id": "AI201"}]}],
                "related_courses": [{"course_id": "AI101"}],
                "dialogue_state": {},
            },
        }

        self.assertEqual(evaluation.score_turn(turn)["related_courses"], evaluation.FAIL)

    def test_no_result_cannot_retain_a_selected_course(self) -> None:
        turn = {
            "expected": evaluation.expected("free_style", ["no_result"], related="empty"),
            "observed": {
                "plan": {"intent_family": "free_style"},
                "final_result": {
                    "final_response_mode": "no_result",
                    "referenced_course_ids": ["AI101"],
                    "related_course_ids": [],
                    "evidence_course_ids": ["AI101"],
                },
                "final_response_mode": "no_result",
                "final_answer": "ยังไม่พบคอร์สครับ",
                "grounding_status": "grounded",
                "retrieved_context_raw": [{"course": {"course_id": "AI101"}}],
                "related_courses": [],
                "dialogue_state": {},
            },
        }

        self.assertEqual(evaluation.score_turn(turn)["grounding"], evaluation.FAIL)

    def test_exact_course_lookup_counts_as_current_turn_grounding(self) -> None:
        observed = {
            "plan": {"clarification_requires_retrieval": False},
            "final_response_mode": "clarify_with_suggestion",
            "final_result": {
                "referenced_course_ids": ["AI101"],
                "clarification_options": [{"label": "AI101", "supporting_course_ids": ["AI101"]}],
            },
            "search_messages": [{"tool_calls": [{"name": "course_id", "args": {"course_id": "AI101"}}]}],
            "retrieved_context_raw": [{"tool_name": "course_id", "course": {"course_id": "AI101"}}],
            "grounding_status": "grounded",
        }

        self.assertEqual(evaluation._clarification_grounding_score(observed), evaluation.PASS)

    @patch("src.main.run_chatbot")
    def test_api_accepts_missing_dialogue_state_for_backwards_compatibility(self, run_chatbot) -> None:
        run_chatbot.return_value = {"answer": "ตอบแล้ว", "related_courses": [], "dialogue_state": {}}

        response = TestClient(app).post("/api/chat", json={"query": "AI301 ราคาเท่าไร"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["dialogue_state"], {
            "resolved_course_ids": [],
            "last_primary_course_id": None,
            "last_related_course_ids": [],
            "active_constraints": {},
            "unresolved_references": [],
            "current_goal": None,
            "pending_clarification": None,
            "clarification_count": 0,
            "last_intent_family": None,
            "last_response_mode": None,
        })

    @patch("src.main.run_chatbot")
    def test_api_passes_hidden_dialogue_state_to_the_graph_adapter(self, run_chatbot) -> None:
        dialogue = {
            "resolved_course_ids": ["AI201"],
            "last_primary_course_id": "AI201",
            "last_related_course_ids": [],
            "active_constraints": {"topic": "Machine Learning"},
            "unresolved_references": [],
            "current_goal": "เรียน Machine Learning",
            "pending_clarification": None,
            "clarification_count": 0,
            "last_intent_family": None,
            "last_response_mode": None,
        }
        run_chatbot.return_value = {"answer": "ตอบแล้ว", "related_courses": [], "dialogue_state": dialogue}

        response = TestClient(app).post("/api/chat", json={"query": "แล้วตัวนี้ล่ะ", "dialogue_state": dialogue})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["dialogue_state"], dialogue)
        passed_state = run_chatbot.call_args.args[2]
        self.assertEqual(passed_state.last_primary_course_id, "AI201")

    def test_retries_rate_limit_then_returns(self) -> None:
        calls = []

        def operation() -> str:
            calls.append(1)
            if len(calls) < 3:
                raise HttpError(429)
            return "ok"

        with patch("src.reliability.time.sleep"), patch("src.reliability.random.uniform", return_value=0):
            self.assertEqual(invoke_with_retry(operation, agent="search_agent", max_retries=2), "ok")

        self.assertEqual(len(calls), 3)

    def test_does_not_retry_non_retryable_error(self) -> None:
        with patch("src.reliability.time.sleep") as sleep:
            with self.assertRaises(ModelInvocationError) as raised:
                invoke_with_retry(lambda: (_ for _ in ()).throw(HttpError(401)), agent="guide_agent", max_retries=2)

        self.assertFalse(raised.exception.diagnostic.retryable)
        self.assertEqual(raised.exception.diagnostic.attempts, 1)
        sleep.assert_not_called()

    def test_classifies_timeout_and_server_error_as_retryable(self) -> None:
        self.assertTrue(is_retryable_model_error(TimeoutError()))
        self.assertTrue(is_retryable_model_error(HttpError(503)))

    @patch("src.main.run_chatbot")
    def test_api_returns_sanitized_503_after_retries_are_exhausted(self, run_chatbot) -> None:
        run_chatbot.side_effect = ModelInvocationError(InvocationDiagnostic(
            agent="search_agent",
            attempts=3,
            retry_count=2,
            retryable=True,
            error_type="APITimeoutError",
            status_code=None,
            request_id=None,
        ))

        response = TestClient(app).post("/api/chat", json={"query": "ทดสอบ"})

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "ระบบ AI กำลังไม่พร้อมใช้งาน กรุณาลองใหม่อีกครั้ง")
        self.assertEqual(response.headers["retry-after"], "2")

    @patch("src.main.run_chatbot")
    def test_api_does_not_label_a_permanent_provider_failure_as_retryable(self, run_chatbot) -> None:
        run_chatbot.side_effect = ModelInvocationError(InvocationDiagnostic(
            agent="guide_agent",
            attempts=1,
            retry_count=0,
            retryable=False,
            error_type="AuthenticationError",
            status_code=401,
            request_id=None,
        ))

        response = TestClient(app).post("/api/chat", json={"query": "ทดสอบ"})

        self.assertEqual(response.status_code, 502)
        self.assertNotIn("AuthenticationError", response.text)

    @patch.object(evaluation, "SCENARIOS", [("01", "failure", "diagnostic", ["ทดสอบ"])])
    @patch.object(evaluation, "invoke")
    def test_evaluation_keeps_a_safe_retry_diagnostic(self, invoke) -> None:
        invoke.side_effect = ModelInvocationError(InvocationDiagnostic(
            agent="search_agent",
            attempts=3,
            retry_count=2,
            retryable=True,
            error_type="APITimeoutError",
            status_code=None,
            request_id="request-123",
        ))

        results = evaluation.run()
        turn = results[0]["turns"][0]

        self.assertEqual(turn["diagnostic"]["attempts"], 3)
        self.assertNotIn("API_KEY", evaluation.render(results))
