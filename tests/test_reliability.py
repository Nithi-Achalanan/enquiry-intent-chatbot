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
        self.assertNotIn("GROQ_API_KEY", evaluation.render(results))
