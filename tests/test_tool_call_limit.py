import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage

from src.graph import _tool_result, should_continue, tool_call_limit_error_node
from src.main import run_chatbot


def tool_message(name: str, args: dict) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "call-1"}])


class ToolCallLimitTests(unittest.TestCase):
    def test_tool_execution_records_count_and_input_in_artifact(self) -> None:
        result = _tool_result(
            {"search_agent_state_memory": [tool_message("course_id", {"course_id": "CS101"})], "tool_call_count": 0, "max_tool_calls": 5},
            "course_id",
            {"found": True},
            "course data",
        )

        self.assertEqual(result["tool_call_count"], 1)
        self.assertEqual(result["tool_call_artifacts"], [{"sequence": 1, "tool_name": "course_id", "tool_input": {"course_id": "CS101"}, "executed": True, "max_calls": 5}])

    def test_sixth_requested_tool_call_routes_to_forced_error_artifact(self) -> None:
        state = {
            "search_agent_state_memory": [tool_message("course_catalog", {})],
            "tool_call_count": 5,
            "max_tool_calls": 5,
            "tool_call_artifacts": [],
        }

        self.assertEqual(should_continue(state), "tool_call_limit_error")
        result = tool_call_limit_error_node(state)
        self.assertIn("attempted call 6", result["tool_call_limit_error"])
        self.assertTrue(result["tool_call_artifacts"][0]["limit_exceeded"])
        self.assertEqual(result["tool_call_artifacts"][0]["tool_name"], "course_catalog")

    def test_personal_data_tool_routes_through_the_same_bounded_loop(self) -> None:
        state = {
            "search_agent_state_memory": [tool_message("personal_data", {})],
            "tool_call_count": 0,
            "max_tool_calls": 5,
        }

        self.assertEqual(should_continue(state), "personal_data_tool")

    @patch("src.main.graph.invoke")
    def test_application_raises_the_recorded_tool_call_limit_error(self, invoke) -> None:
        invoke.return_value = {"tool_call_limit_error": "Tool-call limit exceeded: attempted call 6; maximum is 5."}

        with self.assertRaisesRegex(RuntimeError, "attempted call 6"):
            run_chatbot("ทดสอบ")

        self.assertEqual(invoke.call_args.args[0]["max_tool_calls"], 5)
        self.assertEqual(invoke.call_args.args[0]["tool_call_artifacts"], [])


if __name__ == "__main__":
    unittest.main()
