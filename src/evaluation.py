"""Minimal deterministic evaluation for the course-enquiry POC."""

from collections import defaultdict

from src.agents.template_design import detect_intent
from src.tools.keyword_search import multiple_keyword_search


INTENT_CASES = [
    {"query": "ช่วยแนะนำคอร์ส Python ให้ผมหนึ่งคอร์ส", "expected_intent": "intent_1"},
    {"query": "ช่วยแนะนำคอร์ส Python สำหรับมือใหม่พร้อมรายละเอียด", "expected_intent": "intent_2"},
    {"query": "CS201 กับ CS301 ต่างกันยังไง", "expected_intent": "intent_3"},
    {"query": "อยากเรียน AI แต่ยังไม่รู้ว่าจะเริ่มตรงไหน", "expected_intent": "intent_4"},
    {"query": "AI301 ต้องมีพื้นฐานอะไรบ้าง", "expected_intent": "intent_5"},
]

RETRIEVAL_CASES = [
    {"query": "python beginner", "keywords": ["python", "beginner"], "expected_course_id": "CS201"},
    {"query": "generative ai LLM", "keywords": ["generative ai", "llm"], "expected_course_id": "AI301"},
    {"query": "calculus", "keywords": ["calculus"], "expected_course_id": "MATH201"},
]


def evaluate_intents() -> dict:
    by_intent = defaultdict(lambda: {"total": 0, "correct": 0})
    results = []
    for case in INTENT_CASES:
        actual = detect_intent(case["query"])["intent"]
        correct = actual == case["expected_intent"]
        by_intent[case["expected_intent"]]["total"] += 1
        by_intent[case["expected_intent"]]["correct"] += int(correct)
        results.append({**case, "actual_intent": actual, "correct": correct})
    correct_count = sum(result["correct"] for result in results)
    return {
        "total": len(results), "correct": correct_count, "incorrect": len(results) - correct_count,
        "accuracy_percent": round(correct_count / len(results) * 100, 2),
        "accuracy_by_intent": {
            intent: {**values, "accuracy_percent": round(values["correct"] / values["total"] * 100, 2)}
            for intent, values in by_intent.items()
        },
        "cases": results,
    }


def evaluate_retrieval() -> list[dict]:
    results = []
    for case in RETRIEVAL_CASES:
        ranking = multiple_keyword_search(case["keywords"])
        top_course_id = ranking[0]["course"]["course_id"] if ranking else None
        results.append({**case, "retrieved_ranking": ranking, "actual_top_result": top_course_id, "correct": top_course_id == case["expected_course_id"]})
    return results


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(evaluate_intents())
    print(evaluate_retrieval())
