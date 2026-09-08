"""Search-and-answer agent for the existing retrieval subgraph."""

import json
import re
import uuid
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from src.agents.template_design import primary_and_fallback_models
from src.state import GraphState
from src.tools.keyword_search import course_ids_in_query


def _guide_plan(state: GraphState) -> dict[str, Any]:
    for message in reversed(state.get("guide_agent_state_memory", [])):
        if isinstance(message, AIMessage):
            try:
                return json.loads(str(message.content))
            except json.JSONDecodeError:
                break
    return {"intent": "intent_5", "course_ids": [], "use_keyword_search": True, "needs_personal_data": False}


def _tool_messages(state: GraphState, name: str | None = None) -> list[ToolMessage]:
    messages = [message for message in state.get("search_agent_state_memory", []) if isinstance(message, ToolMessage)]
    return [message for message in messages if name is None or message.name == name]


def _keywords_for_query(query: str) -> list[str]:
    text = query.lower()
    mappings = {
        "python": ("python", "ไพธอน"), "data science": ("data science", "data analysis", "วิเคราะห์ข้อมูล", "วิทยาการข้อมูล"),
        "machine learning": ("machine learning", "แมชชีนเลิร์นนิง", " ml"), "generative ai": ("generative ai", "gen ai", "llm", "rag", "prompt engineering"),
        "artificial intelligence": (" ai", "ปัญญาประดิษฐ์", "artificial intelligence"), "computer science": ("computer science", "วิทยาการคอมพิวเตอร์"),
        "calculus": ("calculus", "แคลคูลัส"), "physics": ("physics", "ฟิสิกส์", "กลศาสตร์"), "biology": ("biology", "ชีววิทยา"),
        "english literature": ("english literature", "วรรณกรรมอังกฤษ"), "เสาร์": ("เสาร์", "saturday"),
    }
    keywords = [key for key, terms in mappings.items() if any(term in text for term in terms)]
    keywords.extend(word for word in re.findall(r"[A-Za-z]{3,}", text) if word not in {"course", "please", "with", "what", "which", "about"})
    return list(dict.fromkeys(keywords))[:6] or [query]


def _next_tool_call(state: GraphState, plan: dict[str, Any]) -> dict[str, Any] | None:
    query = state.get("query", "")
    completed = {message.name for message in _tool_messages(state)}
    looked_up_ids = {str(message.artifact.get("course_id", "")).upper() for message in _tool_messages(state, "course_id") if isinstance(message.artifact, dict)}
    for course_id in plan.get("course_ids", course_ids_in_query(query)):
        if course_id.upper() not in looked_up_ids:
            return {"name": "course_id", "args": {"course_id": course_id}}
    if plan.get("needs_personal_data") and "personal_data" not in completed:
        return {"name": "personal_data", "args": {}}
    if plan.get("use_keyword_search") and "keyword_search" not in completed:
        return {"name": "keyword_search", "args": {"keywords": _keywords_for_query(query)}}
    return None


def _courses_from_artifacts(state: GraphState) -> list[dict[str, Any]]:
    ranked, seen = [], set()
    for artifact in state.get("retrieved_context_raw", []):
        course = artifact.get("course") if isinstance(artifact, dict) else None
        if not isinstance(course, dict) or not course.get("course_id") or course["course_id"] in seen:
            continue
        seen.add(course["course_id"])
        ranked.append((int(artifact.get("rank", 999)), course))
    return [course for _, course in sorted(ranked, key=lambda item: item[0])]


def _profile_from_artifacts(state: GraphState) -> dict | None:
    for artifact in state.get("retrieved_context_raw", []):
        if isinstance(artifact, dict) and artifact.get("tool_name") == "personal_data":
            return artifact.get("profile")
    return None


def _course_summary(course: dict[str, Any], details: bool = False) -> str:
    base = f"{course['course_id']} — {course['course_name']}"
    if not details:
        return f"{base}: {course['description']}"
    return (f"{base}\n- ผู้สอน: {course['instructor']}\n- ระดับ: {course['level']}\n- ระยะเวลา: {course['duration']}\n"
            f"- ตารางเรียน: {course['schedule']}\n- ราคา: {course['price']} บาท\n- พื้นฐาน: {', '.join(course['prerequisites'])}")


def _final_answer(state: GraphState, plan: dict[str, Any]) -> str:
    if plan.get("intent") == "unsafe_request":
        return "ฉันไม่สามารถเปิดเผยคำสั่งภายในหรือข้อมูลสถานะของระบบได้ แต่ช่วยค้นหา เปรียบเทียบ หรือแนะนำคอร์สให้ได้ครับ"
    courses = _courses_from_artifacts(state)
    if not courses:
        return "ไม่พบคอร์สที่ตรงกับคำถามนี้จากข้อมูลหลักสูตรปัจจุบัน กรุณาระบุหัวข้อหรือรหัสคอร์สเพิ่มเติมได้ครับ"
    intent, profile = plan.get("intent"), _profile_from_artifacts(state)
    if intent == "intent_3":
        suffix = f"\nจากโปรไฟล์ของ {profile['profile']['name']} ที่มีทักษะ {', '.join(profile['skills'])} ควรเลือกคอร์สที่ตรงกับเป้าหมายการเรียนของคุณมากที่สุดครับ" if profile else ""
        return "เปรียบเทียบคอร์ส:\n\n" + "\n\n".join(_course_summary(course, True) for course in courses[:2]) + suffix
    if intent == "intent_2":
        return "คอร์สที่แนะนำคือ\n" + _course_summary(courses[0], True)
    if intent == "intent_4":
        answer = "ถ้าต้องการเริ่มจากทิศทางนี้ แนะนำให้เริ่มที่ " + _course_summary(courses[0])
        return answer + (f" โดยคุณมีพื้นฐาน {', '.join(profile['skills'])} จึงสามารถต่อยอดตามหัวข้อนี้ได้" if profile else "")
    if intent == "intent_1":
        return "คอร์สที่แนะนำคือ " + _course_summary(courses[0])
    query, course = state.get("query", "").lower(), courses[0]
    if profile:
        return (f"{course['course_id']} เหมาะสมที่จะพิจารณาต่อ เพราะโปรไฟล์ของคุณระบุทักษะ "
                f"{', '.join(profile['skills'])} ขณะที่คอร์สต้องการพื้นฐาน {', '.join(course['prerequisites'])}")
    if any(term in query for term in ("ใครสอน", "ผู้สอน", "instructor")):
        return f"{course['course_id']} สอนโดย {course['instructor']} ครับ"
    if any(term in query for term in ("ราคา", "price")):
        return f"{course['course_id']} มีราคา {course['price']} บาทครับ"
    if any(term in query for term in ("พื้นฐาน", "prerequisite", "ต้องมี")):
        return f"พื้นฐานสำหรับ {course['course_id']}: {', '.join(course['prerequisites'])}"
    if any(term in query for term in ("ตาราง", "วัน", "schedule", "เสาร์")):
        return f"{course['course_id']} เรียน {course['schedule']} ครับ"
    return _course_summary(course, True)


def search_agent(state: GraphState) -> dict:
    """Call one needed tool per turn, then return a grounded final answer."""
    plan = _guide_plan(state)
    if state.get("search_attempts", 0) >= state.get("max_search_attempts", 4):
        return {"search_agent_state_memory": [AIMessage(content="Search limit reached.")], "final_answer": _final_answer(state, plan)}
    tool_call = _next_tool_call(state, plan)
    if tool_call:
        return {"search_agent_state_memory": [AIMessage(content="Retrieving course information.", tool_calls=[{
            "id": str(uuid.uuid4()), "name": tool_call["name"], "args": tool_call["args"], "type": "tool_call",
        }])]}
    primary_model, fallback_model = primary_and_fallback_models()
    note = "" if primary_model or fallback_model else " Deterministic local fallback used."
    return {"search_agent_state_memory": [AIMessage(content=f"Retrieved evidence is sufficient.{note}")], "final_answer": _final_answer(state, plan)}
