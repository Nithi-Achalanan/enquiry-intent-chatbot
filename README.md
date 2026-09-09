# Enquiry Intent Chatbot

Course-enquiry chatbot powered by the existing LangGraph workflow, with a FastAPI API and a plain HTML/CSS/JavaScript chat interface served by the same process.

## Run locally

1. Create and activate a Python virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env`, then set a non-empty `GROQ_API_KEY` and `GROQ_MODEL`. The application validates these settings at startup and exits with a clear error if either required value is missing.
4. Start the application:

   ```bash
   uvicorn src.main:app --reload
   ```

Open `http://localhost:8000`. The chat interface sends enquiries to `POST /api/chat`; `GET /api/health` reports service availability.

The API delegates enquiry understanding, retrieval, and answering to the existing LangGraph workflow; it does not duplicate agent or tool logic. Enquiry Intent Detection here is more than classification: it combines semantic enquiry understanding, conversation and reference resolution, missing-information detection, the next-best conversational action, retrieval, and a grounded answer. The LLM remains responsible for semantic decisions; Python validates the resulting contracts instead of routing meaning with keywords or regular expressions.

The guide agent uses validated structured model output so `openai/gpt-oss` models return a model-decided enquiry plan before the answer agent responds.

## Enquiry and response contracts

Agent 1 keeps a free-text `semantic_intent` and also assigns one stable 4+1 product intent family:

- `recommend_course` — choose one primary recommendation.
- `recommend_with_details` — choose one primary recommendation with useful factual details.
- `compare_courses` — compare two or more resolved or retrieved options.
- `explore_direction` — help the user discover a suitable learning direction.
- `free_style` — factual, filtering, suitability, unresolved-reference, refusal, and other valid course enquiries.

Intent describes what the user wants. A separate response mode controls how the current turn behaves: `recommend_one`, `recommend_one_with_details`, `compare`, `course_info`, `explore`, `clarify`, `clarify_with_suggestion`, or `refuse`. A completed turn can also become `no_result` when retrieval finds no suitable course.

Clarification is a tracked response behaviour, not another intent. The guide first considers conversation history, structured dialogue state, course retrieval, personal data, and a safe interpretation that can be re-checked. When useful information is already available, the assistant can answer under an explicit assumption or offer a grounded suggestion followed by one focused question instead of blocking on clarification.

## Dialogue state, tools, and grounding

`POST /api/chat` continues to accept `query` and `conversation`, and additionally accepts an optional hidden `dialogue_state`. Each response returns the updated state so the browser can send it with the next turn without displaying it. This state carries resolved course IDs, the previous primary and related courses, active constraints, unresolved references, the current goal, lightweight last-intent/mode context, and an optional `pending_clarification`.

The pending clarification records the information target, reason, exact question shown to the user, any visible options, their supporting course IDs, whether retrieval was required, and the attempt number. On the next turn Agent 1 classifies the user's response as answered, partially answered, rejected, topic changed, or not applicable. Useful information is merged into the flexible `active_constraints`; an answer clears or advances the gap, a partial answer preserves what was learned and narrows the next question, and a topic change abandons the irrelevant gap. Current explicit constraints override older structured values while unrelated constraints remain available. After two consecutive unsuccessful attempts on essentially the same gap, the assistant stops repeating questions and makes the best grounded progress available.

The stored pending state is built from Agent 2's final user-facing output, not an Agent 1 draft. This ensures that short replies such as “ทำ content” are interpreted against the question the user actually saw. The browser only stores and returns this state; it does not interpret dialogue semantics.

### Ground before suggesting

Agent 1 decides what information is missing and whether a useful clarification requires catalogue evidence. It has no course tools and does not invent visible course choices. Agent 2 selects tools, verifies the catalogue, and constructs the actual grounded question and answer.

A generic question about the user, such as asking what kind of work they want AI to help with, does not assert catalogue availability and needs no course retrieval. A course name, course ID, availability claim, catalogue-backed learning direction, comparison candidate, or visible course-backed option must be supported by retrieved course evidence before it is shown. Each structured clarification option carries `supporting_course_ids`; unsupported IDs are removed deterministically, and an option left without valid support is not displayed. The assistant falls back to a safe open-ended question when grounded options cannot be produced.

Agent 2 remains LLM-led and can use `course_catalog`, `course_id`, and `personal_data`. Personal data is an evidence source, not another intent family, and only relevant profile fields should be used when they materially improve a recommendation, comparison, exploration, or suitability answer.

The complete catalogue can be retained as retrieval evidence without becoming a list of cards. Agent 2 explicitly selects `related_course_ids`; the API validates those IDs against retrieved evidence and loads card contents from the local catalogue. Clarification-only, refusal, and no-result responses normally return no related cards.

Final results include a structured response mode, primary/referenced/related/evidence course IDs, the user-facing answer, an optional clarification target and question, and grounded clarification options. Deterministic ID, option, and mode validation prevents fabricated course references. The semantic grounding check evaluates factual claims inside clarification responses as well as direct answers; a failed check is corrected once from the available evidence, with no unbounded regeneration loop.

Optional examples in `local_data/intent_examples.json` (or the path set by `GUIDE_EXAMPLE_SET_PATH`) provide few-shot semantic guidance to Agent 1. They are not compared with user strings and are not a keyword lookup table; the guide still works when the file is absent.

Locally generated integration-evaluation artifacts are kept under the ignored `test_results/` directory.

## Run the behaviour evaluation

Run the live multi-turn behaviour scenarios and write `test_results/baseline_raw.json` plus `test_results/chat_evaluation.md`:

```powershell
.\.venv\Scripts\python.exe scripts\run_baseline_evaluation.py
```

Every relevant turn records the intent family, planned and final response modes, pending clarification before and after the turn, resolution, target, whether retrieval preceded clarification, clarification grounding, repeated-question status, and related-course grounding. The report retains the existing behaviour scores and adds Clarification Resolution Accuracy, Ground-Before-Suggest Compliance, Clarification Loop Rate, Grounded Clarification Option Rate, and Multi-turn Information Accumulation Accuracy. Execution success is reported separately and is never presented as chatbot accuracy.

## Demonstrating grounded multi-turn resolution

A useful presentation starts with a vague AI request. Agent 1 can plan `explore_direction`, identify `learning_direction` as missing, and set `clarification_requires_retrieval=true` without naming options. Agent 2 then calls `course_catalog`, finds real evidence such as AI201 for Machine Learning and AI301 for Generative AI, and asks a question using only those supported directions. The returned dialogue state stores that exact question and both supporting IDs.

If the user replies “น่าจะ Gen AI”, Agent 1 marks the pending clarification as answered and adds the direction to active constraints. Agent 2 then uses real catalogue evidence for the recommendation. This makes enquiry intent detection a grounded conversational resolution process rather than a single-turn classifier.

## Model reliability

The model client uses a 30-second request timeout and two bounded retries for transient provider failures. Override them with `GROQ_TIMEOUT_SECONDS` and `GROQ_RETRY_ATTEMPTS`; exhausted transient failures return HTTP 503.
