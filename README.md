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

Clarification is a last resort. The guide first considers conversation history, structured dialogue state, course retrieval, personal data, and a safe interpretation that can be re-checked. When useful information is already available, the assistant can answer under an explicit assumption or offer a grounded suggestion followed by one focused question instead of blocking on clarification.

## Dialogue state, tools, and grounding

`POST /api/chat` continues to accept `query` and `conversation`, and additionally accepts an optional hidden `dialogue_state`. Each response returns the updated state so the browser can send it with the next turn without displaying it. This state carries resolved course IDs, the previous primary and related courses, active constraints, unresolved references, and the current goal. Current explicit constraints override older structured values, while unrelated constraints remain available across turns. If state is omitted, the agents can reconstruct useful context from the conversation.

Agent 2 remains LLM-led and can use `course_catalog`, `course_id`, and `personal_data`. Personal data is an evidence source, not another intent family, and only relevant profile fields should be used when they materially improve a recommendation, comparison, exploration, or suitability answer.

The complete catalogue can be retained as retrieval evidence without becoming a list of cards. Agent 2 explicitly selects `related_course_ids`; the API validates those IDs against retrieved evidence and loads card contents from the local catalogue. Clarification-only, refusal, and no-result responses normally return no related cards.

Final results include a structured response mode, primary/referenced/related/evidence course IDs, the user-facing answer, and an optional clarification question. Deterministic ID and mode validation prevents fabricated course references, and a lightweight semantic grounding check detects unsupported factual claims. A failed semantic check is corrected once from the available evidence, with no unbounded regeneration loop.

Optional examples in `local_data/intent_examples.json` (or the path set by `GUIDE_EXAMPLE_SET_PATH`) provide few-shot semantic guidance to Agent 1. They are not compared with user strings and are not a keyword lookup table; the guide still works when the file is absent.

Locally generated integration-evaluation artifacts are kept under the ignored `test_results/` directory.

## Run the behaviour evaluation

Run the live multi-turn behaviour scenarios and write `test_results/baseline_raw.json` plus `test_results/chat_evaluation.md`:

```powershell
.\.venv\Scripts\python.exe scripts\run_baseline_evaluation.py
```

Every relevant turn is scored separately for Intent Family, Response Mode, Retrieval, Conversation Resolution, Clarification Behaviour, Personalization, Grounding, Related Courses, Thai Response, and Final Behaviour. The report publishes named metrics such as Intent Family Accuracy, Response Mode Accuracy, Retrieval Success, Multi-turn Resolution Accuracy, Clarification Quality, Grounded Answer Rate, Related Course Accuracy, and End-to-End Behaviour Accuracy. Execution success is reported separately and is never presented as chatbot accuracy.

## Model reliability

The model client uses a 30-second request timeout and two bounded retries for transient provider failures. Override them with `GROQ_TIMEOUT_SECONDS` and `GROQ_RETRY_ATTEMPTS`; exhausted transient failures return HTTP 503.
