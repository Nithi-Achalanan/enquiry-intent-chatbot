# Enquiry Intent Chatbot

The chat interface keeps long conversations scrollable in the message pane, renders assistant replies with safe Markdown formatting, clamps related-course descriptions to two lines, and includes a Reset chat button that clears the current browser session.

This repository was created entirely by **Nithi Achalanan** individually.

A two-agent conversational course-enquiry system built with **LangGraph**, **LangChain**, **OpenAI**, and **FastAPI**.

The project is designed around **Enquiry Intent Detection**, but the intent layer is intentionally more capable than a normal classifier. Instead of only assigning a label to each message, the system:

- understands the user's current goal semantically,
- classifies the request into one of five stable enquiry families,
- resolves references and constraints from previous turns,
- detects missing or ambiguous information,
- decides whether to answer, retrieve, clarify, re-check, compare, or explore,
- tracks clarification questions across multiple turns,
- retrieves course/profile evidence when needed,
- and produces a grounded final answer with verified related-course cards.

The system remains intentionally small and inspectable for a technical-test / POC use case.

---

## System Architecture and Flowchart

The application keeps the existing two-agent architecture.

```text
User
 │
 │  query
 │  conversation
 │  dialogue_state
 ▼
┌─────────────────────────────────────────────┐
│ Agent 1 — Template / Guide / How-to-Answer │
│                                             │
│ • semantic enquiry understanding            │
│ • 4+1 intent family                         │
│ • reference resolution                      │
│ • active constraint tracking                │
│ • missing-information detection             │
│ • clarification planning                    │
│ • next response-mode planning               │
└─────────────────────┬───────────────────────┘
                      │
                      │ GuidePlan
                      ▼
┌─────────────────────────────────────────────┐
│ Agent 2 — Search-and-Answer Agent           │
│                                             │
│ • decides which tool is required            │
│ • retrieves real course/profile evidence    │
│ • can loop through tools                    │
│ • creates the user-facing response          │
│ • selects Related Course IDs                │
└───────────┬─────────────────────┬───────────┘
            │                     │
            ▼                     ▼
     course_catalog /          personal_data
       course_id
            │                     │
            └──────────┬──────────┘
                       ▼
                 factual evidence
                       │
                       ▼
              structured final result
                       │
             deterministic validation
                       │
               semantic grounding
                       │
                       ▼
            answer + related_courses
                       │
                       ▼
                 FastAPI / UI
```

LangGraph controls the flow between Agent 1, Agent 2, and the three tools.

---

## Agent 1 — Enquiry Understanding and Answer Planning

`src/agents/template_design.py`

Agent 1 is the semantic planning layer.

It **does not retrieve course data** and **does not answer the user directly**.

Its job is to understand:

1. What is the user actually trying to do?
2. What is already known from this conversation?
3. Which course references can be resolved?
4. Which constraints are active?
5. What information is still missing?
6. Can the system make useful progress without asking?
7. Does the next clarification require real catalogue evidence?
8. What response behaviour should Agent 2 execute?

The semantic decision remains LLM-led. The code intentionally avoids Python keyword/regex intent classifiers.

---

## 4 + 1 Enquiry Intent Families

Agent 1 returns a free-text `semantic_intent` plus one stable product-level `intent_family`.

| Intent family | Purpose |
| --- | --- |
| `recommend_course` | Select one primary course for a sufficiently clear user goal. |
| `recommend_with_details` | Select one primary course and provide useful factual details. |
| `compare_courses` | Compare two or more explicit, resolved, or retrieved course options. |
| `explore_direction` | Help a user who is interested in learning but does not yet know the right direction. |
| `free_style` | Direct course facts, filtering, suitability, unresolved references, unknown IDs, refusal, and other valid course enquiries. |

Example:

```json
{
  "semantic_intent": "The user wants to explore AI courses for work but does not yet know which direction fits them.",
  "intent_family": "explore_direction"
}
```

This keeps semantic understanding flexible while maintaining a stable product contract.

---

## Intent Is Separate from Response Behaviour

The user's intent and the system's next action are not the same thing.

Agent 1 also returns a `planned_response_mode`.

Supported modes:

- `recommend_one`
- `recommend_one_with_details`
- `compare`
- `course_info`
- `explore`
- `clarify`
- `clarify_with_suggestion`
- `refuse`

After retrieval, Agent 2 may also produce:

- `no_result`

For example:

```text
intent_family = explore_direction
planned_response_mode = clarify_with_suggestion
```

means that the user is still exploring, but the bot can provide grounded help and then ask one focused question.

---

## Multi-Turn Dialogue State

The application keeps both:

```text
conversation
```

and:

```text
dialogue_state
```

They serve different purposes.

- `conversation` preserves what the user and assistant actually said.
- `dialogue_state` preserves what the system has resolved and is currently tracking.

The structured state includes:

```text
resolved_course_ids
last_primary_course_id
last_related_course_ids
active_constraints
unresolved_references
current_goal
pending_clarification
clarification_count
last_intent_family
last_response_mode
```

The frontend keeps this state privately and sends it back with the next request.

This allows short follow-up messages such as:

```text
"แล้วตัวนี้ล่ะ"
"ทำ content"
"น่าจะ Gen AI"
```

to be interpreted in the context of the previous turn instead of being treated as unrelated standalone queries.

---

## Constraint Accumulation and Override

User constraints can accumulate over multiple turns.

```text
Turn 1:
"I want a Data course."

active_constraints:
topic = Data
```

```text
Turn 2:
"For a beginner."

active_constraints:
topic = Data
level = beginner
```

```text
Turn 3:
"I want something shorter."

active_constraints:
topic = Data
level = beginner
duration = short
```

A newer explicit user statement overrides an older value.

```text
Turn 4:
"Actually, intermediate is okay."

active_constraints:
topic = Data
level = intermediate
duration = short
```

---

## Reference Resolution

The dialogue state helps resolve references such as:

- "this course"
- "that one"
- "the previous one"
- "the first one"
- "คอร์สนี้"
- "ตัวนี้"
- "อันนั้น"
- "เมื่อกี้"

Example:

```text
User:
ช่วยเลือกคอร์ส Machine Learning ให้หนึ่งคอร์ส

Assistant:
ผมแนะนำ AI201

User:
แล้วคอร์สนี้ต้องมีพื้นฐานอะไร
```

The next turn can resolve:

```text
คอร์สนี้ → AI201
```

and retrieve the exact course rather than asking the user which course they mean.

If no reliable referent exists, the bot asks one focused clarification instead of guessing.

---

## Clarification Tracking

Clarification is a response behaviour, not a sixth intent.

When a clarification question is actually shown to the user, the system stores a `PendingClarification`.

Conceptually:

```json
{
  "target": "learning_direction",
  "reason": "The user's goal is still too broad for a useful recommendation.",
  "question": "คุณอยากนำ AI ไปช่วยงานแบบไหนเป็นหลักครับ?",
  "options": [],
  "supporting_course_ids": [],
  "retrieval_required": false,
  "attempt": 1
}
```

On the next turn, Agent 1 determines whether the user:

- answered the pending clarification,
- partially answered it,
- rejected / could not answer it,
- changed topic,
- or did not interact with it.

Example:

```text
User:
อยากเรียน AI แต่ยังไม่รู้ว่าจะเรียนอะไร

Assistant:
คุณอยากนำ AI ไปช่วยงานแบบไหนเป็นหลักครับ?

User:
ทำ content
```

The second message is interpreted as an answer to the pending question. The useful information is merged into `active_constraints`, and the dialogue can continue toward retrieval and recommendation.

### Clarification Loop Protection

The system avoids repeatedly asking the same question.

A same-target clarification is bounded. After repeated unsuccessful attempts, the workflow stops forcing the same question and instead makes the best grounded progress available or returns a safe no-result response.

The implementation also detects an exact repeated user-facing clarification question and performs a bounded correction rather than entering an infinite loop.

---

## Intent Switching During a Conversation

The previous intent does **not** lock the next turn.

Every current message is semantically evaluated again using:

```text
current query
+
conversation
+
dialogue state
+
pending clarification
```

This supports:

```text
Intent continuation:
recommend → recommend

Intent evolution:
explore → clarify → recommend

Intent switching:
explore → course_info
recommend → compare
compare → course_info
```

Example:

```text
Assistant:
คุณอยากใช้ AI กับงานแบบไหนครับ?

User:
จริง ๆ ขอถามราคา CS101 ก่อน
```

The old pending clarification is abandoned and the current enquiry becomes a `free_style / course_info` turn.

The user never has to finish an old clarification before asking something else.

---

## Ground Before Suggesting

One of the main reliability rules in this project is:

> Any user-visible suggestion that implies a real course or catalogue-backed learning direction must be supported by retrieved course evidence first.

This prevents the LLM from inventing course offerings.

Agent 1 has no course tools. It can identify that the user needs choices, but it must not invent catalogue-backed options.

If real catalogue choices are needed:

```text
Agent 1
  │
  │ clarification_requires_retrieval = true
  ▼
Agent 2
  │
  ▼
course_catalog()
  │
  ▼
real catalogue evidence
  │
  ▼
grounded options / suggestion / question
```

Example:

```text
User:
อยากเรียน AI แต่ยังไม่รู้ว่าจะไปทางไหนดี
```

Agent 1 may decide that a catalogue-backed clarification is useful, but it does not pre-write course-backed options as if they exist.

Instead, Agent 2 retrieves the real catalogue first.

Only after retrieval may it present directions or courses supported by the actual catalogue.

### Generic Clarification vs Catalogue-Backed Clarification

A generic user-focused question does not require course retrieval:

```text
คุณอยากเอา AI ไปช่วยงานประเภทไหนมากที่สุดครับ?
```

A catalogue-backed statement does:

```text
จากคอร์สที่มี คุณสามารถเลือก Machine Learning หรือ Generative AI ได้
```

The latter is allowed only when retrieved course evidence supports those options.

---

## Grounded Clarification Options

When a clarification shows catalogue-backed choices, the final result carries structured supporting IDs.

Example:

```json
{
  "clarification_question": "คุณสนใจ Machine Learning หรือ Generative AI มากกว่ากัน?",
  "clarification_options": [
    {
      "label": "Machine Learning",
      "supporting_course_ids": ["AI201"]
    },
    {
      "label": "Generative AI",
      "supporting_course_ids": ["AI301"]
    }
  ]
}
```

The deterministic validator removes unsupported IDs/options before the response reaches the user.

---

## Agent 2 — Search and Answer

`src/agents/data_retriever.py`

Agent 2 executes the plan from Agent 1.

It remains LLM-led and decides which tool is required, except for explicit reliability contracts such as mandatory catalogue retrieval before a catalogue-backed clarification.

The agent can call at most one tool per graph invocation, then LangGraph routes the result back to Agent 2 so it can inspect the evidence and decide what to do next.

```text
Agent 2
  │
  ├── course_catalog
  │
  ▼
Agent 2
  │
  ├── personal_data
  │
  ▼
Agent 2
  │
  └── final answer
```

---

## Tools

The current retrieval subgraph exposes three business tools.

### 1. `course_catalog`

Reads the complete local course catalogue.

Used for:

- semantic course discovery,
- recommendation candidates,
- filtering,
- exploration,
- finding a semantic comparison target,
- catalogue-backed clarification choices.

The complete catalogue is evidence. It is **not automatically rendered as Related Courses**.

### 2. `course_id`

Retrieves one exact course using a known/resolved course ID.

Used for:

- direct course facts,
- price,
- schedule,
- prerequisites,
- instructor,
- exact-course suitability,
- previous-turn course follow-ups,
- unknown-ID detection.

Exact lookup artifacts are copied into shared retrieval evidence so final grounding and Related Course validation use the same factual source.

### 3. `personal_data`

Reads the local mock learner profile.

Used when profile evidence materially improves:

- personalized recommendation,
- course suitability,
- exploration,
- prerequisite comparison,
- personal comparison.

Personal data is an evidence source, **not another intent family**.

---

## Graph Orchestration

`src/graph.py`

The main graph remains simple:

```text
START
  │
  ▼
template_agent
  │
  ▼
retrieval subgraph
  │
  ▼
END
```

The retrieval subgraph is:

```text
START
  │
  ▼
search_agent
  │
  ├── course_catalog_tool ──┐
  ├── course_id_tool ───────┤
  ├── personal_data_tool ───┤
  │                         │
  └─────────────────────────┘
            │
            ▼
       search_agent
            │
            ├── another tool
            │
            └── final answer
```

A hard tool-call limit prevents an unbounded tool loop.

If Agent 1 marks:

```text
clarification_requires_retrieval = true
```

Agent 2 retrieves `course_catalog` before it can finalize that catalogue-backed clarification.

---

## GuidePlan Contract

Agent 1 passes a structured plan directly through graph state.

| Field | Purpose |
| --- | --- |
| `semantic_intent` | Flexible natural-language description of what the user currently wants. |
| `intent_family` | Stable 4+1 product intent. |
| `planned_response_mode` | Behaviour expected from the current turn. |
| `resolved_course_ids` | Course references already resolved from query/history. |
| `active_constraints` | Flexible accumulated user constraints. |
| `unresolved_references` | References that still cannot safely be resolved. |
| `missing_information` | Information that materially affects the answer. |
| `assumptions` | Short inspectable assumptions, not hidden chain-of-thought. |
| `required_information` | Evidence Agent 2 needs. |
| `personal_data_needed` | Whether profile evidence can materially improve the result. |
| `clarification_needed` | Whether the turn requires a clarification. |
| `clarification_target` | The information gap the bot is trying to resolve. |
| `clarification_requires_retrieval` | Whether visible choices require catalogue evidence first. |
| `clarification_option_goal` | Semantic description of the real options Agent 2 should find. |
| `pending_clarification_resolution` | How the current message affected the previous pending question. |

The plan is also preserved in Agent 1 message memory for inspectability.

---

## FinalAnswerResult Contract

Agent 2 does not return only an uncontrolled text string.

The final response is structured before being exposed through the API.

Important fields include:

```text
final_response_mode
answer
primary_course_id
referenced_course_ids
related_course_ids
evidence_course_ids
clarification_question
clarification_target
clarification_options
```

Python validates structural contracts after the LLM has made semantic choices.

Examples:

- a recommendation must have a real primary course,
- referenced and related IDs must exist in retrieved evidence,
- a comparison requires enough verified targets,
- unsupported clarification-option IDs are removed,
- `no_result`, `refuse`, and clarification-only responses do not expose unrelated course cards.

---

## Grounding Validation

Grounding is checked in two layers.

### Deterministic validation

Python verifies:

- course IDs,
- Related Course IDs,
- comparison target count,
- recommendation primary ID,
- clarification-option evidence,
- allowed response-mode transitions.

This validation does not perform semantic intent classification.

### Semantic grounding verification

A structured LLM check reviews the final user-facing answer against:

- retrieved course evidence,
- relevant personal-data evidence,
- clarification options.

It is intended to detect unsupported factual claims such as invented:

- prices,
- schedules,
- prerequisites,
- instructors,
- course IDs,
- profile facts,
- catalogue availability,
- course-backed learning directions.

If grounding fails, the system performs one bounded correction attempt using the same evidence.

---

## Related Courses

Related Course cards are explicitly selected by Agent 2.

The system does not render every course returned by `course_catalog`.

```text
retrieval evidence
      │
      ▼
Agent 2 selects related_course_ids
      │
      ▼
validate each ID against retrieval evidence
      │
      ▼
load the real local course object
      │
      ▼
frontend Related Course cards
```

This allows the complete catalogue to be available for semantic reasoning while keeping the UI focused.

---

## State Artifacts

| Field | Stores | Written by | Main purpose |
| --- | --- | --- | --- |
| `query` | Current user enquiry | API / entry point | Current-turn input. |
| `conversation` | Previous user/assistant messages | Frontend / API | Raw conversational context. |
| `dialogue_state` | Structured cross-turn state | Agents | Reference, constraint, and clarification tracking. |
| `guide_plan` | Structured Agent 1 plan | Agent 1 | Direct contract for Agent 2. |
| `guide_agent_state_memory` | Agent 1 messages | Agent 1 | Inspectability / compatibility. |
| `search_agent_state_memory` | Agent 2 and ToolMessages | Retrieval subgraph | Tool-loop context. |
| `retrieved_context_raw` | Raw factual tool artifacts | Tool nodes | Grounding, validation, Related Courses. |
| `search_attempts` | Number of search/tool iterations | Retrieval flow | Observability. |
| `tool_call_count` | Number of executed tool calls | Tool nodes | Hard loop limit. |
| `tool_call_artifacts` | Tool execution records | Tool nodes | Evaluation / debugging. |
| `final_result` | Structured final-answer contract | Agent 2 | Validation and API extraction. |
| `final_answer` | User-facing text | Agent 2 | Chat response. |
| `grounding_status` | Grounding/correction status | Agent 2 | Reliability signal. |
| `grounding_issues` | Validation / grounding problems | Agent 2 | Debugging and evaluation. |

---

## Local Data

### Course Catalogue

`local_data/course.jsonl`

The local course catalogue is the factual source of truth for course information.

Course records include fields such as:

```text
course_id
course_name
instructor
description
category
level
target_audience
duration
schedule
price
prerequisites
```

The model may semantically select among these records, but it must not invent new course data.

### Personal Data

`local_data/personal_data.json`

Contains one mock learner profile used for personalization testing.

### Intent Examples

`local_data/intent_examples.json`

Contains optional few-shot semantic examples for Agent 1.

They are not a Python lookup table and are not matched against the user's text with keyword rules.

An alternate file can be configured with:

```text
GUIDE_EXAMPLE_SET_PATH
```

---

## Project Structure

```text
.
├── docs/
│   └── full-stack-reliability-engineer/
│       ├── CAPACITY_AND_OVERLOAD.md
│       ├── DECISIONS.md
│       ├── DEPENDENCY_AND_FAILURE_MAP.md
│       ├── FAILURE_MODE_REGISTER.md
│       ├── FINAL_REPORT.md
│       ├── RELIABILITY_PLAN.md
│       ├── RESILIENCE_TEST_PLAN.md
│       ├── RUN_LOG.md
│       ├── SCOPE.md
│       ├── SERVICE_CATALOG.md
│       ├── SLI_SLO_ERROR_BUDGET.md
│       ├── TIMEOUT_RETRY_IDEMPOTENCY.md
│       └── UI_CHANGE_REPORT.md
│
├── frontend/
│   ├── app.js
│   ├── index.html
│   └── style.css
│
├── local_data/
│   ├── course.jsonl
│   ├── intent_examples.json
│   └── personal_data.json
│
├── scripts/
│   └── run_baseline_evaluation.py
│
├── src/
│   ├── agents/
│   │   ├── data_retriever.py
│   │   └── template_design.py
│   ├── tools/
│   │   ├── course_catalog.py
│   │   ├── course_id.py
│   │   └── personal_data.py
│   ├── config.py
│   ├── graph.py
│   ├── main.py
│   ├── reliability.py
│   └── state.py
│
├── tests/
│   ├── test_clarification_grounding.py
│   ├── test_config.py
│   ├── test_llm_only_design.py
│   ├── test_reliability.py
│   └── test_tool_call_limit.py
│
├── .env.example
├── .gitattributes
├── .gitignore
├── README.md
└── requirements.txt
```

---

## System Map

| System part | What it does | Key files |
| --- | --- | --- |
| Enquiry understanding | Semantic understanding, 4+1 intent selection, missing-information and clarification planning. | `src/agents/template_design.py` |
| Search and answer | Chooses tools, reviews evidence, answers, and produces structured final output. | `src/agents/data_retriever.py` |
| Dialogue state | Tracks resolved courses, constraints, pending clarification, and prior-turn context. | `src/state.py` |
| Workflow | Connects the guide agent and retrieval loop using LangGraph. | `src/graph.py` |
| Course discovery | Returns the factual local catalogue. | `src/tools/course_catalog.py` |
| Exact course lookup | Retrieves one exact known/resolved course ID. | `src/tools/course_id.py` |
| Personalization | Loads the mock learner profile. | `src/tools/personal_data.py` |
| API / application | Serves frontend, `/api/chat`, and `/api/health`. | `src/main.py` |
| Model reliability | Timeout, bounded retry, backoff, and provider diagnostics. | `src/reliability.py` |
| Web interface | Chat UI and Related Course cards. | `frontend/` |
| Behaviour evaluation | Runs live multi-turn scenarios and writes ignored evaluation artifacts. | `scripts/run_baseline_evaluation.py` |

---

## API Behaviour

The frontend and backend run from the same FastAPI process.

### Request

`POST /api/chat`

```json
{
  "query": "ผมอยากเรียน AI แต่ยังไม่รู้ว่าจะไปทางไหนดี",
  "conversation": [],
  "dialogue_state": null
}
```

### Response

Conceptually:

```json
{
  "answer": "คำตอบภาษาไทย...",
  "related_courses": [],
  "dialogue_state": {
    "resolved_course_ids": [],
    "last_primary_course_id": null,
    "last_related_course_ids": [],
    "active_constraints": {
      "topic": "AI"
    },
    "unresolved_references": [],
    "current_goal": "Explore an AI learning direction",
    "pending_clarification": null,
    "clarification_count": 0,
    "last_intent_family": "explore_direction",
    "last_response_mode": "explore"
  }
}
```

`dialogue_state` is application state and is not rendered directly to the user.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Nithi-Achalanan/enquiry-intent-chatbot.git
cd enquiry-intent-chatbot
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env`, then configure:

```env
API_KEY=YOUR_OPENAI_API_KEY
MODEL=gpt-4.1-mini
OPENAI_TIMEOUT_SECONDS=30
OPENAI_RETRY_ATTEMPTS=2
```

Do not commit API credentials.

---

## Run

Start the FastAPI application:

```bash
uvicorn src.main:app --reload
```

Then open:

```text
http://localhost:8000
```

The same Python process serves:

- the frontend,
- `POST /api/chat`,
- `GET /api/health`.

No separate Node/React frontend process is required.

---

## Example Conversation

```text
User:
ผมอยากเรียน AI แต่ยังไม่รู้ว่าจะไปทางไหนดี

Agent 1:
intent_family = explore_direction
missing information = learning direction
clarification_requires_retrieval = true

Agent 2:
course_catalog()

Evidence:
AI201 → Machine Learning
AI301 → Generative AI

Assistant:
จากคอร์สที่มี ตอนนี้มีตัวเลือกที่ไปทาง Machine Learning
และ Generative AI ครับ คุณสนใจแนวไหนมากกว่ากัน?

Dialogue state:
pending_clarification.target = learning_direction
supporting_course_ids = [AI201, AI301]

User:
น่าจะ Gen AI

Agent 1:
pending clarification = answered
active constraint = Generative AI

Agent 2:
uses real catalogue evidence
→ grounded next response
```

The important part is that the visible course-backed choices are created **after** real catalogue retrieval.

---

## Tests

The repository contains contract-focused unit tests for areas including:

- fixed 4+1 intent families,
- free-text semantic intent,
- absence of Python keyword intent classification,
- response-mode compatibility,
- dialogue constraint accumulation,
- constraint override,
- reference state,
- course catalogue artifacts,
- exact course evidence,
- personal-data retrieval,
- Related Course validation,
- clarification-state creation and clearing,
- pending-clarification resolution,
- catalogue-backed clarification options,
- forced catalogue retrieval before grounded suggestions,
- repeated-question detection,
- clarification attempt limits,
- API dialogue-state round trips,
- provider retry handling,
- model configuration,
- tool-call limits.

Run:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

These tests focus on structured behaviour/contracts instead of requiring one exact LLM sentence.

---

## Behaviour Evaluation

Run:

```bash
python scripts/run_baseline_evaluation.py
```

Generated evaluation artifacts are written to the ignored:

```text
test_results/
```

The evaluation covers multi-turn behaviour including:

- vague exploration,
- generic clarification,
- unresolved references,
- retrieval-resolved ambiguity,
- previous-course pronouns,
- multi-turn constraint accumulation,
- constraint replacement,
- personalized recommendation,
- exact-course personalized suitability,
- semantic comparison,
- no-result handling,
- unknown course IDs,
- prompt injection,
- mixed Thai/English,
- typos,
- direct factual questions,
- short clarification answers,
- partial clarification answers,
- rejected clarification,
- clarification-loop bounds,
- topic / intent switching,
- catalogue-backed clarification cards.

The evaluation records execution success separately from chatbot behaviour.

Useful metrics include:

- Intent Family Accuracy
- Response Mode Accuracy
- Retrieval Success
- Multi-turn Resolution Accuracy
- Clarification Resolution Accuracy
- Ground-Before-Suggest Compliance
- Clarification Loop Rate
- Grounded Clarification Option Rate
- Multi-turn Information Accumulation Accuracy
- Grounded Answer Rate
- Related Course Accuracy
- End-to-End Behaviour Accuracy

---

## Reliability and Safety Handling

### Model timeout and retry

The project includes model reliability configuration for:

```text
OPENAI_TIMEOUT_SECONDS
OPENAI_RETRY_ATTEMPTS
```

Transient provider failures are handled with bounded retry logic rather than unbounded attempts.

### Tool-call limit

The retrieval subgraph has a hard maximum tool-call count to prevent an unbounded agent/tool loop.

### Prompt / state protection

The agents are instructed not to expose internal system prompts, hidden state, ToolMessages, secrets, chain-of-thought, or raw profile internals.

### No-result handling

When the catalogue cannot support the requested course, the system should return `no_result` instead of promoting an unrelated course.

### Grounded course data

The local course catalogue is the factual source of truth for course names, IDs, prices, instructors, schedules, prerequisites, and availability.

---

## Tech Stack

- Python
- LangGraph
- LangChain Core
- LangChain OpenAI
- OpenAI API
- Pydantic
- FastAPI
- Uvicorn
- Plain HTML
- CSS
- Vanilla JavaScript
- Local JSON-based course/profile data

---

## Production Considerations

This repository is intentionally a technical-test / POC implementation.

For a production system, likely next steps include:

### Server-side conversation state

The POC returns `dialogue_state` to the browser and receives it again on the next turn.

A production system would normally send a `conversation_id` and store trusted state server-side, for example in Redis or a database.

### Authentication and profile access control

The current personal-data source is a mock local profile.

Production personalization should enforce authentication, authorization, data minimization, and profile-level access control.

### Larger course catalogue

The complete catalogue is practical for the current small local dataset.

For a much larger catalogue, replace full-catalogue retrieval with an indexed retrieval layer while preserving the same evidence and final-validation contracts.

### Context management

Long-running conversations should use a context budget and may summarize old turns while preserving structured dialogue state and important references.

### Observability

Production telemetry should capture:

- latency per agent/model/tool,
- provider retries,
- tool-call counts,
- grounding failures,
- no-result rate,
- clarification rate,
- repeated clarification rate,
- intent/mode distribution,
- evaluation regressions.

### Stronger grounding

Higher-risk deployments may add more deterministic claim extraction, independent verification, or human review for selected outputs.

---

## Design Summary

```text
Understand semantically
        ↓
Resolve conversation state
        ↓
Detect information gaps
        ↓
Choose one of 4+1 intent families
        ↓
Choose the next conversational behaviour
        ↓
Retrieve before making catalogue-backed suggestions
        ↓
Answer / compare / recommend / clarify
        ↓
Validate IDs and response contracts
        ↓
Check semantic grounding
        ↓
Return answer + verified Related Courses
        ↓
Store structured dialogue state for the next turn
```

The key principle is:

> **The LLM makes semantic decisions, while Python enforces evidence and behavioural contracts.**

This keeps the chatbot flexible enough for natural multi-turn enquiries while reducing hallucinated course offerings and inconsistent response behaviour.
