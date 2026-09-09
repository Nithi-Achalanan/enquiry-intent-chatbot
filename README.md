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

The API delegates answers and related-course retrieval to the LangGraph workflow; it does not duplicate agent or tool logic. The LLM interprets each request, decides the answer strategy, and assesses the complete course catalogue from factual course details. There are no keyword extractors, fuzzy keyword matching, hard-coded intent labels, or canned intent examples.

The guide agent uses Groq's native JSON Schema output mode so `openai/gpt-oss` models return a validated, model-decided enquiry plan before the answer agent responds.

Locally generated integration-evaluation artifacts are kept under the ignored `test_results/` directory.

## Run the baseline evaluation

Run all 15 live scenarios and write `test_results/baseline_raw.json` plus
`test_results/chat_evaluation.md`:

```powershell
.\.venv\Scripts\python.exe scripts\run_baseline_evaluation.py
```

## Model reliability

The model client uses a 30-second request timeout and two bounded retries for transient provider failures. Override them with `GROQ_TIMEOUT_SECONDS` and `GROQ_RETRY_ATTEMPTS`; exhausted transient failures return HTTP 503.
