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

The API delegates answers and related-course retrieval to the existing LangGraph workflow; it does not duplicate agent or tool logic.

The guide agent uses Groq's native JSON Schema output mode so `openai/gpt-oss` models return a validated enquiry plan instead of an unreliable forced tool call.

Locally generated integration-evaluation artifacts are kept under the ignored `test_results/` directory.
