# Restaurant Reservation Agent

A minimal ReAct-style reservation demo built with React, FastAPI, Gemini's Generate Content API, and three CSV-backed tools.

For a complete plain-language explanation of the architecture, booking flow, data files, capacity rules, operations, and troubleshooting, read the [System Guide](SYSTEM_DOCUMENTATION.md).

## What it does

The agent accepts natural-language chat messages, collects missing reservation details, selects a restaurant, checks capacity, and asks for explicit confirmation before writing a booking. A successful reply includes the restaurant, date, time, party size, and confirmation ID.

Supported preferences:

- Cuisine: Chinese, Korean, or Singaporean
- Price preference: `low`, `high`, or a numeric yuan budget
- Party size: 1–12
- Time: one of the configured half-hour slots in `backend/data/availability.csv`

The fixed price benchmark is ¥100: budgets and restaurant average prices at or below
¥100 are low tier; values above ¥100 are high tier. If a budget contradicts an
explicit tier in the same message, the agent asks the user to clarify.

Example conversation:

```text
You: Low-tier Korean for 4 tomorrow at 19:00.
Agent: Seoul Kitchen is available on 2026-08-15 at 19:00 for 4 people. Reply 'yes, book it' to confirm.
You: Yes, book it.
Agent: Reservation confirmed — Seoul Kitchen on 2026-08-15 at 19:00 for 4 people. Confirmation: RSV0001.
```

## Architecture

```text
React chat → POST /api/chat → FastAPI → Gemini agent loop
                                           ├─ search_restaurants
                                           ├─ check_availability
                                           └─ make_reservation
                                                    ↓
                                              CSV demo state
```

The loop follows Gemini's [manual function-calling flow](https://ai.google.dev/gemini-api/docs/function-calling): it sends function declarations, executes validated local calls, and returns each structured function response to the model. Per-session content history preserves the model's function-call metadata between steps. Arguments are validated with Pydantic, and the loop stops after six tool calls.

## Setup

Requirements: Python 3.10+ and Node.js 20.19+.

1. Create a Gemini API key in Google AI Studio.
2. Open the root `.env` file and replace the placeholder:

   ```dotenv
   GEMINI_API_KEY=your_real_key
   GEMINI_MODEL=gemini-3.5-flash-lite
   ```

   The `.env` file is ignored by Git. Do not commit or share it. Google lists [`gemini-3.5-flash-lite`](https://ai.google.dev/gemini-api/docs/pricing) input and output as free of charge on the API free tier, subject to the account's current rate limits and data-use terms.

3. Install and run the backend:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   python -m backend.main
   ```

4. In another terminal, run the frontend:

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

5. Open `http://localhost:5173`.

The backend health endpoint is `GET http://127.0.0.1:8000/api/health`.

## API

`POST /api/chat`

```json
{
  "session_id": "session_123",
  "message": "High-tier Chinese for 2 on 2099-08-15 at 19:00"
}
```

The response contains `reply`, `status`, a compact tool `trace`, an optional structured `reservation`, and `done`. Business conditions such as missing information and unavailable slots return normal conversational responses. Invalid HTTP payloads return 422; Gemini configuration/provider failures return 503/502.

## Manual cancellation

Cancellation is intentionally outside the chat application:

1. Stop the backend so it cannot write the CSV at the same time.
2. Open `backend/data/reservations.csv`.
3. Find the row by `confirmation_id`.
4. Change only its `status` from `confirmed` to `cancelled`.
5. Keep the row and headers intact, save the file, and restart the backend.

Do not delete the row. Availability subtracts only `confirmed` reservations, so changing the status releases the seats while preserving the audit record. Any unknown status is treated as a controlled data error.

## Tests

```bash
pytest
cd frontend && npm run lint && npm run build
```

Tests use temporary CSV files and a fake Gemini Generate Content client, so they do not consume API quota. A live end-to-end test requires a real key in `.env` and should be performed through the UI.
