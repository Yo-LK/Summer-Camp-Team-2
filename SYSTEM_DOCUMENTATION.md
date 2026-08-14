# Restaurant Reservation Agent: System Guide

This document explains the system from a user's and developer's point of view. It covers what each part does, how a chat message becomes a reservation, how availability is calculated, how data is stored, and how to operate the demo safely.

## 1. What the system does

The application lets a user request a restaurant reservation through a simple chat interface.

The user provides:

- Cuisine: Chinese, Korean, or Singaporean
- Price preference: `low`, `high`, or a numeric yuan budget
- Date
- Time
- Party size from 1 to 12

The system then:

1. Asks for any missing or unclear information.
2. Selects a restaurant matching the cuisine and tier.
3. Checks whether enough seats remain.
4. Shows the exact proposed reservation.
5. Waits for a separate confirmation such as `yes, book it`.
6. Checks capacity again and writes the reservation to CSV.
7. Returns the restaurant, date, time, party size, and confirmation ID.

Example:

```text
User: Low-tier Korean for 4 tomorrow at 19:00.
Agent: Seoul Kitchen is available on 2026-08-15 at 19:00 for 4 people.
       Reply 'yes, book it' to confirm.
User: Yes, book it.
Agent: Reservation confirmed — Seoul Kitchen on 2026-08-15 at 19:00
       for 4 people. Confirmation: RSV0006.
```

## 2. System overview

```text
Browser
  |
  | User submits a chat message
  v
React frontend
  |
  | POST /api/chat
  v
FastAPI backend
  |
  v
Reservation agent <--------------------------+
  |                                           |
  | Prompt + function declarations            | Structured tool result
  v                                           |
Gemini model                                  |
  |                                           |
  | Requests a function call                  |
  v                                           |
Tool router                                   |
  |                                           |
  +--> search_restaurants --------------------+
  +--> check_availability --------------------+
  +--> make_reservation --> CSV files --------+
```

Gemini decides what information is missing and which allowed tool should run. Gemini never writes the CSV directly. The backend validates the tool arguments, executes local Python code, and sends the structured result back to Gemini.

Trusted instructions, confirmation enforcement, validation, and file access remain on the backend.

## 3. Repository structure

| Path | Purpose |
| --- | --- |
| `frontend/src/App.jsx` | Chat messages, input box, submit button, loading state, and inline errors |
| `frontend/src/index.css` | Minimal responsive chat styling |
| `frontend/vite.config.js` | Proxies frontend `/api` requests to FastAPI on port 8000 |
| `backend/app.py` | Creates the FastAPI application, loads `.env`, and configures CORS |
| `backend/main.py` | Starts the backend server |
| `backend/api/routes.py` | Implements `/api/health` and `/api/chat` |
| `backend/api/models.py` | Validates API request and response bodies |
| `backend/agent/react_loop.py` | Session state, Gemini calls, tool loop, confirmation gate, and final responses |
| `backend/agent/tool_router.py` | Declares tools for Gemini, validates arguments, and calls local tool functions |
| `backend/agent/schemas.py` | Pydantic models for tool arguments, pending bookings, responses, and traces |
| `backend/tools/search_restaurants.py` | Selects a restaurant by cuisine and price tier |
| `backend/tools/check_availability.py` | Validates a requested slot and calculates remaining seats |
| `backend/tools/make_reservations.py` | Rechecks capacity and atomically writes a confirmed reservation |
| `backend/tools/common.py` | Shared CSV validation, locking, status validation, and structured errors |
| `backend/data/restaurants.csv` | Restaurant catalog |
| `backend/data/availability.csv` | Recurring seat capacity for each restaurant and time |
| `backend/data/reservations.csv` | Persistent reservation records |
| `tests/` | Backend, API, agent-loop, storage, and tool tests |
| `.env` | Local Gemini API key and model; ignored by Git |
| `.gitignore` | Prevents secrets, dependencies, caches, and builds from being committed |

## 4. How a reservation is created

### Step 1: The frontend creates a session

When the page loads, React creates a random `session_id`. Every message from that browser page uses the same ID.

The frontend sends:

```json
{
  "session_id": "s_123abc",
  "message": "Low-tier Korean for 4 on 2026-08-15 at 19:00"
}
```

### Step 2: FastAPI validates the request

FastAPI and Pydantic reject:

- Missing fields
- Blank messages
- Messages longer than 2,000 characters
- Invalid session IDs
- Unexpected request fields

Valid requests are passed to the reservation agent.

### Step 3: The agent collects requirements

The agent tells Gemini to collect all five requirements before using a tool:

- Cuisine
- Price tier or yuan budget
- Date
- Time
- Party size

If information is missing, Gemini asks only for the missing details. Conversation content is stored in memory under the `session_id`, so details can be provided over multiple messages.

Dates are normalized to `YYYY-MM-DD`, times to 24-hour `HH:MM`, and relative dates are interpreted using the `Asia/Shanghai` timezone.

### Step 4: Gemini requests `search_restaurants`

The tool supports three cuisines and two fixed price tiers:

- `low`: average price at or below ¥100
- `high`: average price above ¥100

Users may state the tier directly or provide a numeric budget. A budget at or below
¥100 maps to low tier; a budget above ¥100 maps to high tier. If the user supplies a
budget and a contradictory explicit tier, the agent asks for clarification without
calling a tool.

After filtering by cuisine and the fixed threshold, the tool preserves its deterministic
selection rule: it chooses the cheapest eligible low-tier restaurant or the most
expensive eligible high-tier restaurant. If no restaurant exists in the requested
cuisine and tier, it returns a controlled `no_restaurant` error.

The tool returns a restaurant ID and restaurant details. The ID is used by later tools instead of trusting a model-generated restaurant name.

### Step 5: Gemini requests `check_availability`

The tool validates the date, time, party size, restaurant hours, offered slots, reservation data, and capacity.

If the requested time cannot fit the party, the tool returns up to three nearby times that can.

If the slot is available, the backend stores an exact pending proposal containing:

- Restaurant ID and name
- Date
- Time
- Party size

The API returns `awaiting_confirmation`. No CSV row has been written yet.

### Step 6: The user confirms separately

The reservation can be written only after a later, unambiguous confirmation such as:

```text
yes, book it
```

The first request cannot create a reservation, even if it says "book it". Any intervening non-confirmation message makes the pending proposal stale and requires another availability check.

During the confirmation turn, the backend allows only `make_reservation`. Its arguments must exactly match the pending proposal.

### Step 7: The reservation is written

`make_reservation` acquires a process lock, reads the current CSV again, and rechecks availability. This protects against another reservation taking seats between the first check and confirmation.

If capacity remains, the tool:

1. Creates the next confirmation ID, such as `RSV0006`.
2. Writes a complete replacement CSV to a temporary file.
3. Flushes the file to disk.
4. Atomically replaces the original reservations file.

The user receives a deterministic confirmation message only after the write succeeds.

## 5. The ReAct-style tool loop

The application uses a basic ReAct pattern:

```text
Model chooses action -> backend runs tool -> model observes result -> repeat or answer
```

Important safeguards:

- Maximum of six tool calls per request
- Unknown tools are rejected
- All tool arguments are validated with Pydantic
- One corrected call is permitted after malformed arguments
- Tool errors are returned to the model as structured observations
- Only short action summaries are returned in the API trace; private chain-of-thought is not exposed
- `make_reservation` is unavailable until explicit confirmation

Gemini Generate Content requests are stateless, so the backend sends the saved session content on each step. Model function-call content is preserved so Gemini's function-call IDs and thought signatures remain valid.

## 6. How restaurant selection works

`restaurants.csv` contains two restaurants for each supported cuisine.

Columns:

| Column | Meaning |
| --- | --- |
| `restaurant_id` | Stable internal ID, such as `R003` |
| `name` | Display name |
| `cuisine` | Chinese, Korean, or Singaporean |
| `average_price` | Average price in yuan used to classify the fixed tier |
| `rating` | Demo rating returned with search results |
| `hours` | Opening range in 24-hour time |

The tier is based on a fixed ¥100 threshold and is not relative to the other restaurants
in the cuisine.

## 7. How availability and capacity work

`availability.csv` defines recurring capacity.

Columns:

| Column | Meaning |
| --- | --- |
| `restaurant_id` | Restaurant owning the slot |
| `weekday` | A weekday name or `Everyday` |
| `time` | Offered reservation time |
| `total_seats` | Total seats available across all bookings at that time |

Remaining seats are calculated as:

```text
remaining seats = total seats - sum of confirmed party sizes
```

Only reservation rows with `status=confirmed` consume seats. Rows with `status=cancelled` consume zero seats.

### Why multiple reservations can use the same time

A time is a pool of seats, not an exclusive appointment. For example, if a slot has 24 seats:

```text
First reservation:   5 people
Second reservation:  6 people
Total occupied:     11 people
Remaining:          13 seats
```

Both reservations are valid because their combined party size does not exceed 24.

The current duplicate check considers a reservation identical only when all of these match:

- Session ID
- Restaurant ID
- Date
- Time
- Party size

Repeating the exact confirmation returns the existing reservation. Changing the party size creates a different reservation if capacity remains.

## 8. Reservation data

`reservations.csv` is the persistent booking store.

Columns:

| Column | Meaning |
| --- | --- |
| `confirmation_id` | Generated ID such as `RSV0006` |
| `session_id` | Browser conversation that created the booking |
| `restaurant_id` | Selected restaurant |
| `date` | Reservation date |
| `time` | Reservation time |
| `party_size` | Number of seats consumed while confirmed |
| `status` | Must be `confirmed` or `cancelled` |
| `created_at` | Creation timestamp in Asia/Shanghai |

Unknown statuses are treated as invalid data. The system returns a controlled storage error instead of silently counting or ignoring them.

## 9. Session behavior

Conversation history is stored in backend memory and keyed by `session_id`.

- Refreshing the page creates a new frontend session ID.
- Restarting the backend clears all conversation and pending-confirmation state.
- Confirmed CSV reservations remain after a restart.
- Repeating confirmation immediately after a successful booking returns the existing confirmation instead of writing another row.
- Starting another request after a completed booking begins a new in-memory conversation.

This is a single-process demo. It does not provide user accounts, shared sessions, or durable chat history.

## 10. API reference

### Health check

```http
GET /api/health
```

Example response:

```json
{
  "status": "ok",
  "agent_configured": true,
  "model": "gemini-3.5-flash-lite"
}
```

### Chat

```http
POST /api/chat
Content-Type: application/json
```

Request:

```json
{
  "session_id": "session_123",
  "message": "High-tier Chinese for 2 on 2026-08-15 at 19:00"
}
```

Response:

```json
{
  "session_id": "session_123",
  "reply": "Imperial Garden is available ...",
  "status": "awaiting_confirmation",
  "trace": [],
  "reservation": null,
  "done": false
}
```

Possible `status` values:

| Status | Meaning |
| --- | --- |
| `needs_info` | More or corrected information is required |
| `awaiting_confirmation` | The exact proposal is available but not written |
| `unavailable` | Capacity is insufficient or the requested slot is not available |
| `confirmed` | A reservation was successfully persisted |
| `error` | The agent stopped safely because it could not complete the workflow |

HTTP behavior:

- `200`: Normal conversation, including business problems such as unavailable slots
- `422`: Invalid HTTP request body
- `502`: Non-retryable Gemini/provider failure
- `503`: Missing configuration, rate limit, timeout, connection failure, or temporary provider overload
- `500`: Unexpected backend failure

## 11. Error handling

The system handles these common problems:

- Missing or ambiguous cuisine, price preference, date, time, or party size
- Unsupported cuisines and tiers
- Dates in the past
- Malformed dates and times
- Parties smaller than 1 or larger than 12
- Times outside restaurant hours
- Times not listed in availability data
- Insufficient capacity, with nearby alternatives
- Capacity changing before confirmation
- Missing, malformed, or unreadable CSV files
- Invalid reservation statuses or party sizes in stored data
- Invalid or hallucinated model tool calls
- More than six tool calls
- Missing Gemini API key
- Gemini rate limits, timeouts, connectivity failures, or high demand
- Frontend/backend network failures

Tool failures use this structure:

```json
{
  "ok": false,
  "error": {
    "code": "invalid_party_size",
    "message": "Party size must be between 1 and 12.",
    "retryable": false
  }
}
```

## 12. Running the application

Requirements:

- Python 3.10 or newer
- Node.js 20.19 or newer
- A Gemini API key

In the repository root, configure `.env`:

```dotenv
GEMINI_API_KEY=your_real_key
GEMINI_MODEL=gemini-3.5-flash-lite
```

Do not commit or share `.env`.

### Terminal 1: backend

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m backend.main
```

The backend runs at `http://127.0.0.1:8000`.

### Terminal 2: frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

Both terminals must remain running. Restart the backend after changing `.env` or backend Python files.

## 13. Cancelling a reservation safely

Cancellation is intentionally not available through the chat interface.

1. Stop the backend with `Ctrl+C`.
2. Open `backend/data/reservations.csv` as a text file.
3. Find the row using its `confirmation_id`.
4. Change only `confirmed` to `cancelled`.
5. Keep the row and CSV header.
6. Save the file and restart the backend.

Example before:

```csv
RSV0006,session_123,R003,2026-08-15,19:00,4,confirmed,2026-08-14T15:30:00+08:00
```

Example after:

```csv
RSV0006,session_123,R003,2026-08-15,19:00,4,cancelled,2026-08-14T15:30:00+08:00
```

The seats become available immediately because only confirmed rows count toward capacity.

## 14. Deleting reservation rows

Deleting a row also releases its seats, but cancellation is preferred.

Deleting causes these side effects:

- Booking history is permanently lost.
- The confirmation ID can no longer be looked up.
- If the highest-numbered row is deleted, that confirmation ID may be reused.
- Editing while the backend writes can lose or corrupt data.

If deletion is necessary, stop the backend, delete only the intended data row, preserve the CSV header, save, and restart. Never delete the header.

## 15. Editing CSV files

Some editor CSV previews are read-only. In VS Code:

1. Right-click the CSV in Explorer.
2. Select **Open With...**.
3. Choose **Text Editor**.
4. Make the change and save.

Rules for safe edits:

- Stop the backend first.
- Keep the column header unchanged.
- Keep the same number and order of comma-separated fields.
- Use only `confirmed` or `cancelled` in the status column.
- Do not open the CSV in a tool that silently changes date or time formatting.

## 16. Testing

Run backend tests from the repository root:

```bash
source .venv/bin/activate
pytest
```

Run frontend checks:

```bash
cd frontend
npm run lint
npm run build
```

Tests use temporary CSV files and a fake Gemini client, so normal automated tests do not change the real reservation data or consume Gemini quota.

## 17. Troubleshooting

### The frontend says it cannot reach the service

- Confirm the backend terminal is running.
- Open `http://127.0.0.1:8000/api/health`.
- Confirm the frontend is running on port 5173.

### Health says `agent_configured: false`

- Confirm `.env` is in the repository root.
- Confirm `GEMINI_API_KEY` is not blank or still a placeholder.
- Restart the backend after editing `.env`.

### Gemini is unavailable or under high demand

This is a retryable provider response. Wait briefly and submit again. The backend does not create a reservation unless `make_reservation` succeeds.

### The same time appears more than once in reservations

This is valid while the total confirmed party sizes fit within the slot's seat capacity. See [How availability and capacity work](#7-how-availability-and-capacity-work).

### A cancelled reservation still appears in the CSV

This is intentional. Cancelled rows preserve history but do not consume seats.

### The CSV reports an invalid status

Change the affected status to exactly `confirmed` or `cancelled`. Unknown values intentionally stop availability calculations.

## 18. Demo limitations

This project is deliberately minimal and is not production-ready.

- No authentication or customer identity
- No cancellation through chat or UI
- No durable conversation storage
- No database transactions
- No cross-process or multi-server CSV locking
- No protection against users sharing a session ID
- No administration UI
- No automatic cleanup of old reservations
- No timezone choice; reservation interpretation uses Asia/Shanghai

For production, replace CSV storage with a transactional database, add authenticated users and authorization, implement cancellation as a validated tool, and run concurrency tests against the chosen deployment architecture.
