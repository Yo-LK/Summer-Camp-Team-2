## Restaurant Reservation Agent

### Update along the way

restaurant-reservation-agent/
│
├── backend/                          # Python backend: API + Agent + Tools
│   │
│   ├── data/                         # CSV databases — the agent's persistent state
│   │   ├── restaurants.csv           
│   │   ├── availability.csv          
│   │   └── reservations.csv         
│   ├── tools/                        # Tool implementations — pure functions that read/write CSVs
│   │   ├── search_restaurants.py     # Tool 1: Filters restaurants.csv by location, cuisine, price_range; returns matching list
│   │   ├── check_availability.py     # Tool 2: Looks up one row in availability.csv; returns whether slot has capacity
│   │   └── make_reservation.py       # Tool 3: Validates capacity, appends to reservations.csv, decrements availability.csv
│   │
│   ├── agent/                        # ReAct orchestration — the "brain" that decides which tool to call next
│   │   ├── react_loop.py             # Core reasoning engine: runs the Reason→Act→Observe loop, handles error recovery
│   │   ├── tool_router.py            # Maps agent reasoning strings to structured tool calls (name + arguments)
│   │   └── schemas.py                # Pydantic models defining the shape of tool inputs/outputs
│   │
│   ├── api/                          # HTTP layer — bridges the frontend to the agent
│   │   ├── routes.py                 # FastAPI endpoint definitions: E.g. /chat, /select, /restaurants, /health
│   │   └── models.py                 # Pydantic request/response contracts: ChatRequest, SelectRequest, AgentState 
│   ├── app.py                        # FastAPI application factory: assembles routes, CORS, middleware
│   └── main.py                       # Entry point
├── frontend/                         # Static web UI — renders the agent's reasoning trace and chat interface
│   ├── index.html                    # Main page shell: split-pane layout (chat left, ReAct trace right)
│   ├── styles.css                    # Styling for chat bubbles, trace cards, choice buttons, and layout
│   └── app.js                        # Frontend logic: sends API requests, renders agent state, handles user choices
│
├── tests/                            # Unit tests for tools and agent logic
│   ├── test_tools.py                 # Tests for search_restaurants, check_availability, make_reservation against mock CSVs
│   └── test_agent.py                 # Tests for the ReAct loop: happy path, unavailable time, alternative selection flow
│
├── requirements.txt                  # Python dependency list: fastapi, uvicorn, pydantic
└── README.md                         # Project overview, setup instructions, and demo scenario descriptions