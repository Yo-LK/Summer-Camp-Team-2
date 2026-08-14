from __future__ import annotations

from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.agent.react_loop import ReservationAgent
from backend.agent.tool_router import ToolRouter
from backend.api.routes import router


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def create_app(
    agent: Optional[ReservationAgent] = None,
    data_dir: Optional[Path] = None,
) -> FastAPI:
    load_dotenv(PROJECT_ROOT / ".env")
    app = FastAPI(title="Restaurant Reservation Agent", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.state.agent = agent or ReservationAgent(
        router=ToolRouter(data_dir=data_dir) if data_dir else None
    )
    app.include_router(router)
    return app


app = create_app()
