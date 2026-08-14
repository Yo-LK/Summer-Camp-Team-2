from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.agent_runtime import create_runtime
from app.api.routes import router
from app.services.ollama_client import load_knowledge
from app.skills import get_skill_registry
from config.config import settings


runtime = create_runtime(get_skill_registry())

app = FastAPI(title="CampusBot Starter", version="0.1.0")
app.state.runtime = runtime
app.state.knowledge = load_knowledge()
app.mount("/static", StaticFiles(directory=str(settings.web_root)), name="static")
app.include_router(router)

