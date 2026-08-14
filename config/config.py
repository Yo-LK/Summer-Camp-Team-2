from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_root: Path
    app_root: Path
    web_root: Path
    knowledge_root: Path
    logs_root: Path
    ollama_url: str
    ollama_model: str
    system_prompt_path: Path

    @classmethod
    def load(cls) -> "Settings":
        project_root = Path(__file__).resolve().parents[1]
        app_root = project_root / "app"
        knowledge_root = project_root / "knowledge"
        web_root = project_root / "web"
        logs_root = project_root / "logs"
        logs_root.mkdir(exist_ok=True)

        return cls(
            project_root=project_root,
            app_root=app_root,
            web_root=web_root,
            knowledge_root=knowledge_root,
            logs_root=logs_root,
            ollama_url=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat"),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen3:0.6b"),
            system_prompt_path=project_root / "prompt.txt",
        )


settings = Settings.load()
