# skills/library_skill.py
from typing import Any, Dict, Optional
from .base import BaseSkill


class LibrarySkill(BaseSkill):

    def __init__(self):
        super().__init__(
            name="library",
            # 【4. 修改 Description 与关键词提取】
            description=(
                "查询深圳大学图书馆的服务信息，包含位置分布、开馆闭馆时间、借阅规则及馆藏分布。"
                "匹配特征词：图书馆, 图书, 借书, 还书, 开馆时间, 闭馆, 阅览室, library"
            ),
            keywords=[
                "图书馆",
                "图书",
                "借书",
                "还书",
                "开馆",
                "闭馆",
                "阅览室",
                "馆藏",
                "library",
            ],
        )

    def get_system_prompt(self) -> str:
        # 【1. 独立的 System Prompt 指令】
        return (
            "【图书馆小助手指令】\n"
            "你是深圳大学图书馆服务助手。\n"
            "1. 请明确、具体地依据 [图书馆数据] 回答有关地理位置、开馆时间及借阅服务的问题。\n"
            "2. 保持回答清晰简明。"
        )

    def process(
        self, user_input: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        input_text = user_input.strip()
        knowledge = (context or {}).get("knowledge", {})
        lib_data = knowledge.get("library") #modify
        #Updated only the knowledge lookup keys to match the actual knowledge.json schema:
        #campus_info → university and library_info → library.
        #The original Skill structure, prompts, interfaces, and failure behavior were preserved.

        if not lib_data:
            return self.format_output(
                status="unavailable",
                response="抱歉，暂未查找到图书馆的相关服务信息。",
                error_detail="Library information not present in knowledge.",
            )

        final_prompt = (
            f"{self.get_system_prompt()}\n\n"
            f"[图书馆数据]: {lib_data}\n\n"
            f"[问题]: {input_text}"
        )

        return self.format_output(status="success", response=final_prompt)