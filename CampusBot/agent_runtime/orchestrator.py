# runtime/orchestrator.py
from typing import Any, Dict, Optional
from skills.base import BaseSkill
from .router import SkillRouter


class RuntimeOrchestrator:

    def __init__(self, router: SkillRouter, skills: Dict[str, BaseSkill]):
        self.router = router
        self.skills = skills

    def handle_request(
        self, user_input: str, context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """接收请求 -> 路由选择 -> 执行技能 -> 返回统一格式"""
        context = context or {}
        #add
        if not user_input or not user_input.strip():
            return {
                "status": "error",
                "skill": "none",
                "response": "Message cannot be empty.",
                "error_detail": "Empty user input.",
            }
        
        # 1. 自动选择合适的 Skill
        selected_skill = self.router.route(user_input)

        # if not selected_skill:
        #     return {
        #         "status": "error",
        #         "skill": "unknown",
        #         "response": "未能匹配到合适的处理技能。",
        #     }

        # # 2. 执行选中的 Skill
        # result = selected_skill.process(user_input, context)

        # # 3. 补充 Runtime 层的元数据
        # result["routed_skill"] = selected_skill.name
        # return result

        if selected_skill is None:
            return {
                "status": "unmatched",
                "skill": "none",
                "response": (
                    "No suitable skill was found for this request."
                ),
                "error_detail": None,
            }

        try:
            result = selected_skill.process(
                user_input,
                context,
            )

            result["routed_skill"] = selected_skill.name

            return result

        except Exception as exc:
            return {
                "status": "error",
                "skill": selected_skill.name,
                "response": (
                    "The selected skill could not process the request."
                ),
                "error_detail": str(exc),
                "routed_skill": selected_skill.name,
            }
        