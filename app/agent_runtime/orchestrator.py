# runtime/orchestrator.py
import re
from typing import Any, Dict, List, Optional

from app.governance import check_request
from app.skills.base import BaseSkill

from .router import SkillRouter


class RuntimeOrchestrator:

    def __init__(self, router: SkillRouter, skills: Dict[str, BaseSkill]):
        self.router = router
        self.skills = skills

    def _detect_output_language(self, user_input: str) -> Optional[str]:
        text = user_input.lower()

        if any(token in text for token in ["in chinese", "中文", "to chinese", "用中文", "以中文"]):
            return "Chinese"
        if any(token in text for token in ["in english", "英文", "to english", "用英语", "以英语"]):
            return "English"
        if any(token in text for token in ["in japanese", "日文", "to japanese", "用日语", "以日语"]):
            return "Japanese"
        if any(token in text for token in ["in spanish", "西班牙语", "to spanish", "用西班牙语"]):
            return "Spanish"
        if any(token in text for token in ["in french", "法语", "to french", "用法语"]):
            return "French"

        return None

    def _strip_output_language_instruction(self, user_input: str) -> str:
        cleaned = user_input.strip()
        patterns = [
            r"\s*return the response in [a-zA-Z\u4e00-\u9fff]+\.?",
            r"\s*answer in [a-zA-Z\u4e00-\u9fff]+\.?",
            r"\s*translate(\s+the\s+response)?\s+to\s+[a-zA-Z\u4e00-\u9fff]+\.?",
            r"\s*in [a-zA-Z\u4e00-\u9fff]+\.?$",
            r"\s*用中文\.?$",
            r"\s*用英语\.?$",
            r"\s*用日语\.?$",
        ]
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        return " ".join(cleaned.split())

    def _build_workflow(self, user_input: str) -> List[Dict[str, Any]]:
        target_language = self._detect_output_language(user_input)
        factual_input = self._strip_output_language_instruction(user_input)
        if not factual_input:
            factual_input = user_input.strip()

        base_skill = self.router.route(factual_input)
        if base_skill is None:
            base_skill = self.skills.get("campus")
        if base_skill is None:
            return []

        workflow: List[Dict[str, Any]] = [{
            "type": "fact",
            "skill": base_skill.name,
            "text": factual_input,
            "target_language": None,
        }]

        if target_language and base_skill.name != "translation":
            workflow.append({
                "type": "transform",
                "skill": "translation",
                "source": "previous",
                "target_language": target_language,
                "text": "",
            })

        return workflow

    def handle_request(
        self, user_input: str, context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """接收请求 -> 安全校验 -> 任务分解 -> 顺序执行技能 -> 返回统一格式"""
        context = context or {}
        if not user_input or not user_input.strip():
            return {
                "status": "error",
                "skill": "none",
                "response": "Message cannot be empty.",
                "error_detail": "Empty user input.",
            }

        guardrail_result = check_request(user_input)
        if not guardrail_result.allowed:
            return {
                "status": "blocked",
                "skill": "none",
                "response": "Request blocked.",
                "error_detail": guardrail_result.reason,
            }

        workflow = self._build_workflow(user_input)
        if not workflow:
            return {
                "status": "unmatched",
                "skill": "none",
                "response": "No suitable skill was found for this request.",
                "error_detail": None,
            }

        executed_steps: List[Dict[str, Any]] = []
        final_text = ""
        final_status = "success"
        last_skill_name = workflow[0]["skill"]
        last_successful_response = ""

        try:
            for step in workflow:
                skill_name = step["skill"]
                skill = self.skills.get(skill_name)
                if skill is None:
                    return {
                        "status": "unmatched",
                        "skill": "none",
                        "response": f"Skill '{skill_name}' is not registered.",
                        "error_detail": None,
                        "workflow": executed_steps,
                    }

                if step.get("type") == "transform":
                    if step.get("source") != "previous":
                        raise ValueError("Transform steps must consume the previous step output.")
                    if not final_text:
                        raise ValueError("No previous output available to transform.")
                    step_input = final_text
                else:
                    step_input = step.get("text", "")

                skill_context = dict(context)
                if step.get("target_language"):
                    skill_context["target_language"] = step["target_language"]

                result = skill.process(step_input, skill_context)
                executed_steps.append({
                    "type": step.get("type"),
                    "skill": skill.name,
                    "input": step_input,
                    "status": result.get("status"),
                    "response": result.get("response"),
                    "target_language": step.get("target_language"),
                })

                final_status = result.get("status", "success")
                last_skill_name = skill.name
                final_text = result.get("response") or final_text
                last_successful_response = result.get("response") or last_successful_response

                if result.get("status") not in {"success"}:
                    return {
                        "status": result.get("status", "error"),
                        "skill": skill.name,
                        "response": result.get("response") or "The selected skill could not process the request.",
                        "error_detail": result.get("error_detail"),
                        "routed_skill": " -> ".join(step["skill"] for step in workflow if step.get("skill")),
                        "workflow": executed_steps,
                    }

            routed_skill = " -> ".join(step["skill"] for step in workflow)
            return {
                "status": final_status,
                "skill": last_skill_name,
                "response": last_successful_response or final_text,
                "error_detail": None,
                "routed_skill": routed_skill,
                "workflow": executed_steps,
            }

        except Exception as exc:
            return {
                "status": "error",
                "skill": last_skill_name,
                "response": "The selected skill could not process the request.",
                "error_detail": str(exc),
                "routed_skill": " -> ".join(step["skill"] for step in workflow),
                "workflow": executed_steps,
            }
