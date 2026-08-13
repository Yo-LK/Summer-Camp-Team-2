# skills/campus_skill.py
from typing import Any, Dict, Optional
from .base import BaseSkill


class CampusSkill(BaseSkill):

    def __init__(self):
        super().__init__(
            name="campus",
            # 【4. 修改 Description 与关键词提取】
            description=(
                "处理与深圳大学（深大/SZU）相关的通用校情、校史、校训、建校时间及校区分布等查询。"
                "匹配特征词：深圳大学, 深大, szu, 校训, 建校, 成立时间, 校区, 校长, 概况"
            ),
            keywords=[
                "深圳大学",
                "深大",
                "szu",
                "shenzhen university",
                "校训",
                "成立",
                "建校",
                "校区",
                "校长",
                "大学",
            ],
        )

    def get_system_prompt(self) -> str:
        # 【1. 独立的 System Prompt 指令】
        return (
            "【深圳大学官方信息助手指令】\n"
            "你是一个严谨的深圳大学校园信息助手。\n"
            "1. 请严格仅依据给定的 [参考知识] 回答用户关于深圳大学的问题。\n"
            "2. 如果 [参考知识] 中没有包含回答该问题所需的信息，请直接回答：'抱歉，官方知识库中暂未收录相关信息。'\n"
            "3. 绝对不可编造、推测或捏造任何信息。"
        )

    def process(
        self, user_input: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        input_text = user_input.strip()
        knowledge = (context or {}).get("knowledge", {})
        #campus_data = knowledge.get("campus_info") or {}
        # modify_To use keys like the keys in the knowledfe.json file
        campus_data = knowledge.get("university") or {}


        # 关键词与知识库条目特征提取
        matched_facts = []
        if isinstance(campus_data, dict):
            for key, value in campus_data.items():
                if key in input_text or any(
                    k in input_text for k in key.split()
                ):
                    matched_facts.append(f"• {key}: {value}")

        # 信息缺失兜底（避免 AI 幻觉）
        if not matched_facts:
            return self.format_output(
                status="unavailable",
                response="抱歉，官方知识库中暂未收录相关信息。",
                error_detail="Campus knowledge base lookup miss.",
            )

        context_str = "\n".join(matched_facts)
        final_prompt = (
            f"{self.get_system_prompt()}\n\n"
            f"[参考知识]:\n{context_str}\n\n"
            f"[问题]: {input_text}"
        )

        return self.format_output(status="success", response=final_prompt)