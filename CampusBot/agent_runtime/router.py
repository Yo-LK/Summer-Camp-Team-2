# runtime/router.py
from typing import Dict, Optional
from skills.base import BaseSkill


class SkillRouter:

    def __init__(self, skills: Dict[str, BaseSkill]):
        self.skills = skills

    def route(self, user_input: str) -> Optional[BaseSkill]:
        """根据用户输入的关键词，自动选择最合适的 Skill"""
        text = user_input.strip().lower()

        if not text:
            return None
        # # 1. 优先匹配翻译技能

        # if any(kw in text for kw in ["translate", "翻译", "英文", "中文"]):
        #     if "translation" in self.skills:
        #         return self.skills["translation"]

        # # 2. 匹配图书馆技能
        # if any(
        #     kw in text
        #     for kw in ["library", "图书馆", "借书", "开馆时间", "馆长"]
        # ):
        #     if "library" in self.skills:
        #         return self.skills["library"]

        # # 3. 默认兜底使用校园通用技能
        # return self.skills.get("campus")

        #now using Skill keyword
        # Specific Skills must be checked before the general Campus Skill.
        priority = ["translation", "library", "campus"]

        for skill_name in priority:
            skill = self.skills.get(skill_name)

            if skill is None:
                continue

            if any(
                keyword.lower() in text
                for keyword in skill.keywords
            ):
                return skill

        # No matching Skill
        return None