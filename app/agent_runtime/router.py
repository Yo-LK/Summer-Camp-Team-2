# runtime/router.py
from typing import Dict, Optional

from app.skills.base import BaseSkill


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

        '''
        The original Router duplicated translation and 
        library keywords directly in router.py and sent 
        every unmatched request to the Campus Skill. 
        I updated it to use the keywords defined inside each Skill, 
        so keyword changes are automatically reflected in routing. 
        I also added the priority order Translation → Library → Campus 
        for overlapping matches and changed the default fallback to None 
        so unrelated requests are not incorrectly routed to the Campus Skill.

        '''

        '''
        [Known Limitation]
        Known limitation: The current Router uses deterministic keyword-based rules. It is simple, testable, and works without additional LLM calls, but it may fail to recognize paraphrased, ambiguous, or multilingual requests when no configured keyword matches. A future version could use an LLM-based intent classifier as a fallback.
        '''

        # No configured keyword matched.
        return None
