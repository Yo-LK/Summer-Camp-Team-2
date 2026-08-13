# skills/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseSkill(ABC):
    """所有 Agent 技能的抽象基类，规范输入、输出契约与关键词配置"""

    def __init__(
        self,
        name: str,
        description: str,
        keywords: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.keywords = keywords or []  # 用于 Skill Router 提取与匹配的特征关键词

    @abstractmethod
    def get_system_prompt(self) -> str:
        """【1. 独立的指令行为】获取当前技能专属的 System Prompt"""
        pass

    @abstractmethod
    def process(
        self, user_input: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """【2. 定义清晰的输入】
        Input:
            user_input (str): 用户原始输入文本
            context (Dict[str, Any], optional): 包含知识库知识（knowledge）等外部信息

        【3. 定义清晰的输出】
        Output:
            Dict[str, Any]:
            {
                "status": "success" | "unavailable" | "error" | "blocked",
                "skill": str,
                "response": str,
                "error_detail": Optional[str]
            }
        """
        pass

    def format_output(
        self, status: str, response: str, error_detail: Optional[str] = None
    ) -> Dict[str, Any]:
        """统一规范输出结构的辅助函数"""
        return {
            "status": status,
            "skill": self.name,
            "response": response,
            "error_detail": error_detail,
        }