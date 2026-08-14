from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseSkill(ABC):
    """
    EN: Abstract base class for all Agent skills, defining input/output contracts and keyword configuration.
    KO: 모든 에이전트 스킬의 추상 기반 클래스로, 입출력 계약 및 키워드 구성을 정의합니다.
    ZH: 所有 Agent 技能的抽象基类，规范输入、输出契约与关键词配置。
    """
      
    def __init__(
        self,
        name: str,
        description: str,
        keywords: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        # EN: Characteristic keywords for Skill Router extraction and matching.
        # KO: Skill Router가 추출 및 매칭하는 데 사용하는 특징 키워드입니다.
        # ZH: 用于 Skill Router 提取与匹配的特征关键词
        self.keywords = keywords or []  # 用于 Skill Router 提取与匹配的特征关键词

    @abstractmethod
    def get_system_prompt(self) -> str:
        # EN: [1. Independent instruction behavior] Retrieve the specific System Prompt for the current skill.
        # KO: [1. 독립적인 지시 행동] 현재 스킬 전용 시스템 프롬프트를 가져옵니다.
        # ZH: 【1. 独立的指令行为】获取当前技能专属的 System Prompt
        pass

    @abstractmethod
    def process(
        self, user_input: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        EN: [2. Define clear input]
        Input:
            user_input (str): Raw user input text
            context (Dict[str, Any], optional): External information including knowledge base knowledge

        [3. Define clear output]
        Output:
            Dict[str, Any]:
            {
                "status": "success" | "unavailable" | "error" | "blocked",
                "skill": str,
                "response": str,
                "error_detail": Optional[str]
            }
        KO: [2. 명확한 입력 정의]
        Input:
            user_input (str): 사용자 원시 입력 텍스트
            context (Dict[str, Any], optional): 지식 베이스(knowledge) 등을 포함한 외부 정보

        [3. 명확한 출력 정의]
        Output:
            Dict[str, Any]:
            {
                "status": "success" | "unavailable" | "error" | "blocked",
                "skill": str,
                "response": str,
                "error_detail": Optional[str]
            }
        ZH: 【2. 定义清晰的输入】
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
        # EN: Helper function to standardize output structure.
        # KO: 출력 구조를 표준화하기 위한 도우미 함수입니다.
        # ZH: 统一规范输出结构的辅助函数
        return {
            "status": status,
            "skill": self.name,
            "response": response,
            "error_detail": error_detail,
        }