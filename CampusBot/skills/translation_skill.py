# skills/translation_skill.py
from typing import Any, Dict, Optional
from .base import BaseSkill


class TranslationSkill(BaseSkill):

    def __init__(self):
        super().__init__(
            name="translation",
            # 【4. 修改 Description 与关键词提取】
            description=(
                "提供中英文双向准确翻译服务，无需查阅校园知识库。"
                "匹配特征词：翻译, translate, 英文, 中文, 英语, 汉语, 翻译成"
            ),
            keywords=[
                "翻译",
                "translate",
                "英文",
                "中文",
                "英语",
                "汉语",
                "translation",
            ],
        )

    def get_system_prompt(self) -> str:
        # 【1. 独立的 System Prompt 指令】
        return (
            "【精准翻译助手指令】\n"
            "你是一个精准的双语翻译助手。\n"
            "1. 如果输入为英文，请翻译为中文；如果输入为中文，请翻译为英文。\n"
            "2. 只输出最终翻译结果，绝对不要输出任何额外的开场白或解释说明。"
        )

    def process(
        self, user_input: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        input_text = user_input.strip()

        if not input_text:
            return self.format_output(
                status="error",
                response="输入的待翻译文本不能为空。",
                error_detail="Empty input text provided to TranslationSkill.",
            )

        final_prompt = f"{self.get_system_prompt()}\n\n[需要翻译的文本]: {input_text}"

        return self.format_output(status="success", response=final_prompt)