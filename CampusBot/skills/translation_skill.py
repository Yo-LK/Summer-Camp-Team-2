# skills/translation_skill.py
from typing import Any, Dict, Optional
from .base import BaseSkill


class TranslationSkill(BaseSkill):

    def __init__(self):
        super().__init__(
            name="translation",
            # EN: [1. Clarify responsibility: Multi-language precise translation, style adaptation (spoken vs. formal), and linguistic query handling]
            # KO: [1. 책임 명확화: 다국어 정밀 번역, 구어체/문어체 스타일 맞춤 설정 및 신조어/모호한 구문 문의 처리]
            # ZH: 【1. 明确职责：负责多国语言精准地道翻译，支持日常口语与严谨书面语风格切换，并对模糊语法/词汇主动提问】
            description=(
                "Provides precise, natural, and authentic translation services across multiple global languages. "
                "Dynamically adapts tone to casual spoken register or rigorous formal/academic written style based on user requirements or context. "
                "Matched keywords: translate, translation, interpreter, multilingual, oral, spoken, formal, written, english, chinese, japanese, french, spanish."
            ),
            keywords=[
                "translate",
                "translation",
                "interpreter",
                "multilingual",
                "oral",
                "spoken",
                "formal",
                "written",
                "english",
                "chinese",
                "japanese",
                "french",
                "spanish",
                "german",
            ],
        )

    def get_system_prompt(self) -> str:
        # EN: [2. System Prompt: Chief Translator persona, step-by-step reasoning, bilingual retention, and courteous tone]
        # KO: [2. 시스템 프롬프트: 수석 번역가 페르소나, 단계별 추론, 이중 언어 보존 및 품격 있는 어조]
        # ZH: 【2. System Prompt：优雅耐性的深大首席翻译官角色、逐步推理、风格判定、保留原词与英文输出】
        return (
            "[Shenzhen University Chief Translator Instructions]\n"
            "Role & Persona:\n"
            "You are the Chief Translator of Shenzhen University (SZU). "
            "Your tone is exceptionally patient, polished, academic, and courteous (语气富有耐性，言语得体，符合有风度的首席翻译官).\n\n"
            "Core Guidelines:\n"
            "1. Precise & Authentic Translation: Translate each sentence accurately and natively. If encountering unknown terms, slang, or ambiguous grammar, politely inquire about the user's intended meaning while temporarily retaining the original words in brackets e.g., '[original term]' to leave room for correction.\n"
            "2. Contextual Reasoning (CoT): Think step-by-step to analyze the required register before translating:\n"
            "   - Daily/Casual Conversation: Use a lighthearted, natural, and authentic spoken tone (日常交流偏口语化和轻快).\n"
            "   - Formal Scenarios (History, Science, Academic Knowledge Transfer): Use rigorous, precise, and scientific written expressions (关键场合偏向严谨、科学的书面表达).\n"
            "3. English Retention: Always keep clear English explanations, bilingual terminology, or English translations in your final output (输出语句必须保留英文表达或对照).\n"
            "4. Exception & Language Mixture Handling: If the input contains mixed languages or unrecognized scripts, offer patient guidance, point out the ambiguous parts, and ask for clarification gently without abruptly stopping thought."
        )

    def process(
        self, user_input: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        # EN: [3. Clear Input Definition: Validate and extract target text, register preference, and context]
        # KO: [3. 명확한 입력 정의: 입력 텍스트, 스타일 선호도 및 컨텍스트 구조 검증 및 정제]
        # ZH: 【3. 定义清晰的输入：校验 user_input 类型，并从 context 中提取目标语言和文体偏好】
        if not user_input or not isinstance(user_input, str):
            # EN: [5. Exception Guidance: Handle empty or invalid input gracefully with Chief Translator persona]
            # KO: [5. 예외 상황 안내: 생각 프로세스를 임의로 중단하지 않고 수석 번역가의 입장에서 친절하게 입력 유도]
            # ZH: 【5. 异常情况引导：对空输入/无效输入提供首席翻译官式的友好引导，不可随意停止】
            return self.format_output(
                status="invalid_input",
                response=(
                    "您好！我是深圳大学的首席翻译官。请问有什么文本或口语需要我为您翻译吗？您可以告诉我希望偏向日常口语还是严谨的书面表达。"
                    " (Hello! I am the Chief Translator of SZU. How may I assist you with translation today? Please feel free to specify whether you prefer a casual spoken tone or a formal written style.)"
                ),
                error_detail="User input is empty or not a valid string.",
            )

        input_text = user_input.strip()
        ctx = context or {}
        target_language = ctx.get("target_language", "Auto-detect / 自动检测")
        register_style = ctx.get("register_style", "Auto-infer / 自动推断")  # e.g., 'casual', 'formal'

        # EN: [5. Exception Guidance: Check for mixed language or unrecognized input anomalies]
        # KO: [5. 예외 상황 안내: 혼합 언어 또는 인식할 수 없는 텍스트 입력 처리]
        # ZH: 【5. 异常情况引导：针对多语言参杂或难以辨识的文本，提供翻译官口吻的专业询问与引导】
        # (Pass through to LLM with guidance instructions so reasoning never stops abruptly)
        
        # EN: [4. Clear Output Construction: Assembling step-by-step reasoning prompt and response rules]
        # KO: [4. 명확한 출력 구조화: 단계별 추론 프롬프트 및 응답 규칙 조립]
        # ZH: 【4. 格式化输出构造：结合 CoT 逐步思考提示词、风格判定与保留英文指令】
        final_prompt = (
            f"{self.get_system_prompt()}\n\n"
            f"[Translation Task Context / 翻译上下文]:\n"
            f"• Target Language Preference / 目标语言: {target_language}\n"
            f"• Desired Style Register / 风格偏好: {register_style}\n\n"
            f"[Source Text / 待翻译文本]:\n{input_text}\n\n"
            f"[Chief Translator Workflow / 翻译官工作流]:\n"
            f"1. Step-by-Step Reasoning (逐步思考与推理)：分析源文本语种、语法结构及上下文场景（判别是日常轻快口语，还是严谨科学的书面表达）。\n"
            f"2. Terminology & Uncertainty Check (疑难词汇检查)：如遇到不确定的词汇或混杂语法，暂留原文如 '[Original]' 并以礼貌语气向提问者请教。\n"
            f"3. Deliver Response (输出翻译)：以富有耐性、有风度的首席翻译官口吻呈现优质翻译，并保留英文对照。"
        )

        return self.format_output(status="success", response=final_prompt)