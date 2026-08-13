# skills/campus_skill.py
from typing import Any, Dict, Optional
from .base import BaseSkill


class CampusSkill(BaseSkill):

    def __init__(self):
        super().__init__(
            name="campus",
            # EN: [1. Clarify responsibility: History, architecture, landmark figures, and culture of SZU]
            # KO: [1. 책임 명확화: 심천대학의 역사, 건축, 명사 및 문화 관련 질의 처리]
            # ZH: 【1. 明确职责：负责解答深圳大学的历史、建筑、校园文化与校友/名人相关问题】
            description=(
                "Handles queries related to Shenzhen University (SZU) history, architectural design, landmarks, "
                "notable alumni/professors, school culture, motto, and overall campus heritage. "
                "Matched keywords: shenzhen university, szu, history, architecture, building, alumni, culture, landmark, motto."
            ),
            keywords=[
                "shenzhen university",
                "szu",
                "history",
                "architecture",
                "building",
                "building name",
                "landmark",
                "alumni",
                "famous figure",
                "professor",
                "culture",
                "motto",
                "campus",
            ],
        )

    def get_system_prompt(self) -> str:
        # EN: [2. System Prompt: Professional guide persona, step-by-step reasoning, bilingual retention, and polite tone]
        # KO: [2. 시스템 프롬프트: 전문 안내원 페르소나, 단계별 추론, 이중 언어 보존 및 품격 있는 어조]
        # ZH: 【2. System Prompt：优雅风度的深大讲解员角色、逐步推理、英文保留与彬彬有礼的回答】
        return (
            "[Shenzhen University Professional Campus Guide Instructions]\n"
            "Role & Persona:\n"
            "You are an elegant, patient, and highly articulate official tour guide of Shenzhen University (SZU). "
            "Your speech is polished, welcoming, and refined (符合有风度的讲解员，语气富有耐性，言语得体).\n\n"
            "Core Guidelines:\n"
            "1. Factuality & Honesty: Answer questions about SZU culture, history, buildings, and notable figures truthfully based on the [Reference Knowledge]. "
            "If information is missing, politely and tactfully express that you do not know (e.g., '十分抱歉，关于这一细节，目前记录库中暂未收录……').\n"
            "2. Step-by-Step Reasoning (CoT): Think step-by-step before answering to analyze the user's focus (e.g., historical timeline, architectural significance, or figure background).\n"
            "3. English Retention: Keep essential English descriptions, names, or terminology alongside Chinese responses (输出语句需要保留英文表达或英文对照).\n"
            "4. Exceptional Guidance: In cases of error or ambiguous input, offer graceful guidance and hints without stopping the reasoning process abruptly."
        )

    def process(
        self, user_input: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        # EN: [3. Clear Input Definition: Validate and clean input text and context structured dict]
        # KO: [3. 명확한 입력 정의: 입력 텍스트 및 컨텍스트 구조 검증 및 정제]
        # ZH: 【3. 定义清晰的输入：对 user_input 与 context 进行校验与类型限定】
        if not user_input or not isinstance(user_input, str):
            # EN: [5. Exception Guidance: Handle empty or invalid input gracefully without stopping thinking]
            # KO: [5. 예외 상황 안내: 생각 프로세스를 임의로 중단하지 않고 친절하게 입력 유도]
            # ZH: 【5. 异常情况引导：对空输入/无效输入提供友好的讲解员式引导，不可随意停止】
            return self.format_output(
                status="invalid_input",
                response=(
                    "您好！我是深圳大学的校园文化讲解员。请问有什么关于深大的历史、古今建筑或校园名人的问题想要了解吗？"
                    " (Hello! I am your SZU campus guide. Please feel free to ask any questions about Shenzhen University's history, architecture, or notable figures.)"
                ),
                error_detail="User input is empty or not a valid string.",
            )

        input_text = user_input.strip()
        knowledge = (context or {}).get("knowledge", {})
        campus_data = knowledge.get("university") or {}

        '''
        I modified the code because the actual knowledge.
        json uses university and library instead of campus_info and library_info. 
        The original code also required JSON field names to appear directly in the question.
        For example, founded and established have similar meanings but do not match as strings. 
        Therefore, I changed it to pass the selected Skill’s relevant data section to the LLM.

        '''
        # EN: Match campus facts (history, architecture, figures, culture) from knowledge base.
        # KO: 지식 베이스에서 캠퍼스 정보(역사, 건축, 인물, 문화) 매칭.
        # ZH: 从知识库检索深大历史、建筑、名人、文化相关的匹配条目
        matched_facts = []
        if isinstance(campus_data, dict):
            for key, value in campus_data.items():
                #if key in input_text or any(k in input_text for k in key.split()):
                matched_facts.append(f"• {key}: {value}")

        # EN: [5. Exception & Missing Info Handling: Tactful guidance when information is unavailable]
        # KO: [5. 예외 및 정보 부족 처리: 정보를 찾을 수 없을 때 우아하고 체계적인 유도]
        # ZH: 【5. 异常与信息不可用处理：未查到知识时以讲解员口吻委婉回复并提供进一步探寻建议，保留英文】
        if not matched_facts:
            return self.format_output(
                status="unavailable",
                response=(
                    "十分抱歉，关于您提到的这一点，我目前的导览资料库中暂未记载相关的详细信息。"
                    "不过您可以关注深大学术史或官方档案馆的最新发布。若您对荔园的标志性建筑（如粤海校区/丽湖校区）或创校历史感兴趣，我很乐意为您继续讲解！\n"
                    "(Regrettably, the requested detail is currently unavailable in my guided knowledge repository. "
                    "Please feel free to ask about SZU's iconic landmarks, campuses, or founding history, and I would be delighted to assist you.)"
                ),
                error_detail="Campus knowledge base lookup miss.",
            )

        context_str = "\n".join(matched_facts)
        
        # EN: [4. Clear Output Construction: Assembling step-by-step reasoning prompt and response rules]
        # KO: [4. 명확한 출력 구조화: 단계별 추론 프롬프트 및 응답 규칙 조립]
        # ZH: 【4. 格式化输出构造：结合 CoT 逐步思考提示词与保留英文指令】
        final_prompt = (
            f"{self.get_system_prompt()}\n\n"
            f"[Reference Knowledge / 参考资料]:\n{context_str}\n\n"
            f"[User Question / 访客提问]: {input_text}\n\n"
            f"[Guidance Task / 讲解任务]:\n"
            f"1. Think step-by-step (逐步思考分析)：先厘清提问涉及的是历史、建筑还是名人，提取资料中的核心事实。\n"
            f"2. Formulate a polite, elegant tour-guide style response (风度有礼的讲解员回复)。\n"
            f"3. Ensure key information retains English translation or terminology (保留英文对应表达)。"
        )

        return self.format_output(status="success", response=final_prompt)