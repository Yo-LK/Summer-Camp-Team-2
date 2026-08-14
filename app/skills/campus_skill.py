from typing import Any, Dict, Optional
from app.services.ollama_client import ask_model, get_last_llm_payload
from .base import BaseSkill
from .language_utils import infer_source_language


class CampusSkill(BaseSkill):

    def __init__(self):
        super().__init__(
            name="campus",
            # EN: [1. Clarify responsibility: History, architecture, landmark figures, president, and culture of SZU]
            # KO: [1. 책임 명확화: 심천대학의 역사, 건축, 명사, 총장 및 문화 관련 질의 처리]
            # ZH: 【1. 明确职责：负责解答深圳大学的历史、建筑、校园文化、校长与校友/名人相关问题】
            description=(
                "Handles queries related to Shenzhen University (SZU) history, architectural design, landmarks, "
                "notable alumni/professors, president, school culture, motto, and overall campus heritage. "
                "Matched keywords: shenzhen university, szu, history, architecture, building, alumni, culture, landmark, motto, president."
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
                "president",
                "校长",
                "culture",
                "motto",
                "campus",
            ],
        )

    def get_system_prompt(self) -> str:
        # EN: [2. System Prompt: Professional guide persona, step-by-step reasoning, user language matching, and polite tone]
        # KO: [2. 시스템 프롬프트: 전문 안내원 페르소나, 단계별 추론, 질문 언어 맞춤 및 품격 있는 어조]
        # ZH: 【2. System Prompt：优雅风度的深大讲解员角色、逐步推理、提问语言自适应与彬彬有礼的回答】
        return (
            "[Shenzhen University Professional Campus Guide Instructions]\n"
            "Role & Persona:\n"
            "You are an elegant, patient, and highly articulate official tour guide of Shenzhen University (SZU). "
            "Your speech is polished, welcoming, and refined (符合有风度的讲解员，语气富有耐性，言语得体).\n\n"
            "Core Guidelines:\n"
            "1. Factuality & Honesty: Answer questions about SZU culture, history, buildings, president, and notable figures strictly based on the [Reference Knowledge]. "
            "Never fabricate or guess answers (e.g., if asked about the president and data is absent, do not guess). If information is missing, tactful and truthful admission of not knowing in the user's language is mandatory.\n"
            "2. Language Consistency Rule (语言一致性规则): You MUST detect and respond in the EXACT SAME LANGUAGE as the user's question (e.g., respond in Chinese if asked in Chinese, respond in English if asked in English, etc.).\n"
            "3. Step-by-Step Reasoning (CoT): Think step-by-step before answering to analyze the user's focus (e.g., historical timeline, architectural significance, leadership/figures, or figure background).\n"
            "4. Exceptional Guidance: In cases of error, missing knowledge, or ambiguous input, offer graceful guidance and hints without stopping the reasoning process abruptly."
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
                    "您好！我是深圳大学的校园文化讲解员。请问有什么关于深大的历史、古今建筑、历任校长或校园名人的问题想要了解吗？"
                    " (Hello! I am your SZU campus guide. Please feel free to ask any questions about Shenzhen University's history, architecture, president, or notable figures.)"
                ),
                error_detail="User input is empty or not a valid string.",
            )

        input_text = user_input.strip()
        normalized_input = input_text.lower()
        source_language = infer_source_language(input_text)
        target_language = source_language
        knowledge = (context or {}).get("knowledge", {})
        campus_data = knowledge.get("university") or {}

        # EN: Match campus facts (history, architecture, figures, culture) from knowledge base.
        # KO: 지식 베이스에서 캠퍼스 정보(역사, 건축, 인물, 문화) 매칭.
        # ZH: 从知识库检索深大历史、建筑、名人、文化相关的匹配条目
        matched_facts = []
        semantic_aliases = {
            "established": ["founded", "founding", "establish", "when founded", "成立", "创立", "建校"],
            "campuses": ["campus", "campuses", "校区"],
            "motto": ["motto", "校训"],
            "abbreviation": ["abbreviation", "abbr", "简称"],
            "name": ["name", "full name", "校名"],
        }
        if isinstance(campus_data, dict):
            for key, value in campus_data.items():
                key_lower = key.lower()
                alias_terms = semantic_aliases.get(key_lower, [])
                if (
                    key_lower in normalized_input
                    or any(k in normalized_input for k in key_lower.split())
                    or any(alias in normalized_input for alias in alias_terms)
                ):
                    matched_facts.append(f"• {key}: {value}")

        # EN: [5. Exception & Missing Info Handling: Tactful guidance when information is unavailable]
        # KO: [5. 예외 및 정보 부족 처리: 정보를 찾을 수 없을 때 우아하고 체계적인 유도]
        # ZH: 【5. 异常与信息不可用处理：未查到知识时以讲解员口吻委婉回复不可知，严禁捏造，用提问语言解答】
        if not matched_facts:
            return self.format_output(
                status="unavailable",
                response=(
                    "十分抱歉，关于您提到的这一点，我目前的导览资料库中暂未收录相关的详细信息，因此无法为您确切解答，请您谅解。"
                    "不过您可以关注深圳大学官方档案馆或官网发布的最新信息。若您对荔园的标志性建筑（如粤海校区/丽湖校区）或创校历史感兴趣，我很乐意为您继续讲解！\n"
                    "(Regrettably, the requested detail is currently unavailable in my guided knowledge repository, so I cannot provide a definitive answer at this moment. "
                    "Please feel free to check SZU's official updates, or ask about our iconic landmarks, campuses, or founding history—I would be delighted to assist you.)"
                ),
                error_detail="Campus knowledge base lookup miss.",
            )

        context_str = "\n".join(matched_facts)
        
        # EN: [4. Clear Output Construction: Assembling step-by-step reasoning prompt and response rules]
        # KO: [4. 명확한 출력 구조화: 단계별 추론 프롬프트 및 응답 규칙 조립]
        # ZH: 【4. 格式化输出构造：结合 CoT 逐步思考提示词与语言自适应指令】
        final_prompt = (
            f"{self.get_system_prompt()}\n\n"
            f"[Language Control / 语言控制]:\n"
            f"• Source Language / 源语言: {source_language}\n"
            f"• Target Language / 目标语言: {target_language}\n"
            f"• Rule / 规则: Translation skill is NOT used. You MUST respond in the original source language.\n\n"
            f"[Reference Knowledge / 参考资料]:\n{context_str}\n\n"
            f"[User Question / 访客提问]: {input_text}\n\n"
            f"[Guidance Task / 讲解任务]:\n"
            f"1. Match User Language (提问语言匹配)：首先识别访客提问所使用的语言，全篇回答必须严格使用该语言。\n"
            f"2. Think step-by-step (逐步思考分析)：厘清提问涉及的是历史、建筑、校长还是名人，严格对照参考资料提取核心事实。若资料库未提及相关信息，必须明确并委婉表达不知道，严禁编造或捏造答案。\n"
            f"3. Formulate Response (输出回复)：以风度有礼的讲解员口吻解答。"
        )

        answer = ask_model(final_prompt, skill_name=self.name)
        output = self.format_output(status="success", response=answer)

        if (context or {}).get("debug_prompt"):
            payload = get_last_llm_payload(self.name)
            output["debug"] = {
                "skill": self.name,
                "source_language": source_language,
                "target_language": target_language,
                "skill_prompt": final_prompt,
                "llm_system_message": payload.get("messages", [{}])[0].get("content", ""),
                "llm_user_message": payload.get("messages", [{}, {}])[1].get("content", ""),
                "model": payload.get("model", ""),
            }

        return output