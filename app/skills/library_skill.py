from typing import Any, Dict, Optional
from app.services.ollama_client import (
    append_non_llm_prompt_log,
    ask_model,
    get_last_llm_payload,
)
from .base import BaseSkill
from .language_utils import infer_source_language


class LibrarySkill(BaseSkill):

    def __init__(self):
        super().__init__(
            name="library",
            # EN: [1. Clarify responsibility: SZU Library collection, campuses library types/count, opening hours, booking, and visiting]
            # KO: [1. 책임 명확화: 심천대학 도서관 장서, 캠퍼스별 도서관 종류 및 수, 개관 시간, 예약 및 방문 문의 처리]
            # ZH: 【1. 明确职责：负责解答深大图书馆藏书、各校区图书馆种类与数量、开放时间、座位/场地预约及参观指引】
            description=(
                "Handles queries related to Shenzhen University (SZU) Library system, including book collections, "
                "types and counts of libraries across campuses (Yuehai and Lihu), opening hours, seat/space booking methods, and campus visit regulations. "
                "Matched keywords: library, szu library, book collection, opening hours, booking, reserve seat, visit library, yuehai library, lihu library."
            ),
            keywords=[
                "library",
                "szu library",
                "book",
                "collection",
                "opening hours",
                "open time",
                "booking",
                "reservation",
                "reserve seat",
                "visit",
                "visitor",
                "yuehai library",
                "lihu library",
                "curator",
                "图书馆"
            ],
        )

    def get_system_prompt(self) -> str:
        # EN: [2. System Prompt: Professional librarian persona, step-by-step reasoning, user language matching, and courteous tone]
        # KO: [2. 시스템 프롬프트: 전문 사서 페르소나, 단계별 추론, 질문 언어 맞춤, 품격 있는 어조]
        # ZH: 【2. System Prompt：优雅有风度的深大图书管理员角色、逐步推理、提问语言自适应与耐心得体的回答】
        return (
            "[Shenzhen University Professional Librarian Instructions]\n"
            "Role & Persona:\n"
            "You are a courteous, scholarly, and patient official librarian of Shenzhen University (SZU) Library. "
            "Your tone is warm, polite, and refined (符合有风度的图书管理员，语气富有耐性，言语得体).\n\n"
            "Core Guidelines:\n"
            "1. Factuality & Honesty: Provide accurate information regarding SZU library collections, campus library distributions (e.g., Yuehai & Lihu campuses), opening schedules, seat/space reservations, and visitor policies strictly based on the [Reference Knowledge]. "
            "Never fabricate or guess answers. If information is missing or unavailable, tactfully and truthfully express that you do not know in the user's language.\n"
            "2. Language Consistency Rule (语言一致性规则): You MUST detect and respond in the EXACT SAME LANGUAGE as the user's question (e.g., respond in Chinese if asked in Chinese, respond in English if asked in English, etc.).\n"
            "3. Step-by-Step Reasoning (CoT): Think step-by-step before addressing the query to analyze user focus (e.g., collection search, booking workflow, campus-specific opening hours, or guest visitation protocols).\n"
            "4. Exceptional Guidance: In case of missing input, error, or ambiguous requests, offer helpful guidance gracefully without abruptly halting the reasoning process."
        )

    def process(
        self, user_input: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        # EN: [3. Clear Input Definition: Validate and clean input text and context structured dict]
        # KO: [3. 명확한 입력 정의: 입력 텍스트 및 컨텍스트 구조 검증 및 정제]
        # ZH: 【3. 定义清晰的输入：对 user_input 与 context 进行类型限定与空值校验】
        if not user_input or not isinstance(user_input, str):
            # EN: [5. Exception Guidance: Handle empty or invalid input gracefully with librarian greeting]
            # KO: [5. 예외 상황 안내: 생각 프로세스를 임의로 중단하지 않고 사서의 입장에서 친절하게 입력 유도]
            # ZH: 【5. 异常情况引导：对空输入/无效输入提供图书管理员式的专业引导，不可随意停止】
            return self.format_output(
                status="invalid_input",
                response=(
                    "您好！我是深圳大学图书馆的在线管理员。请问您想了解哪个校区图书馆的开放时间、图书借阅藏书、座位预约还是外来参观事宜呢？"
                    " (Hello! I am your SZU Library assistant. How may I help you with library opening hours, book collections, seat reservations, or visitor guidelines across our campuses?)"
                ),
                error_detail="User input is empty or not a valid string.",
            )

        input_text = user_input.strip()
        normalized_input = input_text.lower()
        source_language = infer_source_language(input_text)
        target_language = source_language
        knowledge = (context or {}).get("knowledge", {})
        library_data = knowledge.get("library") or {}

        # EN: Match library facts (collections, campus types/count, hours, booking, visiting) from knowledge base.
        # KO: 지식 베이스에서 도서관 정보(장서, 캠퍼스별 종류/수, 개관 시간, 예약, 방문) 매칭.
        # ZH: 从知识库检索深大图书馆藏书、校区分布、开放时间、预约与参观相关的匹配条目
        matched_facts = []
        semantic_aliases = {
            "main_branches": ["main branches", "branches", "branch", "分馆", "馆舍", "library branches"],
            "official_address": ["address", "location", "where", "where is", "located", "地址", "在哪里", "位置"],
            "opening_hours": ["opening hours", "open time", "hours", "营业时间", "开放时间"],
        }
        if isinstance(library_data, dict):
            for key, value in library_data.items():
                key_lower = key.lower()
                normalized_key = key_lower.replace("_", " ")
                alias_terms = semantic_aliases.get(key_lower, [])
                if (
                    key_lower in normalized_input
                    or normalized_key in normalized_input
                    or any(token in normalized_input for token in normalized_key.split())
                    or any(alias in normalized_input for alias in alias_terms)
                ):
                    matched_facts.append(f"• {key}: {value}")

        # EN: [5. Exception & Missing Info Handling: Tactful librarian response when information is unavailable]
        # KO: [5. 예외 및 정보 부족 처리: 정보를 찾을 수 없을 때 사서의 어조로 우아하고 체계적인 유도]
        # ZH: 【5. 异常与信息不可用处理：未查到知识时以图书管理员口吻委婉回复不可知，严禁捏造，用提问语言解答】
        if not matched_facts:
            append_non_llm_prompt_log(
                skill_name=self.name,
                user_prompt=input_text,
                reason="library_knowledge_lookup_miss",
            )
            return self.format_output(
                status="unavailable",
                response=(
                    "十分抱歉，关于您询问的这项具体细节，我目前的馆务数据库中暂未记录，因此无法确切解答，请您谅解。"
                    "不过您可以访问深大图书馆官方网站或微信公众号查询实时馆藏与动态。若您想了解粤海校区（包玉刚图书馆/汇典楼/汇智楼）或丽湖校区（启明楼）的日常开放时间与座位预约方式，我很乐意为您解答！\n"
                    "(Regrettably, the specific detail you requested is currently unavailable in my library knowledge portal, so I am unable to offer a definitive answer. "
                    "You may check the official SZU Library website for real-time updates, or feel free to inquire about opening hours or seat booking across our campuses.)"
                ),
                error_detail="Library knowledge base lookup miss.",
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
            f"[User Question / 读者提问]: {input_text}\n\n"
            f"[Librarian Guidance Task / 图书管理员服务任务]:\n"
            f"1. Match User Language (提问语言匹配)：首先识别读者提问所使用的语言，全篇回答必须严格使用该语言。\n"
            f"2. Think step-by-step (逐步思考分析)：分析读者咨询的是藏书查询、校区馆舍分布、开放时间、预约流程还是参观规定，严格依据参考资料回答。若资料库未记录，需委婉说明不知道，严禁编造答案。\n"
            f"3. Formulate Response (输出回复)：以富有耐性、言语得体的管理员口吻解答。"
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