# runtime/orchestrator.py
import re
from typing import Any, Dict, List, Optional

from app.governance import check_request
from app.services.workflow_logger import append_workflow_log
from app.skills.base import BaseSkill
from app.skills.language_utils import infer_source_language

from .router import SkillRouter


class RuntimeOrchestrator:

    def __init__(self, router: SkillRouter, skills: Dict[str, BaseSkill]):
        self.router = router
        self.skills = skills

    def _detect_explicit_output_language(self, user_input: str) -> Optional[str]:
        """
        EN: Check if user explicitly requested an output language (e.g., "respond in Chinese").
        KO: 사용자가 명시적으로 출력 언어를 요청했는지 확인합니다 (예: "중국어로 답변하세요").
        ZH: 检查用户是否显式请求输出语言 (例如："用中文回答")。
        """
        text = user_input.lower()

        if any(token in text for token in ["in chinese", "中文", "to chinese", "用中文", "以中文"]):
            return "Chinese"
        if any(token in text for token in ["in english", "英文", "to english", "用英语", "以英语"]):
            return "English"
        if any(token in text for token in ["in japanese", "日文", "to japanese", "用日语", "以日语"]):
            return "Japanese"
        if any(token in text for token in ["in korean", "韩文", "韓文", "to korean", "用韩语", "用韓語", "以韩语", "以韓語"]):
            return "Korean"
        if any(token in text for token in ["in spanish", "西班牙语", "to spanish", "用西班牙语"]):
            return "Spanish"
        if any(token in text for token in ["in french", "法语", "to french", "用法语"]):
            return "French"
        if any(token in text for token in ["in german", "德语", "德文", "to german", "用德语", "以德语"]):
            return "German"

        return None

    def _is_direct_translation_request(self, user_input: str) -> bool:
        text = user_input.strip().lower()
        if re.match(r"^(please\s+)?translate\b", text):
            return True
        if re.match(r"^翻译", user_input.strip()):
            return True
        return False

    def _get_target_language(self, user_input: str) -> Optional[str]:
        """
        EN: Determine target language for response.
        Priority 1: Explicit language request from user (e.g., "respond in Chinese").
        Priority 2: Source language of user input (if not English).
        Default: None (no translation needed for English).
        
        KO: 응답 대상 언어를 결정합니다.
        우선순위 1: 사용자의 명시적 언어 요청 (예: "중국어로 답변하세요").
        우선순위 2: 사용자 입력의 원본 언어 (영어가 아닌 경우).
        기본값: None (영어의 경우 번역 불필요).
        
        ZH: 确定响应的目标语言。
        优先级 1: 用户的显式语言请求 (例如："用中文回答")。
        优先级 2: 用户输入的源语言 (如果非英语)。
        默认值: None (英语无需翻译)。
        """
        # Priority 1: Check for explicit language request
        explicit_lang = self._detect_explicit_output_language(user_input)
        if explicit_lang:
            return explicit_lang

        # Priority 2: Detect source language and use it if non-English
        # This ensures response is in the language the user asked in
        source_lang = infer_source_language(user_input)
        if source_lang != "English":
            return source_lang

        # Default: no translation needed
        return None

    def _strip_output_language_instruction(self, user_input: str) -> str:
        cleaned = user_input.strip()
        patterns = [
            r"\s*return the response in [a-zA-Z\u4e00-\u9fff]+\.?",
            r"\s*answer in [a-zA-Z\u4e00-\u9fff]+\.?",
            r"\s*respond in [a-zA-Z\u4e00-\u9fff]+\.?",
            r"\s*translate(\s+the\s+response)?\s+to\s+[a-zA-Z\u4e00-\u9fff]+\.?",
            r"\s*translate(\s+the\s+response)?\s+into\s+[a-zA-Z\u4e00-\u9fff]+\.?",
            r"\s*in [a-zA-Z\u4e00-\u9fff]+\.?$",
            r"\s*用中文\.?$",
            r"\s*用英语\.?$",
            r"\s*用日语\.?$",
            r"\s*用韩语\.?$",
            r"\s*用韓語\.?$",
            r"\s*用德语\.?$",
        ]
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        return " ".join(cleaned.split())

    def _should_run_campus_and_library(self, factual_input: str) -> bool:
        """Detect mixed intent that should run both CampusSkill and LibrarySkill."""
        text = factual_input.lower()

        library_markers = {
            "library", "book", "collection", "opening hours", "open time",
            "booking", "reservation", "reserve seat", "visit", "visitor",
            "curator", "图书馆", "借书", "馆长", "开馆时间",
        }
        campus_specific_markers = {
            "founded", "established",
            "history", "architecture", "building", "building name", "landmark",
            "alumni", "famous figure", "professor", "president", "culture",
            "motto", "校长", "校训", "历史", "建筑", "名人",
        }

        has_library = any(marker in text for marker in library_markers)
        has_campus_specific = any(marker in text for marker in campus_specific_markers)
        return has_library and has_campus_specific

    def _get_mixed_intent_skill_order(self, factual_input: str) -> List[str]:
        """Return [first, second] order for campus/library based on prompt appearance."""
        text = factual_input.lower()

        library_markers = [
            "library", "book", "collection", "opening hours", "open time",
            "booking", "reservation", "reserve seat", "visit", "visitor",
            "curator", "图书馆", "借书", "馆长", "开馆时间", "分馆",
        ]
        campus_markers = [
            "founded", "established", "history", "architecture", "building",
            "landmark", "alumni", "professor", "president", "culture",
            "motto", "campus", "校长", "校训", "历史", "建筑", "名人", "成立",
        ]

        library_positions = [text.find(marker) for marker in library_markers if text.find(marker) != -1]
        campus_positions = [text.find(marker) for marker in campus_markers if text.find(marker) != -1]

        first_library = min(library_positions) if library_positions else float("inf")
        first_campus = min(campus_positions) if campus_positions else float("inf")

        if first_library < first_campus:
            return ["library", "campus"]
        return ["campus", "library"]

    def _split_skill_specific_inputs(self, factual_input: str) -> Dict[str, str]:
        """Split a mixed-intent request into campus/library-specific inputs."""
        text = factual_input.strip()

        # Split by sentence punctuation first, then by conjunctions.
        sentence_parts = [s.strip() for s in re.split(r"[?.!;。？！]+", text) if s.strip()]
        segments: List[str] = []
        for sentence in sentence_parts:
            sub_parts = [
                part.strip() for part in re.split(r"\s+(?:and|以及|并且)\s+", sentence, flags=re.IGNORECASE)
                if part.strip()
            ]
            segments.extend(sub_parts if sub_parts else [sentence])

        library_markers = {
            "library", "book", "collection", "opening hours", "open time",
            "booking", "reservation", "reserve seat", "visit", "visitor",
            "curator", "图书馆", "借书", "馆长", "开馆时间", "分馆",
        }
        campus_markers = {
            "founded", "established", "history", "architecture", "building",
            "landmark", "alumni", "professor", "president", "culture",
            "motto", "campus", "校长", "校训", "历史", "建筑", "名人", "成立",
        }

        campus_segments: List[str] = []
        library_segments: List[str] = []

        for segment in segments:
            lowered = segment.lower()
            has_library = any(marker in lowered for marker in library_markers)
            has_campus = any(marker in lowered for marker in campus_markers)

            if has_library and not has_campus:
                library_segments.append(segment)
            elif has_campus and not has_library:
                campus_segments.append(segment)
            elif has_library and has_campus:
                campus_segments.append(segment)
                library_segments.append(segment)

        campus_text = "? ".join(campus_segments).strip()
        library_text = "? ".join(library_segments).strip()

        # Fallback to full input if split/classification fails.
        if not campus_text:
            campus_text = text
        if not library_text:
            library_text = text

        return {
            "campus": campus_text,
            "library": library_text,
        }

    def _build_workflow(self, user_input: str) -> List[Dict[str, Any]]:
        """
        EN: Build multi-step workflow for handling user request.
        Workflow construction:
        1. Check for explicit language request (if any).
        2. Clean input only if explicit language instruction was found.
        3. Route cleaned input to appropriate skill.
        4. Determine target language (explicit request OR source language if non-English).
        5. Add translation step to match user's input language (if not English).
        
        KO: 사용자 요청을 처리하기 위한 다단계 워크플로우를 구축합니다.
        
        ZH: 构建多步骤工作流来处理用户请求。
        """
        # Direct translation requests should remain translation-only.
        if self._is_direct_translation_request(user_input):
            return [{
                "type": "fact",
                "skill": "translation",
                "text": user_input.strip(),
                "target_language": None,
            }]

        # For composition queries like:
        # "What are ...? Translate the response to Korean"
        # always strip output-language directives before routing factual intent.
        factual_input = self._strip_output_language_instruction(user_input)
        if not factual_input:
            factual_input = user_input.strip()

        workflow: List[Dict[str, Any]] = []

        # Multi-fact composition path: CampusSkill -> LibrarySkill
        if self._should_run_campus_and_library(factual_input):
            split_inputs = self._split_skill_specific_inputs(factual_input)
            for skill_name in self._get_mixed_intent_skill_order(factual_input):
                if self.skills.get(skill_name) is None:
                    continue
                workflow.append({
                    "type": "fact",
                    "skill": skill_name,
                    "text": split_inputs[skill_name],
                    "target_language": None,
                })

        # Default single-fact path via router.
        if not workflow:
            base_skill = self.router.route(factual_input)
            if base_skill is None:
                base_skill = self.skills.get("campus")
            if base_skill is None:
                return []
            workflow.append({
                "type": "fact",
                "skill": base_skill.name,
                "text": factual_input,
                "target_language": None,
            })

        # Determine target language:
        # 1. Explicit request takes priority
        # 2. If no explicit request, use source language (if non-English)
        target_language = self._get_target_language(user_input)

        # Add translation step if needed (and not already a translation skill)
        if target_language and any(step.get("skill") != "translation" for step in workflow):
            workflow.append({
                "type": "transform",
                "skill": "translation",
                "source": "all_facts",
                "target_language": target_language,
                "text": "",
            })

        return workflow

    def handle_request(
        self, user_input: str, context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """接收请求 -> 安全校验 -> 任务分解 -> 顺序执行技能 -> 返回统一格式"""
        context = context or {}
        executed_steps: List[Dict[str, Any]] = []

        def _log(status: str) -> None:
            append_workflow_log(user_input=user_input, steps=executed_steps, final_status=status)

        if not user_input or not user_input.strip():
            _log("error")
            return {
                "status": "error",
                "skill": "none",
                "response": "Message cannot be empty.",
                "error_detail": "Empty user input.",
            }

        guardrail_result = check_request(user_input)
        if not guardrail_result.allowed:
            _log("blocked")
            return {
                "status": "blocked",
                "skill": "none",
                "response": "Request blocked.",
                "error_detail": guardrail_result.reason,
            }

        workflow = self._build_workflow(user_input)
        if not workflow:
            _log("unmatched")
            return {
                "status": "unmatched",
                "skill": "none",
                "response": "No suitable skill was found for this request.",
                "error_detail": None,
            }

        previous_result = ""
        final_text = ""
        final_status = "success"
        last_skill_name = workflow[0]["skill"]
        last_successful_response = ""
        fact_outputs: List[Dict[str, str]] = []

        try:
            for step in workflow:
                skill_name = step["skill"]
                skill = self.skills.get(skill_name)
                if skill is None:
                    _log("unmatched")
                    return {
                        "status": "unmatched",
                        "skill": "none",
                        "response": f"Skill '{skill_name}' is not registered.",
                        "error_detail": None,
                        "workflow": executed_steps,
                    }

                if step.get("type") == "transform":
                    if step.get("source") not in {"previous", "all_facts"}:
                        raise ValueError("Transform steps must consume prior step output.")
                    if step.get("source") == "all_facts":
                        if not fact_outputs:
                            raise ValueError("No fact outputs available to transform.")
                        source_for_translation = " ".join(
                            re.sub(r"\s+", " ", entry["response"]).strip()
                            for entry in fact_outputs
                            if entry.get("response")
                        )
                    else:
                        if not previous_result:
                            raise ValueError("No previous output available to transform.")
                        source_for_translation = previous_result

                    target_language = step.get("target_language") or "English"
                    quoted_source = source_for_translation.strip().replace('"', '\\"')
                    step_input = (
                        f'Translate "{quoted_source}" into {target_language}. '
                        f"Translate naturally and keep proper nouns, names, and non-translatable terms in their original form when needed. "
                        f"Do not add titles, section labels, numbering, or wrappers. "
                        f"Preserve the original structure and bullet ordering."
                    )
                else:
                    step_input = step.get("text", "")

                skill_context = dict(context)
                if step.get("target_language"):
                    skill_context["target_language"] = step["target_language"]

                result = skill.process(step_input, skill_context)
                executed_steps.append({
                    "type": step.get("type"),
                    "skill": skill.name,
                    "skill_label": skill.__class__.__name__,
                    "input": step_input,
                    "status": result.get("status"),
                    "response": result.get("response"),
                    "target_language": step.get("target_language"),
                })

                final_status = result.get("status", "success")
                last_skill_name = skill.name
                final_text = result.get("response") or final_text
                if result.get("status") == "success":
                    last_successful_response = result.get("response") or last_successful_response
                    previous_result = last_successful_response
                    if step.get("type") == "fact":
                        fact_outputs.append({
                            "skill_label": skill.__class__.__name__,
                            "response": result.get("response") or "",
                        })
                else:
                    previous_result = previous_result

                if result.get("status") not in {"success"}:
                    _log(result.get("status", "error"))
                    return {
                        "status": result.get("status", "error"),
                        "skill": skill.name,
                        "response": result.get("response") or "The selected skill could not process the request.",
                        "error_detail": result.get("error_detail"),
                        "routed_skill": " -> ".join(step["skill"] for step in workflow if step.get("skill")),
                        "workflow": executed_steps,
                    }

            routed_skill = " -> ".join(step["skill"] for step in workflow)
            _log(final_status)
            return {
                "status": final_status,
                "skill": last_skill_name,
                "response": last_successful_response or final_text,
                "error_detail": None,
                "routed_skill": routed_skill,
                "workflow": executed_steps,
            }

        except Exception as exc:
            _log("error")
            return {
                "status": "error",
                "skill": last_skill_name,
                "response": "The selected skill could not process the request.",
                "error_detail": str(exc),
                "routed_skill": " -> ".join(step["skill"] for step in workflow),
                "workflow": executed_steps,
            }
