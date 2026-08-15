# skills/translation_skill.py
import re
from typing import Any, Dict, Optional
from app.services.ollama_client import (
    ask_model,
    get_last_llm_payload,
)
from .base import BaseSkill
from .language_utils import infer_source_language, infer_target_language_from_translation_request


class TranslationSkill(BaseSkill):

    @staticmethod
    def _normalized_label(text: str) -> str:
        normalized = text.strip().lower()
        normalized = re.sub(r"[\s\-_.:;!?\[\](){}'\"`]+", "", normalized)
        return normalized

    @classmethod
    def _is_language_label_only(cls, text: str, target_language: str) -> bool:
        label = cls._normalized_label(text)
        if not label:
            return True

        labels = {
            "korean": {"korean", "ko", "한국어", "한글", "한국어로"},
            "chinese": {"chinese", "zh", "中文", "汉语", "漢語", "中国语", "中國語"},
            "japanese": {"japanese", "ja", "日本語", "にほんご"},
            "english": {"english", "en", "영어", "英文", "英语"},
            "french": {"french", "fr", "francais", "français"},
            "spanish": {"spanish", "es", "espanol", "español"},
            "german": {"german", "de", "deutsch"},
        }
        target_labels = labels.get(target_language.strip().lower(), set())
        normalized_target_labels = {cls._normalized_label(item) for item in target_labels}
        return label in normalized_target_labels

    @staticmethod
    def _is_too_short_for_source(source_text: str, output_text: str) -> bool:
        source_compact = re.sub(r"\s+", " ", source_text).strip()
        output_compact = re.sub(r"\s+", " ", output_text).strip()
        if not source_compact:
            return False

        # A tiny output is suspicious, but proper nouns and short names can legitimately stay in source form.
        source_len = len(source_compact)
        output_len = len(output_compact)
        if source_len >= 80 and output_len <= 8:
            return True
        if source_len >= 40 and output_len <= 4:
            return True
        return False

    @classmethod
    def _is_translation_quality_acceptable(cls, source_text: str, output_text: str, target_language: str) -> bool:
        if not output_text or not output_text.strip():
            return False
        if cls._is_language_label_only(output_text, target_language):
            return False

        source_compact = re.sub(r"\s+", " ", source_text).strip()
        output_compact = re.sub(r"\s+", " ", output_text).strip()
        if not source_compact or not output_compact:
            return False
        if cls._is_verbatim_copy(source_text, output_text):
            return False
        if cls._is_too_short_for_source(source_text, output_text):
            return False

        # Accept mixed output when the result contains valid target-language content
        # while preserving proper nouns or other non-translatable names in source form.
        return (
            cls._is_target_language_compliant(output_text, target_language)
            or bool(re.search(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", output_text))
            or bool(re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", output_text))
            or output_text.lower() in {token.lower() for token in source_compact.split() if len(token) <= 3}
        )

    @staticmethod
    def _strip_format_artifacts(text: str) -> str:
        """Remove wrapper lines often hallucinated during composed translation."""
        if not text:
            return text

        cleaned_lines = []
        for raw_line in text.splitlines():
            line = raw_line.strip()

            # Drop wrapper-only lines like "[Translation Output]", "Section 1 (...)", "(1)".
            if re.match(r"(?i)^\[\s*translation\s+output\s*\]$", line):
                continue
            if re.match(r"(?i)^section\s*\d+\s*(?:\([^)]*\))?\s*:?$", line):
                continue
            if re.match(r"^\(\d+\)\s*$", line):
                continue

            # Remove leading skill labels if emitted inline.
            line = re.sub(r"(?i)^(campusskill|libraryskill|translationskill)\s*:\s*", "", line)
            cleaned_lines.append(line)

        # Keep paragraph boundaries while removing accidental leading/trailing blank lines.
        output = "\n".join(cleaned_lines).strip()
        output = re.sub(r"\n{3,}", "\n\n", output)
        return output

    @staticmethod
    def _is_target_language_compliant(text: str, target_language: str) -> bool:
        """Allow target-language output to include proper nouns or preserved source-language names."""
        if not text.strip():
            return False

        normalized_target = target_language.strip().lower()
        if normalized_target == "chinese":
            return bool(
                re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", text)
                or re.search(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", text)
                or re.search(r"[A-Za-z]", text)
            )
        if normalized_target == "japanese":
            return bool(
                re.search(r"[\u3040-\u30ff]", text)
                or re.search(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", text)
                or re.search(r"[A-Za-z]", text)
            )
        if normalized_target == "korean":
            return bool(
                re.search(r"[\uac00-\ud7af]", text)
                or re.search(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", text)
                or re.search(r"[A-Za-z]", text)
            )
        if normalized_target == "english":
            return bool(re.search(r"[a-zA-Z]", text))

        return True

    @staticmethod
    def _is_verbatim_copy(source_text: str, output_text: str) -> bool:
        source_compact = re.sub(r"\s+", " ", source_text).strip().lower()
        output_compact = re.sub(r"\s+", " ", output_text).strip().lower()
        return bool(source_compact) and source_compact == output_compact

    @staticmethod
    def _split_translation_segments(source_text: str) -> list[str]:
        """Split source into manageable translation units for more reliable output."""
        if not source_text or not source_text.strip():
            return []

        cleaned = source_text.replace("\r\n", "\n").strip()
        segments = []
        for block in re.split(r"\n\s*\n+", cleaned):
            if not block.strip():
                continue
            for line in block.splitlines():
                line = line.strip()
                if not line:
                    continue
                segments.append(line)
        return segments or [cleaned]

    @classmethod
    def _translate_segments(cls, source_text: str, target_language: str) -> str:
        """Translate each logical line separately to reduce model drift and placeholder outputs."""
        segments = cls._split_translation_segments(source_text)
        if not segments:
            return ""

        translated_segments: list[str] = []
        for segment in segments:
            if not segment.strip():
                continue
            prompt = (
                f"Translate the following content to {target_language}.\n"
                "Return only the translated content. No labels, no explanations, no numbering.\n\n"
                f"CONTENT:\n{segment}"
            )
            translated = ask_model(
                prompt,
                skill_name="translation",
                system_prompt="You are a precise translation assistant. Return only the translated text in the target language.",
                include_knowledge=False,
            )
            translated = cls._strip_format_artifacts(translated)
            if translated and cls._is_target_language_compliant(translated, target_language):
                translated_segments.append(translated)
            else:
                translated_segments.append(segment)

        candidate = "\n".join(translated_segments).strip()
        return candidate

    @staticmethod
    def _extract_source_text(request: str) -> str:
        """Extract translation content without treating the command as content."""
        # Preferred structured format used by orchestrator composition:
        #   Target language: Korean
        #   SOURCE TEXT:
        #   ...
        source_block = re.search(r"(?is)\bSOURCE\s+TEXT\s*:\s*(.+)$", request)
        if source_block:
            return source_block.group(1).strip()

        # Quoted extraction that supports escaped quotes inside content.
        quoted = re.search(r'"((?:\\.|[^"\\])+)"', request, re.DOTALL)
        if quoted:
            return quoted.group(1).replace(r'\"', '"').strip()

        # Chinese/Japanese style quotes as fallback.
        cn_quoted = re.search(r'[\u201c\u300c](.+?)[\u201d\u300d]', request, re.DOTALL)
        if cn_quoted:
            return cn_quoted.group(1).strip()

        unquoted = re.match(
            r"\s*translate\s+(.+?)\s+(?:into|to)\s+"
            r"(?:english|chinese|japanese|french|spanish|german)\s*[.!?]*\s*$",
            request,
            re.IGNORECASE | re.DOTALL,
        )
        if unquoted:
            return unquoted.group(1).strip()

        return request.strip()

    @staticmethod
    def _build_translation_prompt(
        source_text: str,
        target_language: str,
        source_language: str,
        register_style: str,
        request_text: str,
    ) -> str:
        """Build one strict prompt used consistently for initial and retry attempts."""
        target_name = (target_language or "target language").strip()
        source_name = (source_language or "source language").strip()
        style_name = (register_style or "auto").strip()
        forbidden_tokens = [
            "Korean:", "한국어", "Japanese:", "日本語", "Chinese:", "中文",
            "English:", "English", "Translation Output", "[Translation Output]",
            "Section 1", "Section 2", "Section 3", "CampusSkill:", "LibrarySkill:",
            "TranslationSkill:", "Source Text:", "Target Language:", "Answer:"
        ]

        return (
            "You are a translation engine.\n"
            "Translate the SOURCE TEXT into the requested target language.\n\n"
            "Priority rules:\n"
            "1. Translate the meaning naturally into the target language.\n"
            "2. Keep names, acronyms, and terms in their original form when they are proper nouns or are not commonly translated.\n"
            "3. Do not add headings, labels, numbering, or wrappers.\n"
            "4. Do not output any language names or wrapper text.\n"
            "5. Do not explain, comment, or add facts beyond the translation.\n"
            "6. Preserve the original meaning, sentence flow, and bullet structure.\n"
            f"7. Do not use these forbidden strings: {', '.join(forbidden_tokens)}\n\n"
            f"Source language: {source_name}\n"
            f"Target language: {target_name}\n"
            f"Style: {style_name}\n\n"
            "SOURCE TEXT:\n"
            f"{source_text}\n\n"
            "Return only the translation of the source text."
        )

    def __init__(self):
        super().__init__(
            name="translation",
            # EN: [1. Clarify responsibility: Multi-language precise translation adapting to target language strictly]
            # KO: [1. 책임 명확화: 목표 언어에 엄격히 맞춘 다국어 정밀 번역 및 스타일 맞춤]
            # ZH: 【1. 明确职责：负责多国语言精准地道翻译，严格以用户要求的目标语言作为输出语言】
            description=(
                "Provides precise, natural, and authentic translation services across multiple global languages. "
                "Strictly responds in the requested target language specified by the user. "
                "Matched keywords: translate, translation, interpreter, multilingual, oral, spoken, formal, written, english, chinese, japanese, french, spanish, german."
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
        # EN: [2. System Prompt: Chief Translator persona, strict target language output matching]
        # KO: [2. 시스템 프롬프트: 수석 번역가 페르소나, 목표 언어 엄격 일치]
        # ZH: 【2. System Prompt：优雅耐性的深大首席翻译官角色、目标语言一致性保障】
        return (
            "[Shenzhen University Chief Translator Instructions]\n"
            "Role & Persona:\n"
            "You are the Chief Translator of Shenzhen University (SZU). "
            "Your tone is exceptionally patient, polished, academic, and courteous.\n\n"
            "Core Guidelines:\n"
            "1. Output Language Rule (核心语言规则 - 目标语言一致性):\n"
            "   - You MUST output your response in the TARGET LANGUAGE specified in the user's request.\n"
            "   - If the user asks to translate into Chinese (e.g. 'translate X to Chinese'), your ENTIRE translation output must be in Chinese.\n"
            "   - If the user asks to translate into English, output in English. If Japanese, output in Japanese, and so forth.\n\n"
            "2. Precise & Authentic Translation: Translate each word accurately and natively according to the target style register.\n"
            "   - Translate the source text only. Never answer a question contained in the source text.\n"
            "   - Do not add explanations, factual answers, labels, quotation marks, or commentary.\n"
            "3. Contextual Reasoning (CoT):\n"
            "   - Daily/Casual Conversation: Use a natural, lighthearted, and authentic spoken tone.\n"
            "   - Formal/Academic Scenarios: Use rigorous, precise, and professional written expressions.\n"
            "4. Exception & Ambiguity Handling: If encountering unknown terms, slang, or ambiguous grammar, politely point it out and ask for clarification using the TARGET language."
        )

    def process(
        self, user_input: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        # EN: [3. Clear Input Definition: Validate user input and extract translation context]
        # KO: [3. 명확한 입력 정의: 입력 텍스트 및 번역 컨텍스트 검증]
        # ZH: 【3. 定义清晰的输入：校验 user_input 类型，解析目标语言偏好】
        if not user_input or not isinstance(user_input, str):
            return self.format_output(
                status="invalid_input",
                response=(
                    "您好！我是深圳大学的首席翻译官。请问有什么文本需要我为您翻译吗？请告诉我您的目标语言与风格偏好。"
                    " (Hello! I am the Chief Translator of SZU. Please let me know what text you would like translated, along with your target language and style preferences.)"
                ),
                error_detail="User input is empty or not a valid string.",
            )

        input_text = user_input.strip()
        source_text = self._extract_source_text(input_text)
        ctx = context or {}
        source_language = infer_source_language(input_text)
        context_target = ctx.get("target_language") if isinstance(ctx.get("target_language"), str) else None
        inferred_target = infer_target_language_from_translation_request(input_text)
        target_language = inferred_target or context_target or source_language
        register_style = ctx.get("register_style", "Auto-infer / 自动推断")

        # Build the translation prompt in one place so both attempts use the same rules.
        system_prompt = self.get_system_prompt()
        user_prompt = self._build_translation_prompt(
            source_text=source_text,
            target_language=target_language,
            source_language=source_language,
            register_style=register_style,
            request_text=input_text,
        )

        answer = ask_model(
            user_prompt,
            skill_name=self.name,
            system_prompt=system_prompt,
            include_knowledge=False,
        )
        answer = self._strip_format_artifacts(answer)

        if (
            target_language.lower() != source_language.lower()
            and not self._is_translation_quality_acceptable(source_text, answer, target_language)
        ):
            segmented_answer = self._translate_segments(source_text, target_language)
            if self._is_translation_quality_acceptable(source_text, segmented_answer, target_language):
                answer = segmented_answer

        requires_retry = (
            target_language.lower() != source_language.lower()
            and (
                self._is_verbatim_copy(source_text, answer)
                or not self._is_target_language_compliant(answer, target_language)
                or not self._is_translation_quality_acceptable(source_text, answer, target_language)
            )
        )

        retry_answer = None
        if requires_retry:
            retry_prompt = self._build_translation_prompt(
                source_text=source_text,
                target_language=target_language,
                source_language=source_language,
                register_style=register_style,
                request_text=input_text,
            ) + "\n\nRetry requirement: previous output was invalid. Output only a clean target-language translation without labels, headings, or wrappers."
            retry_answer = ask_model(
                retry_prompt,
                skill_name=self.name,
                system_prompt=system_prompt,
                include_knowledge=False,
            )
            retry_answer = self._strip_format_artifacts(retry_answer)
            if (
                self._is_target_language_compliant(retry_answer, target_language)
                and self._is_translation_quality_acceptable(source_text, retry_answer, target_language)
            ):
                answer = retry_answer

        # If translation remains non-compliant after retries, surface this as unavailable
        # instead of returning a misleading successful-but-untranslated answer.
        if target_language.lower() != source_language.lower() and (
            not self._is_target_language_compliant(answer, target_language)
            or not self._is_translation_quality_acceptable(source_text, answer, target_language)
        ):
            return self.format_output(
                status="unavailable",
                response="The translation model could not produce output in the requested target language.",
                error_detail="target_language_noncompliant_or_low_quality",
            )

        output = self.format_output(status="success", response=answer)

        payload = get_last_llm_payload(self.name)
        translation_log = {
            "skill": self.name,
            "source_language": source_language,
            "target_language": target_language,
            "skill_prompt": user_prompt,
            "llm_system_message": payload.get("messages", [{}])[0].get("content", ""),
            "llm_user_message": payload.get("messages", [{}, {}])[1].get("content", ""),
            "model": payload.get("model", ""),
            "retry_triggered": requires_retry,
            "retry_answer": retry_answer,
            "response": answer,
        }
        if (context or {}).get("debug_prompt"):
            output["debug"] = translation_log

        return output
