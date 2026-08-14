from __future__ import annotations

import re
from typing import Optional


def infer_source_language(text: str) -> str:
    if re.search(r"[\uac00-\ud7af]", text):
        return "Korean"
    if re.search(r"[\u3040-\u30ff]", text):
        return "Japanese"
    if re.search(r"[\u4e00-\u9fff]", text):
        return "Chinese"
    return "English"


def infer_target_language_from_translation_request(text: str) -> Optional[str]:
    lowered = text.lower()
    if "to chinese" in lowered or "into chinese" in lowered or "翻译成中文" in text or "翻译为中文" in text:
        return "Chinese"
    if "to english" in lowered or "into english" in lowered or "翻译成英文" in text or "翻译为英文" in text:
        return "English"
    if "to korean" in lowered or "into korean" in lowered or "翻译成韩语" in text or "翻译为韩语" in text:
        return "Korean"
    if "to japanese" in lowered or "into japanese" in lowered or "翻译成日语" in text or "翻译为日语" in text:
        return "Japanese"
    if "to french" in lowered or "into french" in lowered:
        return "French"
    if "to spanish" in lowered or "into spanish" in lowered:
        return "Spanish"
    if "to german" in lowered or "into german" in lowered:
        return "German"
    return None
