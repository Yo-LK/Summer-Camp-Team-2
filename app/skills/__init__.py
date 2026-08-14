# CampusBot/skills/__init__.py

from .base import BaseSkill
from .campus_skill import CampusSkill
from .library_skill import LibrarySkill
from .translation_skill import TranslationSkill

# EN: Create a skill registry to facilitate quick lookup by the Router.
# KO: Router가 빠르게 검색할 수 있도록 스킬 레지스트리를 생성합니다.
# ZH: 创建一个技能字典（Registry），方便 Router 快速查找
def get_all_skills():
    return {
        "campus": CampusSkill(),
        "library": LibrarySkill(),
        "translation": TranslationSkill(),
    }

# EN: Add this line to ensure it works regardless of the import name used externally.
# KO: 외부에서 어떤 이름으로 가져오든 오류가 발생하지 않도록 이 줄을 추가합니다.
# ZH: 增加这行，无论外部用哪种名字导入都不会报错
get_skill_registry = get_all_skills