# CampusBot/skills/__init__.py

from .base import BaseSkill
from .campus_skill import CampusSkill
from .library_skill import LibrarySkill
from .translation_skill import TranslationSkill


# 创建一个技能字典（Registry），方便 Router 快速查找
def get_all_skills():
    return {
        "campus": CampusSkill(),
        "library": LibrarySkill(),
        "translation": TranslationSkill(),
    }


# 增加这行，无论外部用哪种名字导入都不会报错
get_skill_registry = get_all_skills