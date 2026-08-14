# # runtime/__init__.py

# from .orchestrator import RuntimeOrchestrator
# from .router import SkillRouter

# # 定义对外暴露的类
# __all__ = ["SkillRouter", "RuntimeOrchestrator"]


# # 方便外部直接快速创建配置好的 Runtime 实例
# def create_runtime(skills_registry: dict) -> RuntimeOrchestrator:
#     """快速构建并返回一个包含指定技能列表的 RuntimeOrchestrator 实例"""
#     router = SkillRouter(skills_registry)
#     return RuntimeOrchestrator(router=router, skills=skills_registry)
"""CampusBot Runtime package."""

from .orchestrator import RuntimeOrchestrator
from .router import SkillRouter


__all__ = [
    "SkillRouter",
    "RuntimeOrchestrator",
    "create_runtime",
]


def create_runtime(skills_registry: dict) -> RuntimeOrchestrator:
    """Create a RuntimeOrchestrator with registered Skills."""

    router = SkillRouter(skills_registry)

    return RuntimeOrchestrator(
        router=router,
        skills=skills_registry,
    )