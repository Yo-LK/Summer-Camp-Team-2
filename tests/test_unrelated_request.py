import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent_runtime.router import SkillRouter
from app.skills import get_all_skills


class UnrelatedRequestTest(unittest.TestCase):
    def test_unrelated_request_is_not_misrouted(self):
        selected = SkillRouter(get_all_skills()).route(
            "How do I repair a motorcycle engine?"
        )
        self.assertIsNone(selected)


if __name__ == "__main__":
    unittest.main()
