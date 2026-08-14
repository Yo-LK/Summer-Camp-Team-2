import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent_runtime.router import SkillRouter
from app.skills import get_all_skills


class CorrectSkillRoutingTest(unittest.TestCase):
    def test_library_request_routes_to_correct_skill(self):
        selected = SkillRouter(get_all_skills()).route("Where is the library?")
        self.assertIsNotNone(selected)
        self.assertEqual(selected.name, "library")


if __name__ == "__main__":
    unittest.main()
