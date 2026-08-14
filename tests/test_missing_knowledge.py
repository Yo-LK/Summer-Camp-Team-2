import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.skills import CampusSkill


class MissingKnowledgeTest(unittest.TestCase):
    def test_missing_knowledge_does_not_invent_answer(self):
        result = CampusSkill().process(
            "Who is the current president?",
            {"knowledge": {}},
        )
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["skill"], "campus")


if __name__ == "__main__":
    unittest.main()
