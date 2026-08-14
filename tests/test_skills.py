import json
import sys
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from app.skills import CampusSkill, LibrarySkill, TranslationSkill


KNOWLEDGE = json.loads(
    (ROOT / "knowledge" / "knowledge.json").read_text(encoding="utf-8")
)
CONTEXT = {"knowledge": KNOWLEDGE}


class FakeResp:
    def raise_for_status(self):
        return None

    def json(self):
        return {"message": {"content": "mocked-answer"}}


class CampusSkillTests(unittest.TestCase):
    def setUp(self):
        self.skill = CampusSkill()

    def test_campus_skill_has_independent_prompt(self):
        prompt = self.skill.get_system_prompt()

        self.assertIn("Campus Guide", prompt)
        self.assertNotIn("Professional Librarian", prompt)

    def test_campus_skill_returns_structured_output(self):
        with patch("app.services.ollama_client.httpx.post", return_value=FakeResp()):
            result = self.skill.process(
                "What is Shenzhen University's motto?",
                CONTEXT,
            )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["skill"], "campus")
        self.assertIn("response", result)
        self.assertEqual(result["response"], "mocked-answer")


class LibrarySkillTests(unittest.TestCase):
    def setUp(self):
        self.skill = LibrarySkill()

    def test_library_skill_has_independent_prompt(self):
        prompt = self.skill.get_system_prompt()

        self.assertIn("Professional Librarian", prompt)
        self.assertNotIn("Chief Translator", prompt)

    def test_library_skill_uses_library_knowledge(self):
        with patch("app.services.ollama_client.httpx.post", return_value=FakeResp()):
            result = self.skill.process(
                "library official_address?",
                CONTEXT,
            )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["skill"], "library")
        self.assertEqual(result["response"], "mocked-answer")

    def test_library_skill_handles_missing_knowledge(self):
        result = self.skill.process(
            "Where is the library?",
            {"knowledge": {}},
        )

        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["skill"], "library")
        self.assertEqual(
            result["error_detail"],
            "Library knowledge base lookup miss.",
        )


class TranslationSkillTests(unittest.TestCase):
    def setUp(self):
        self.skill = TranslationSkill()

    def test_translation_skill_has_independent_prompt(self):
        prompt = self.skill.get_system_prompt()

        self.assertIn("Chief Translator", prompt)
        self.assertNotIn("Professional Librarian", prompt)

    def test_translation_skill_returns_structured_output(self):
        with patch("app.services.ollama_client.httpx.post", return_value=FakeResp()):
            result = self.skill.process(
                'Translate "Welcome to Shenzhen University" into Chinese.',
                {},
            )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["skill"], "translation")
        self.assertEqual(result["response"], "mocked-answer")

    def test_translation_skill_preserves_input(self):
        message = 'Translate "Welcome to Shenzhen University" into Chinese.'
        with patch("app.services.ollama_client.httpx.post", return_value=FakeResp()):
            result = self.skill.process(message, {"debug_prompt": True})

        self.assertIn(message, result.get("debug", {}).get("skill_prompt", ""))


if __name__ == "__main__":
    unittest.main()