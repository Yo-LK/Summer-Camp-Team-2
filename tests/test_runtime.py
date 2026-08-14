import json
import sys
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from app.agent_runtime import create_runtime
from app.skills import get_all_skills


KNOWLEDGE = json.loads(
    (ROOT / "knowledge" / "knowledge.json").read_text(encoding="utf-8")
)


class FakeResp:
    def raise_for_status(self):
        return None

    def json(self):
        return {"message": {"content": "mocked-answer"}}


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.runtime = create_runtime(get_all_skills())
        self.context = {"knowledge": KNOWLEDGE}

    def _call(self, message: str, context=None):
        with patch("app.services.ollama_client.httpx.post", return_value=FakeResp()):
            return self.runtime.handle_request(message, self.context if context is None else context)

    def test_library_request_routes_to_library(self):
        result = self._call("library official_address?")

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["routed_skill"], "library")

    def test_translation_request_routes_to_translation(self):
        result = self._call('Translate "Welcome to Shenzhen University" into Chinese.')

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["routed_skill"], "translation")

    def test_campus_request_routes_to_campus(self):
        result = self._call("What is Shenzhen University's motto?")

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["routed_skill"], "campus")

    def test_unrelated_request_falls_back_to_campus(self):
        result = self._call("How do I repair a motorcycle engine?")
        self.assertEqual(result["routed_skill"], "campus")

    def test_empty_request_returns_error(self):
        result = self.runtime.handle_request("   ", self.context)

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["skill"], "none")

    def test_missing_library_knowledge_is_unavailable(self):
        result = self._call(
            "Where is the library?",
            context={"knowledge": {}},
        )

        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["routed_skill"], "library")
        self.assertIn("unavailable", result["response"].lower())


if __name__ == "__main__":
    unittest.main()