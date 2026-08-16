import sys
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.skills import TranslationSkill


class TranslationOnlyTest(unittest.TestCase):
    def test_translation_question_is_translated_not_answered(self):
        skill = TranslationSkill()
        with patch(
            "app.skills.translation_skill.ask_model",
            return_value="深圳大学是什么时候成立的？",
        ):
            result = skill.process(
                'Translate "When was Shenzhen University founded?" into Chinese'
            )

        expected = "\u6df1\u5733\u5927\u5b66\u662f\u4ec0\u4e48\u65f6\u5019\u6210\u7acb\u7684\uff1f"
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["response"], expected)
        self.assertNotIn("1983", result["response"])
        self.assertNotIn("2009", result["response"])


if __name__ == "__main__":
    unittest.main()