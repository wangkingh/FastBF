from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from caponlib.cli import main


class CliGuideTests(unittest.TestCase):
    def test_guide_prints_workflow_reminder(self) -> None:
        stream = io.StringIO()

        with redirect_stdout(stream):
            code = main(["guide"])

        self.assertEqual(0, code)
        text = stream.getvalue()
        self.assertIn("FastBF 防忘指南", text)
        self.assertIn("fastbf checkmem", text)
        self.assertIn("docs/USAGE_GUARDRAILS.md", text)

    def test_english_guide_is_available(self) -> None:
        stream = io.StringIO()

        with redirect_stdout(stream):
            code = main(["guide", "--lang", "en"])

        self.assertEqual(0, code)
        text = stream.getvalue()
        self.assertIn("FastBF usage reminder", text)
        self.assertIn("fastbf run my.ini", text)


if __name__ == "__main__":
    unittest.main()
