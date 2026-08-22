"""Static checks for user-facing defaults and legacy-model disclosure."""

import os
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class StaticConsistencyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(ROOT, "regal_explorer.html"), encoding="utf-8") as fh:
            cls.html = fh.read()
        with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as fh:
            cls.readme = fh.read()

    def test_reset_uses_default_venetoclax_cure(self):
        self.assertIn('name:"Venetoclax",w:35,med:12,cure:15', self.html)
        self.assertIn('$("vcure").value=15', self.html)
        self.assertIn('$("vcL").textContent="15%"', self.html)
        self.assertNotIn('$("vcure").value=22', self.html)

    def test_legacy_output_is_not_described_as_posterior(self):
        self.assertIn("v1 legacy model", self.readme)
        self.assertRegex(self.readme, r"\*\*not\*\* a posterior probability")
        self.assertIn("condition on the observed decision to continue", self.readme)


if __name__ == "__main__":
    unittest.main()
