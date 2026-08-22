"""Static invariants for user-facing defaults and legacy-model disclosure."""

from decimal import Decimal
import os
import re
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SURFACE_FILES = (
    "README.md",
    "regal_explorer.html",
    "regal_explorer.py",
    "REGAL_MODEL_DOCUMENTATION.md",
)


class StaticConsistencyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.surfaces = {}
        for name in SURFACE_FILES:
            with open(os.path.join(ROOT, name), encoding="utf-8") as fh:
                cls.surfaces[name] = fh.read()
        cls.html = cls.surfaces["regal_explorer.html"]

    @staticmethod
    def _quoted_attr(tag, attr):
        match = re.search(rf"\b{attr}=(['\"])(.*?)\1", tag, re.IGNORECASE)
        return match.group(2) if match else None

    def test_every_reset_value_matches_its_markup_default(self):
        reset = re.search(
            r'\$\("reset"\)\.addEventListener\("click",\(\)=>\{(?P<body>.*?)\}\);',
            self.html,
            re.DOTALL,
        )
        self.assertIsNotNone(reset, "reset handler not found")
        body = reset.group("body")

        tags_by_id = {}
        for tag in re.findall(r"<input\b[^>]*>", self.html, re.IGNORECASE):
            control_id = self._quoted_attr(tag, "id")
            if control_id:
                tags_by_id[control_id] = tag

        value_writes = dict(
            re.findall(
                r'\$\("([^\"]+)"\)\.value=([-+]?(?:\d+(?:\.\d*)?|\.\d+))',
                body,
            )
        )
        self.assertGreaterEqual(len(value_writes), 12, "unexpectedly narrow reset coverage")
        for control_id, reset_value in value_writes.items():
            self.assertIn(control_id, tags_by_id, f"reset control {control_id!r} has no input")
            markup_value = self._quoted_attr(tags_by_id[control_id], "value")
            self.assertIsNotNone(markup_value, f"input {control_id!r} has no markup default")
            self.assertEqual(
                Decimal(reset_value),
                Decimal(markup_value),
                f"reset value for {control_id!r} differs from its markup default",
            )

        checked_writes = dict(
            re.findall(r'\$\("([^\"]+)"\)\.checked=(true|false)', body)
        )
        self.assertGreaterEqual(len(checked_writes), 2, "checkbox resets are not covered")
        for control_id, reset_value in checked_writes.items():
            self.assertIn(control_id, tags_by_id, f"reset checkbox {control_id!r} has no input")
            markup_checked = bool(
                re.search(r"\schecked(?:\s|=|>)", tags_by_id[control_id], re.IGNORECASE)
            )
            self.assertEqual(
                reset_value == "true",
                markup_checked,
                f"reset checked state for {control_id!r} differs from markup",
            )

        # Dynamic component and event controls are rebuilt from these same canonical defaults.
        self.assertIn("comp=structuredClone(DEFAULT_COMP)", body)
        self.assertIn("ev=structuredClone(DEFAULT_EV)", body)

    def test_legacy_disclosure_bans_obsolete_inference_claims(self):
        forbidden = {
            "formal-null verdict": r"\bnull rejected\b",
            "GPS-cure requirement": r"\bGPS(?:-specific)? cure (?:is )?required\b",
            "durable-benefit requirement": r"\bGPS-specific durable benefit is required\b",
            "success badge": r"\brobustly positive\b",
            "posterior-like bracket": r"\boptimistic bracket\b",
            "thesis verdict": r"\bthesis-support(?:ing)?\b",
            "de-facto cure inference": r"\bde-facto cure\b",
            "old CLI headline": r"\bHEADLINE\s+PLATEAU\b",
            "old rejected status": r"·\s*REJECTED\b",
            "old figure title": r"PLATEAU\s+\(GPS-cure\)\s+probability of success",
        }
        for name, source in self.surfaces.items():
            for claim, pattern in forbidden.items():
                self.assertNotRegex(
                    source,
                    re.compile(pattern, re.IGNORECASE),
                    f"{name} still contains obsolete {claim} language",
                )

            normalized = source.replace("*", "").lower()
            self.assertIn("fixed-scenario", normalized, f"{name} omits the scenario qualifier")
            self.assertRegex(
                normalized,
                r"(?:not a posterior|does not estimate a posterior)",
                f"{name} omits the non-posterior disclosure",
            )


if __name__ == "__main__":
    unittest.main()
