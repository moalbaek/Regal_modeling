"""Static invariants for user-facing defaults and scenario-model disclosure."""

from decimal import Decimal
import os
import re
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAN_SURFACE_FILES = (
    "README.md",
    "regal_explorer.html",
    "regal_explorer.py",
    "REGAL_MODEL_DOCUMENTATION.md",
    "BAT_CONTROL_ARM_RESEARCH.md",
    "docs.html",
)
DISCLOSURE_FILES = (
    "README.md",
    "regal_explorer.html",
    "regal_explorer.py",
    "REGAL_MODEL_DOCUMENTATION.md",
)


class StaticConsistencyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.surfaces = {}
        for name in BAN_SURFACE_FILES:
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
        self.assertEqual(self._quoted_attr(tags_by_id["IA"], "min"), "1")
        self.assertEqual(self._quoted_attr(tags_by_id["FINAL"], "min"), "2")

    def test_scenario_disclosure_bans_obsolete_inference_claims(self):
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
        # These expressions are tripwires for the repo's known historical phrasings,
        # not semantic proof that every possible overclaim is absent. Some deliberately
        # also match a negated sentence; reviewer judgment remains the real guardrail.
        for name in BAN_SURFACE_FILES:
            source = self.surfaces[name]
            for claim, pattern in forbidden.items():
                self.assertNotRegex(
                    source,
                    re.compile(pattern, re.IGNORECASE),
                    f"{name} still contains obsolete {claim} language",
                )

        for name in DISCLOSURE_FILES:
            normalized = self.surfaces[name].replace("*", "").lower()
            self.assertIn("fixed-scenario", normalized, f"{name} omits the scenario qualifier")
            self.assertRegex(
                normalized,
                r"(?:not a posterior|does not estimate a posterior)",
                f"{name} omits the non-posterior disclosure",
            )

    def test_user_facing_surfaces_use_v1_scenario_terminology(self):
        for name in BAN_SURFACE_FILES:
            self.assertNotRegex(
                self.surfaces[name],
                re.compile(r"\blegacy\b", re.IGNORECASE),
                f"{name} still labels the v1 scenario model as legacy",
            )

    def test_explorer_lands_on_v1_with_accessible_model_tabs(self):
        v1_tab = re.search(r'<button\b[^>]*\bid="tab-v1"[^>]*>', self.html)
        v2_tab = re.search(r'<button\b[^>]*\bid="tab-v2"[^>]*>', self.html)
        v1_panel = re.search(r'<div\b[^>]*\bid="panel-v1"[^>]*>', self.html)
        v2_panel = re.search(r'<section\b[^>]*\bid="panel-v2"[^>]*>', self.html)
        for match, label in (
            (v1_tab, "V1 tab"),
            (v2_tab, "V2 tab"),
            (v1_panel, "V1 panel"),
            (v2_panel, "V2 panel"),
        ):
            self.assertIsNotNone(match, f"{label} not found")

        self.assertEqual(self._quoted_attr(v1_tab.group(), "aria-selected"), "true")
        self.assertEqual(self._quoted_attr(v1_tab.group(), "tabindex"), "0")
        self.assertEqual(self._quoted_attr(v2_tab.group(), "aria-selected"), "false")
        self.assertEqual(self._quoted_attr(v2_tab.group(), "tabindex"), "-1")
        self.assertNotRegex(v1_panel.group(), r"\shidden(?:\s|=|>)")
        self.assertRegex(v2_panel.group(), r"\shidden(?:\s|=|>)")

        self.assertRegex(
            self.html,
            r'location\.hash\.replace\(/\^#/,\s*["\']{2}\)\s*\|\|\s*["\']v1["\']',
        )
        self.assertIn('window.addEventListener("hashchange"', self.html)
        select_tab = re.search(
            r"function\s+selectModelTab\([^)]*\)\s*\{(?P<body>.*?)\n\}",
            self.html,
            re.DOTALL,
        )
        self.assertIsNotNone(select_tab, "model-tab selection function not found")
        self.assertRegex(
            select_tab.group("body"),
            r'view\.id\s*===\s*["\']v1["\'][\s\S]*?schedule\(\)',
        )
        for key in ("ArrowRight", "ArrowLeft", "Home", "End"):
            self.assertIn(f'e.key==="{key}"', self.html)

    def test_tab_switches_keep_the_v1_layout_scale_stable(self):
        compact = re.sub(r"\s+", "", self.html)
        self.assertIn("-webkit-text-size-adjust:100%", compact)
        self.assertIn("text-size-adjust:100%", compact)
        self.assertIn("scrollbar-gutter:stable", compact)
        self.assertRegex(
            self.html,
            re.compile(
                r'window\.addEventListener\(\s*["\']resize["\'][\s\S]*?'
                r'panel-v1[\s\S]*?\.hidden[\s\S]*?schedule\(\)',
            ),
        )


if __name__ == "__main__":
    unittest.main()
