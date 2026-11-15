import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from pipeline import load_catalog, load_reference_rules, validate_rule

class BaselineCompatibilityTests(unittest.TestCase):
    def test_all_47_reference_rules_validate(self):
        catalog = load_catalog()
        rules = load_reference_rules()
        seen = set()
        ids = {r["ruleId"] for r in rules}
        errors = []
        for rule in rules:
            # Existing IDs are allowed when validating the baseline itself.
            errors.extend(f"{rule.get('ruleId')}: {e}" for e in validate_rule(rule, catalog, set(), seen))
        self.assertEqual(47, len(rules))
        self.assertEqual([], errors)

if __name__ == "__main__":
    unittest.main()
