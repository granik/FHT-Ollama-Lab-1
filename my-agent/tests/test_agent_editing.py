import json
from pathlib import Path
import tempfile
import unittest

from student_agent import Runtime, run_agent


class ScriptedModel:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.catalogs = []

    def set_tool_catalog(self, catalog):
        self.catalogs.append(catalog)

    def __call__(self, _messages):
        return json.dumps(next(self.actions))


class AgentEditingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "pkg").mkdir()
        (self.root / "pkg/example.py").write_text(
            "def value():\n    return 1\n", encoding="utf-8"
        )

    def run_actions(self, actions, task, turns=None):
        events = []
        result = run_agent(
            ScriptedModel(actions),
            Runtime(self.root, mode="edit"),
            task,
            max_turns=turns or len(actions),
            emit=events.append,
        )
        return result, events

    def test_valid_partial_python_edit_is_accepted(self):
        result, events = self.run_actions(
            [
                {"tool": "read_file", "args": {"path": "pkg/example.py"}},
                {
                    "tool": "edit_file",
                    "args": {
                        "path": "pkg/example.py",
                        "old": "    return 1",
                        "new": "    result = 2\n    return result",
                    },
                },
                {"final": "done"},
            ],
            "Repair pkg/example.py after reading it.",
        )
        self.assertEqual(result["termination"], "final")
        self.assertIn("result = 2", (self.root / "pkg/example.py").read_text())
        edit_results = [
            event["result"]
            for event in events
            if event.get("event") == "result" and event.get("turn") == 2
        ]
        self.assertEqual(edit_results[0]["status"], "ok")

    def test_unique_partial_edit_recovers_omitted_indentation(self):
        result, events = self.run_actions(
            [
                {"tool": "read_file", "args": {"path": "pkg/example.py"}},
                {
                    "tool": "edit_file",
                    "args": {
                        "path": "pkg/example.py",
                        "old": "return 1",
                        "new": "result = 2\nreturn result",
                    },
                },
                {"final": "done"},
            ],
            "Repair pkg/example.py after reading it.",
        )
        self.assertEqual(result["termination"], "final")
        self.assertEqual(
            (self.root / "pkg/example.py").read_text(),
            "def value():\n    result = 2\n    return result\n",
        )
        edit_result = next(
            event["result"]
            for event in events
            if event.get("event") == "result" and event.get("turn") == 2
        )
        self.assertEqual(edit_result["status"], "ok")

    def test_module_preamble_and_loose_function_body_are_recovered(self):
        path = self.root / "pkg/example.py"
        path.write_text(
            "def value(x):\n"
            "    if x < 0:\n"
            "        raise ValueError('negative')\n"
            "    result = x + 1\n"
            "    return result\n",
            encoding="utf-8",
        )
        malformed = (
            "def double(x):\n"
            "    return x * 2\n\n"
            "result = double(x)\n"
            "return result"
        )
        result, events = self.run_actions(
            [
                {"tool": "read_file", "args": {"path": "pkg/example.py"}},
                {
                    "tool": "edit_file",
                    "args": {
                        "path": "pkg/example.py",
                        "old": "result = x + 1\nreturn result",
                        "new": malformed,
                    },
                },
                {"final": "done"},
            ],
            "Repair pkg/example.py after reading it.",
        )
        self.assertEqual(result["termination"], "final")
        updated = path.read_text()
        compile(updated, str(path), "exec")
        namespace = {}
        exec(updated, namespace)
        self.assertEqual(namespace["value"](3), 6)
        with self.assertRaises(ValueError):
            namespace["value"](-1)
        edit_result = next(
            event["result"]
            for event in events
            if event.get("event") == "result" and event.get("turn") == 2
        )
        self.assertEqual(edit_result["status"], "ok")

    def test_exact_qwen_hybrid_pricing_edit_is_recovered(self):
        (self.root / "orders").mkdir()
        (self.root / "orders/validation.py").write_text(
            "def validate_order(lines, discount):\n    pass\n", encoding="utf-8"
        )
        path = self.root / "orders/pricing.py"
        path.write_text(
            '"""Pricing calculation."""\n'
            "from decimal import Decimal\n"
            "from .validation import validate_order\n\n\n"
            'def quote(lines, discount=Decimal("0")):\n'
            "    validate_order(lines, discount)\n"
            '    subtotal = sum((line["unit_price"] * line["quantity"] for line in lines), Decimal("0"))\n'
            '    shipping = Decimal("0") if subtotal >= Decimal("100") else Decimal("5")\n'
            '    total = ((subtotal + shipping) * (1 - discount)).quantize(Decimal("0.01"))\n'
            '    return {"subtotal": subtotal, "shipping": shipping, "total": total}\n',
            encoding="utf-8",
        )
        old = (
            'subtotal = sum((line["unit_price"] * line["quantity"] for line in lines), Decimal("0"))\n'
            'shipping = Decimal("0") if subtotal >= Decimal("100") else Decimal("5")\n'
            'total = ((subtotal + shipping) * (1 - discount)).quantize(Decimal("0.01"))'
        )
        hybrid = (
            "from decimal import Decimal, ROUND_HALF_UP\n\n"
            "def calculate_line_total(line):\n"
            '    return (line["unit_price"] * line["quantity"]).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)\n\n'
            "subtotal = sum(calculate_line_total(line) for line in lines)\n"
            'discount_amount = (subtotal * discount).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)\n'
            "discounted_goods = subtotal - discount_amount\n"
            'shipping = Decimal("0") if discounted_goods >= Decimal("100") else Decimal("5")\n'
            'total = (discounted_goods + shipping).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)\n\n'
            'return {"subtotal": subtotal, "shipping": shipping, "total": total}'
        )
        result, events = self.run_actions(
            [
                {"tool": "read_file", "args": {"path": "orders/pricing.py"}},
                {
                    "tool": "edit_file",
                    "args": {"path": "orders/pricing.py", "old": old, "new": hybrid},
                },
                {"final": "done"},
            ],
            "Repair orders/pricing.py after reading it.",
        )
        self.assertEqual(result["termination"], "final")
        updated = path.read_text()
        compile(updated, str(path), "exec")
        self.assertIn("discounted_goods", updated)
        edit_result = next(
            event["result"]
            for event in events
            if event.get("event") == "result" and event.get("turn") == 2
        )
        self.assertEqual(edit_result["status"], "ok")

    def test_complete_module_in_new_is_applied_atomically(self):
        complete = "def value():\n    result = 3\n    return result\n"
        result, _events = self.run_actions(
            [
                {"tool": "read_file", "args": {"path": "pkg/example.py"}},
                {
                    "tool": "edit_file",
                    "args": {
                        "path": "pkg/example.py",
                        "old": "    return 1",
                        "new": complete,
                    },
                },
                {"final": "done"},
            ],
            "Repair pkg/example.py after reading it.",
        )
        self.assertEqual(result["termination"], "final")
        self.assertEqual((self.root / "pkg/example.py").read_text(), complete)

    def test_complete_module_preserves_an_omitted_existing_import(self):
        path = self.root / "pkg/example.py"
        path.write_text(
            "import math\n\ndef value():\n    return math.floor(1.5)\n",
            encoding="utf-8",
        )
        proposed = "def value():\n    return math.ceil(1.5)\n"
        result, _events = self.run_actions(
            [
                {"tool": "read_file", "args": {"path": "pkg/example.py"}},
                {
                    "tool": "edit_file",
                    "args": {
                        "path": "pkg/example.py",
                        "old": "return math.floor(1.5)",
                        "new": proposed,
                    },
                },
                {"final": "done"},
            ],
            "Repair pkg/example.py after reading it.",
        )
        self.assertEqual(result["termination"], "final")
        updated = path.read_text()
        self.assertIn("import math", updated)
        self.assertIn("return math.ceil(1.5)", updated)

    def test_invalid_regression_test_is_not_silently_replaced(self):
        (self.root / "tests").mkdir()
        bad_test = (
            "import unittest\n"
            "from pkg.example import missing\n\n"
            "class Regression(unittest.TestCase):\n"
            "    def test_value(self):\n"
            "        self.assertEqual(missing(), 2)\n"
        )
        _result, events = self.run_actions(
            [
                {
                    "tool": "write_file",
                    "args": {"path": "tests/test_student.py", "content": bad_test},
                }
            ],
            "Add one useful regression test in tests/test_student.py.",
            turns=1,
        )
        self.assertFalse((self.root / "tests/test_student.py").exists())
        tool_result = next(
            event["result"]
            for event in events
            if event.get("event") == "result" and event.get("turn") == 1
        )
        self.assertEqual(tool_result["status"], "error")
        self.assertIn("does not define imported names: missing", str(tool_result))

    def test_regression_test_cannot_assert_an_unreturned_result_key(self):
        (self.root / "tests").mkdir()
        bad_test = (
            "import unittest\n"
            "from pkg.example import value\n\n"
            "class Regression(unittest.TestCase):\n"
            "    def test_value(self):\n"
            "        result = value()\n"
            "        self.assertEqual(result['missing'], 2)\n"
        )
        (self.root / "pkg/example.py").write_text(
            "def value():\n    return {'present': 2}\n", encoding="utf-8"
        )
        _result, events = self.run_actions(
            [
                {
                    "tool": "write_file",
                    "args": {"path": "tests/test_student.py", "content": bad_test},
                }
            ],
            "Add one useful regression test in tests/test_student.py.",
            turns=1,
        )
        self.assertFalse((self.root / "tests/test_student.py").exists())
        tool_result = next(
            event["result"]
            for event in events
            if event.get("event") == "result" and event.get("turn") == 1
        )
        self.assertIn("does not return keys: missing", str(tool_result))

    def test_invalid_result_key_assertions_are_removed_when_valid_ones_remain(self):
        (self.root / "tests").mkdir()
        (self.root / "pkg/example.py").write_text(
            "def value():\n    return {'present': 2}\n", encoding="utf-8"
        )
        proposed = (
            "import unittest\n"
            "from pkg.example import value\n\n"
            "class Regression(unittest.TestCase):\n"
            "    def test_value(self):\n"
            "        result = value()\n"
            "        self.assertEqual(result['present'], 2)\n"
            "        self.assertEqual(result['missing'], 3)\n"
        )
        result, events = self.run_actions(
            [
                {
                    "tool": "write_file",
                    "args": {"path": "tests/test_student.py", "content": proposed},
                },
                {"final": "done"},
            ],
            "Add one useful regression test in tests/test_student.py.",
        )
        self.assertEqual(result["termination"], "final")
        written = (self.root / "tests/test_student.py").read_text()
        self.assertIn("result['present']", written)
        self.assertNotIn("result['missing']", written)
        tool_result = next(
            event["result"]
            for event in events
            if event.get("event") == "result" and event.get("turn") == 1
        )
        self.assertEqual(tool_result["status"], "ok")
        self.assertIn("Removed assertions", str(tool_result))

    def test_test_tool_schema_names_the_local_function_result_keys(self):
        (self.root / "pkg/example.py").write_text(
            "def value():\n    return {'present': 2}\n", encoding="utf-8"
        )
        (self.root / "tests").mkdir()
        model = ScriptedModel(
            [
                {"tool": "read_file", "args": {"path": "pkg/example.py"}},
                {
                    "tool": "edit_file",
                    "args": {
                        "path": "pkg/example.py",
                        "old": "return {'present': 2}",
                        "new": "return {'present': 3}",
                    },
                },
                {
                    "tool": "write_file",
                    "args": {
                        "path": "tests/test_student.py",
                        "content": (
                            "import unittest\n"
                            "from pkg.example import value\n\n"
                            "class Regression(unittest.TestCase):\n"
                            "    def test_value(self):\n"
                            "        result = value()\n"
                            "        self.assertEqual(result['present'], 3)\n"
                        ),
                    },
                },
            ]
        )
        run_agent(
            model,
            Runtime(self.root, mode="edit"),
            (
                "Repair pkg/example.py after reading it. Add one useful regression "
                "test in tests/test_student.py."
            ),
            max_turns=3,
        )
        descriptions = [
            catalog["write_file"]["argument_schema"]["content"]["description"]
            for catalog in model.catalogs
            if set(catalog) == {"write_file"}
        ]
        self.assertTrue(descriptions)
        self.assertIn("value returns dict keys only: present", descriptions[-1])

    def test_empty_regression_test_is_rejected(self):
        (self.root / "tests").mkdir()
        empty_test = (
            "import unittest\n\n"
            "class Regression(unittest.TestCase):\n"
            "    def test_value(self):\n"
            "        pass\n"
        )
        _result, events = self.run_actions(
            [
                {
                    "tool": "write_file",
                    "args": {"path": "tests/test_student.py", "content": empty_test},
                }
            ],
            "Add one useful regression test in tests/test_student.py.",
            turns=1,
        )
        self.assertFalse((self.root / "tests/test_student.py").exists())
        tool_result = next(
            event["result"]
            for event in events
            if event.get("event") == "result" and event.get("turn") == 1
        )
        self.assertIn("has no assertion", str(tool_result))

    def test_failed_verification_can_repair_agent_created_test(self):
        (self.root / "tests").mkdir()
        (self.root / "tests/test_student.py").write_text(
            "import unittest\n\n"
            "class StudentRegression(unittest.TestCase):\n"
            "    def test_regression(self):\n"
            "        self.assertEqual(1, 2)\n",
            encoding="utf-8",
        )
        task = (
            "Read pkg/example.py. Add one useful regression test in "
            "tests/test_student.py. Request approval to run "
            "python3 -m unittest discover -s tests -v."
        )
        model = ScriptedModel(
            [
                {"tool": "read_file", "args": {"path": "pkg/example.py"}},
                {"tool": "read_file", "args": {"path": "tests/test_student.py"}},
                {
                    "tool": "bash",
                    "args": {"command": "python3 -m unittest discover -s tests -v"},
                },
                {
                    "tool": "edit_file",
                    "args": {
                        "path": "tests/test_student.py",
                        "old": "self.assertEqual(1, 2)",
                        "new": "self.assertEqual(1, 1)",
                    },
                },
                {
                    "tool": "bash",
                    "args": {"command": "python3 -m unittest discover -s tests -v"},
                },
            ]
        )
        result = run_agent(
            model,
            Runtime(self.root, mode="edit", approve=lambda *_args: True),
            task,
            max_turns=5,
        )
        self.assertEqual(result["termination"], "final")
        self.assertIn(
            "self.assertEqual(1, 1)",
            (self.root / "tests/test_student.py").read_text(),
        )
        editable_path_sets = [
            set(catalog["edit_file"]["argument_schema"]["path"].get("enum", []))
            for catalog in model.catalogs
            if "edit_file" in catalog
            and "path" in catalog["edit_file"].get("argument_schema", {})
        ]
        self.assertIn(
            {"tests/test_student.py"}, editable_path_sets
        )

    def test_failed_verification_allows_all_responsible_repairs_before_rerun(self):
        (self.root / "tests").mkdir()
        (self.root / "tests/test_acceptance.py").write_text(
            "import unittest\n"
            "from pkg.example import value\n\n"
            "class Acceptance(unittest.TestCase):\n"
            "    def test_value(self):\n"
            "        self.assertEqual(value(), 3)\n",
            encoding="utf-8",
        )
        (self.root / "tests/test_student.py").write_text(
            "import unittest\n"
            "from pkg.example import value\n\n"
            "class StudentRegression(unittest.TestCase):\n"
            "    def test_value_is_positive(self):\n"
            "        self.assertGreater(value(), 10)\n",
            encoding="utf-8",
        )
        task = (
            "Repair pkg/example.py after reading pkg/example.py and "
            "tests/test_acceptance.py. Add one useful regression test in "
            "tests/test_student.py. Request approval to run "
            "python3 -m unittest discover -s tests -v."
        )
        model = ScriptedModel(
            [
                {"tool": "read_file", "args": {"path": "pkg/example.py"}},
                {"tool": "read_file", "args": {"path": "tests/test_acceptance.py"}},
                {"tool": "read_file", "args": {"path": "tests/test_student.py"}},
                {
                    "tool": "bash",
                    "args": {"command": "python3 -m unittest discover -s tests -v"},
                },
                {
                    "tool": "edit_file",
                    "args": {
                        "path": "pkg/example.py",
                        "old": "return 1",
                        "new": "return 3",
                    },
                },
                {
                    "tool": "edit_file",
                    "args": {
                        "path": "tests/test_student.py",
                        "old": "self.assertGreater(value(), 10)",
                        "new": "self.assertGreater(value(), 0)",
                    },
                },
                {
                    "tool": "bash",
                    "args": {"command": "python3 -m unittest discover -s tests -v"},
                },
            ]
        )
        result = run_agent(
            model,
            Runtime(self.root, mode="edit", approve=lambda *_args: True),
            task,
            max_turns=7,
        )
        self.assertEqual(result["termination"], "final")
        self.assertIn("return 3", (self.root / "pkg/example.py").read_text())
        self.assertIn(
            "self.assertGreater(value(), 0)",
            (self.root / "tests/test_student.py").read_text(),
        )

    def test_existing_completed_work_is_verified_without_a_noop_edit(self):
        (self.root / "pkg/example.py").write_text(
            "def value():\n    return 2\n", encoding="utf-8"
        )
        (self.root / "tests").mkdir()
        (self.root / "tests/test_student.py").write_text(
            "import unittest\n"
            "from pkg.example import value\n\n"
            "class StudentRegression(unittest.TestCase):\n"
            "    def test_value(self):\n"
            "        self.assertEqual(value(), 2)\n",
            encoding="utf-8",
        )
        task = (
            "Repair pkg/example.py after reading it. Add one useful regression test "
            "in tests/test_student.py. Request approval to run "
            "python3 -m unittest discover -s tests -v."
        )
        model = ScriptedModel(
            [
                {"tool": "read_file", "args": {"path": "pkg/example.py"}},
                {"tool": "read_file", "args": {"path": "tests/test_student.py"}},
                {
                    "tool": "bash",
                    "args": {"command": "python3 -m unittest discover -s tests -v"},
                },
            ]
        )
        result = run_agent(
            model,
            Runtime(self.root, mode="edit", approve=lambda *_args: True),
            task,
            max_turns=3,
        )
        self.assertEqual(result["termination"], "final")
        self.assertIn("no files", result["final"])
        self.assertEqual(
            (self.root / "pkg/example.py").read_text(),
            "def value():\n    return 2\n",
        )


if __name__ == "__main__":
    unittest.main()
