import math
import ast
import unittest
from pathlib import Path

_GENERATOR_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_fidelity_baseline.py"
_TREE = ast.parse(_GENERATOR_PATH.read_text(encoding="utf-8"))
_NODES = [node for node in _TREE.body if isinstance(node, ast.FunctionDef)
          and node.name in {"generation_deadline", "warning_threshold_exceeded"}]
_NAMESPACE = {"float": float}
exec(compile(ast.Module(body=_NODES, type_ignores=[]), str(_GENERATOR_PATH), "exec"), _NAMESPACE)
generation_deadline = _NAMESPACE["generation_deadline"]
warning_threshold_exceeded = _NAMESPACE["warning_threshold_exceeded"]


class FidelityRuntimePolicyTest(unittest.TestCase):
    def test_warning_only_policy_has_no_deadline_even_after_resume(self):
        deadline = generation_deadline(
            1000.0, 12.0, 13 * 3600.0, continue_after_warning=True)
        self.assertTrue(math.isinf(deadline))

    def test_default_policy_retains_registered_hard_deadline(self):
        deadline = generation_deadline(1000.0, 12.0, 3600.0)
        self.assertEqual(deadline, 1000.0 + 12 * 3600.0 - 3600.0)

    def test_warning_threshold_is_observational(self):
        self.assertFalse(warning_threshold_exceeded(11 * 3600.0, 12.0))
        self.assertTrue(warning_threshold_exceeded(13 * 3600.0, 12.0))

    def test_invalid_runtime_policy_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            generation_deadline(1000.0, 0.0)
        with self.assertRaises(ValueError):
            warning_threshold_exceeded(-1.0, 12.0)


if __name__ == "__main__":
    unittest.main()
