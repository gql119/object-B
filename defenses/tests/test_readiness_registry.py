import json
from pathlib import Path
import unittest


class ReadinessRegistryTests(unittest.TestCase):
    def test_no_method_is_marked_ready_without_implementation(self):
        registry_path = Path(__file__).parents[1] / "defense_readiness.json"
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        self.assertFalse(registry["formal_launch_authorized"])
        self.assertFalse(registry["ready_for_gpu"])
        self.assertEqual(set(registry["methods"]), {"OP", "AVATAR", "D-VAE", "LE"})
        for method in registry["methods"].values():
            self.assertNotEqual(method["status"], "READY_VERIFIED")
            self.assertFalse(method["detector_adapter_present"])
            self.assertFalse(method["verified_checkpoint"])
            self.assertFalse(method["formal_clean_poison_runs"])


if __name__ == "__main__":
    unittest.main()
