from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def load_module():
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "run_langmatch_experiment.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_langmatch_experiment", script_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class ThinkBlockSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_split_think_block_returns_hidden_and_visible_parts(self):
        hidden, visible = self.module.split_think_output(
            "<think>hidden chain</think>Visible summary\n<answer>C</answer>"
        )

        self.assertEqual(hidden, "hidden chain")
        self.assertEqual(visible, "Visible summary\n<answer>C</answer>")

    def test_split_think_block_preserves_plain_output(self):
        hidden, visible = self.module.split_think_output("<answer>81</answer>")

        self.assertEqual(hidden, "")
        self.assertEqual(visible, "<answer>81</answer>")


if __name__ == "__main__":
    unittest.main()
