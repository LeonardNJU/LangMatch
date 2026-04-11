from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def load_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "prepare_langmatch_manifests.py"
    )
    spec = importlib.util.spec_from_file_location(
        "prepare_langmatch_manifests", script_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class PromptModeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_hidden_mode_does_not_require_visible_reasoning(self):
        system_prompt, suffix = self.module.get_prompt_contract(
            benchmark="math500", condition="zh", mode="hidden"
        )

        self.assertIn("<answer>...</answer>", system_prompt)
        self.assertIn("不必展示推理过程", suffix)
        self.assertNotIn("写出推理过程", suffix)

    def test_compact_mode_allows_brief_visible_reasoning(self):
        system_prompt, suffix = self.module.get_prompt_contract(
            benchmark="mmlu_pro", condition="wy", mode="compact"
        )

        self.assertIn("<answer>...</answer>", system_prompt)
        self.assertIn("略陳", suffix)
        self.assertIn("一短段", suffix)
        self.assertNotIn("詳陳", suffix)

    def test_gpt5_translation_uses_max_completion_tokens(self):
        kwargs = self.module.get_generation_token_kwargs("gpt-5.4", 2048)
        self.assertEqual(kwargs, {"max_completion_tokens": 2048})

    def test_non_gpt5_translation_uses_max_tokens(self):
        kwargs = self.module.get_generation_token_kwargs("gpt-4o", 2048)
        self.assertEqual(kwargs, {"max_tokens": 2048})


if __name__ == "__main__":
    unittest.main()
