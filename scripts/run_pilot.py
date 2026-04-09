from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm


PROMPT_VARIANTS = {
    "base": (
        "You are a helpful assistant. Follow the user's instructions carefully. "
        "Except for code and direct quotations, keep explanations and narration concise. "
        "Avoid colloquial filler. Eliminate redundancy and aim for compact, clear expression."
    ),
    "zh_compact": (
        "你是一个有帮助的助手。请严格遵循用户要求。"
        "除代码和直接指令外，所有解释与叙述都尽量写得简洁。"
        "禁止口语化赘述，删繁就简，要求表达精炼、意思清楚。"
    ),
    "wy": (
        "汝为善应人问之助手，当谨循其命。凡码与引文外，释理叙事宜从简。"
        "禁绝白话冗词，务求辞约义明。"
    ),
}


SUPPORTED_IFEVAL_INSTRUCTIONS = {
    "punctuation:no_comma",
    "change_case:english_lowercase",
    "change_case:english_capital",
    "change_case:capital_word_frequency",
    "detectable_format:json_format",
    "detectable_format:number_bullet_lists",
    "detectable_format:number_highlighted_sections",
    "detectable_format:multiple_sections",
    "detectable_format:title",
    "detectable_content:number_placeholders",
    "detectable_content:postscript",
    "startend:end_checker",
    "startend:quotation",
    "keywords:existence",
    "keywords:frequency",
    "keywords:forbidden_words",
    "length_constraints:number_words",
    "length_constraints:number_sentences",
    "length_constraints:number_paragraphs",
    "combination:two_responses",
    "combination:repeat_prompt",
}

ANSWER_LETTERS = [chr(ord("A") + i) for i in range(10)]


@dataclass
class TaskExample:
    benchmark: str
    example_id: str
    prompt: str
    metadata: dict[str, Any]
    max_tokens: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small WYW prompt pilot.")
    parser.add_argument(
        "--backend", choices=["openai", "transformers"], default="openai"
    )
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--local-model-path", default=None)
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--ifeval-samples", type=int, default=18)
    parser.add_argument("--mmlu-samples", type=int, default=28)
    parser.add_argument("--math500-samples", type=int, default=0)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--max-tokens-override", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--conditions", nargs="+", default=["base", "zh_compact", "wy"])
    return parser.parse_args()


def normalize_base_url(raw: str | None) -> str:
    if not raw:
        raise ValueError("Missing base URL. Pass --base-url or set OPENAI_BASE_URL.")
    url = raw.rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


def build_client(args: argparse.Namespace) -> OpenAI:
    if not args.api_key:
        raise ValueError("Missing API key. Pass --api-key or set OPENAI_API_KEY.")
    return OpenAI(api_key=args.api_key, base_url=normalize_base_url(args.base_url))


def build_transformers_backend(args: argparse.Namespace) -> tuple[Any, Any]:
    import torch

    if not args.local_model_path:
        raise ValueError(
            "Missing local model path. Pass --local-model-path for transformers backend."
        )
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.local_model_path, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.local_model_path,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def apply_max_tokens_override(
    tasks: list[TaskExample], max_tokens_override: int | None
) -> list[TaskExample]:
    if max_tokens_override is None:
        return tasks
    return [
        TaskExample(
            benchmark=task.benchmark,
            example_id=task.example_id,
            prompt=task.prompt,
            metadata=task.metadata,
            max_tokens=max_tokens_override,
        )
        for task in tasks
    ]


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text, flags=re.UNICODE))


def count_sentences(text: str) -> int:
    parts = re.split(r"(?<=[.!?。！？])\s+|\n+", text.strip())
    return sum(1 for part in parts if part.strip())


def count_paragraphs(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    divider_matches = re.split(r"\n\s*\*{3,}\s*\n", stripped)
    if len(divider_matches) > 1:
        return sum(1 for part in divider_matches if part.strip())
    blank_split = re.split(r"\n\s*\n", stripped)
    return sum(1 for part in blank_split if part.strip())


def relation_holds(observed: int, relation: str | None, expected: int | None) -> bool:
    if expected is None:
        return True
    if relation in (None, "exactly"):
        return observed == expected
    if relation == "at least":
        return observed >= expected
    if relation == "less than":
        return observed < expected
    if relation == "more than":
        return observed > expected
    if relation == "at most":
        return observed <= expected
    raise ValueError(f"Unsupported relation: {relation}")


def count_highlighted_sections(text: str) -> int:
    matches = re.findall(r"(?<!\*)\*([^\n*][^*]*?)\*(?!\*)", text)
    return sum(1 for match in matches if match.strip())


def count_bullets(text: str) -> int:
    return len(re.findall(r"(?m)^\*\s+", text))


def count_placeholders(text: str) -> int:
    return len(re.findall(r"\[[^\[\]\n]+\]", text))


def count_capital_words(text: str) -> int:
    return len(re.findall(r"\b[A-Z]{2,}\b", text))


def keyword_count(text: str, keyword: str) -> int:
    return len(re.findall(re.escape(keyword), text, flags=re.IGNORECASE))


def infer_expected_count(kwargs_item: dict[str, Any]) -> int | None:
    for key in [
        "num_words",
        "num_bullets",
        "num_sections",
        "num_highlights",
        "num_placeholders",
        "capital_frequency",
    ]:
        value = kwargs_item.get(key)
        if value is not None:
            return int(value)
    return None


def is_supported_ifeval(example: dict[str, Any]) -> bool:
    ids = example["instruction_id_list"]
    if not ids or any(inst not in SUPPORTED_IFEVAL_INSTRUCTIONS for inst in ids):
        return False
    if "http://" in example["prompt"] or "https://" in example["prompt"]:
        return False
    for kwargs_item in example["kwargs"]:
        if not isinstance(kwargs_item, dict):
            return False
        num_words = kwargs_item.get("num_words")
        if num_words is not None and int(num_words) > 180:
            return False
        num_sections = kwargs_item.get("num_sections")
        if num_sections is not None and int(num_sections) > 8:
            return False
        num_bullets = kwargs_item.get("num_bullets")
        if num_bullets is not None and int(num_bullets) > 8:
            return False
        num_highlights = kwargs_item.get("num_highlights")
        if num_highlights is not None and int(num_highlights) > 8:
            return False
    return True


def load_ifeval_tasks(sample_count: int, seed: int) -> list[TaskExample]:
    dataset = load_dataset("google/IFEval", split="train")
    supported = [example for example in dataset if is_supported_ifeval(example)]

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in supported:
        key = example["instruction_id_list"][0]
        grouped[key].append(example)

    rng = random.Random(seed)
    for items in grouped.values():
        rng.shuffle(items)

    ordered_keys = sorted(
        grouped, key=lambda key: (len(grouped[key]), key), reverse=True
    )
    selected: list[dict[str, Any]] = []
    round_index = 0
    while len(selected) < min(sample_count, len(supported)):
        progressed = False
        for key in ordered_keys:
            items = grouped[key]
            if round_index < len(items) and len(selected) < sample_count:
                selected.append(items[round_index])
                progressed = True
        if not progressed:
            break
        round_index += 1

    tasks = []
    for example in selected:
        tasks.append(
            TaskExample(
                benchmark="ifeval",
                example_id=str(example["key"]),
                prompt=example["prompt"],
                metadata={
                    "instruction_id_list": example["instruction_id_list"],
                    "kwargs": example["kwargs"],
                },
                max_tokens=192,
            )
        )
    return tasks


def build_mmlu_prompt(example: dict[str, Any]) -> str:
    option_lines = [
        f"{ANSWER_LETTERS[i]}. {option}" for i, option in enumerate(example["options"])
    ]
    options_text = "\n".join(option_lines)
    return (
        f"Question: {example['question']}\n\n"
        f"Options:\n{options_text}\n\n"
        "Respond with the single correct option letter only."
    )


def load_mmlu_tasks(sample_count: int, seed: int) -> list[TaskExample]:
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in dataset:
        grouped[example["category"]].append(example)

    rng = random.Random(seed)
    for items in grouped.values():
        rng.shuffle(items)

    categories = sorted(grouped)
    selected: list[dict[str, Any]] = []
    round_index = 0
    while len(selected) < min(sample_count, len(dataset)):
        progressed = False
        for category in categories:
            items = grouped[category]
            if round_index < len(items) and len(selected) < sample_count:
                selected.append(items[round_index])
                progressed = True
        if not progressed:
            break
        round_index += 1

    tasks = []
    for example in selected:
        tasks.append(
            TaskExample(
                benchmark="mmlu_pro",
                example_id=str(example["question_id"]),
                prompt=build_mmlu_prompt(example),
                metadata={
                    "answer": example["answer"],
                    "category": example["category"],
                    "question": example["question"],
                },
                max_tokens=8,
            )
        )
    return tasks


def has_asy_markup(text: str) -> bool:
    return "[asy]" in text or "\\begin{asy}" in text


def build_math500_prompt(example: dict[str, Any]) -> str:
    return (
        f"Problem / 题目:\n{example['problem']}\n\n"
        "Solve the problem. 只输出最终答案，不要解释。"
        "Use compact ASCII math when possible: fractions as a/b, sqrt as sqrt(...), "
        "tuples as (a,b), lists comma-separated, and do not use \\boxed."
    )


def load_math500_tasks(sample_count: int, seed: int) -> list[TaskExample]:
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    filtered = [
        example for example in dataset if not has_asy_markup(example["problem"])
    ]

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for example in filtered:
        grouped[(example["subject"], int(example["level"]))].append(example)

    rng = random.Random(seed)
    for items in grouped.values():
        rng.shuffle(items)

    keys = sorted(grouped)
    selected: list[dict[str, Any]] = []
    round_index = 0
    while len(selected) < min(sample_count, len(filtered)):
        progressed = False
        for key in keys:
            items = grouped[key]
            if round_index < len(items) and len(selected) < sample_count:
                selected.append(items[round_index])
                progressed = True
        if not progressed:
            break
        round_index += 1

    tasks = []
    for example in selected:
        tasks.append(
            TaskExample(
                benchmark="math500",
                example_id=str(example["unique_id"]),
                prompt=build_math500_prompt(example),
                metadata={
                    "answer": example["answer"],
                    "subject": example["subject"],
                    "level": int(example["level"]),
                    "unique_id": example["unique_id"],
                },
                max_tokens=64,
            )
        )
    return tasks


def _replace_latex_frac(text: str) -> str:
    pattern = re.compile(r"\\(?:dfrac|tfrac|frac)\{([^{}]+)\}\{([^{}]+)\}")
    previous = None
    while previous != text:
        previous = text
        text = pattern.sub(lambda m: f"({m.group(1)})/({m.group(2)})", text)
    return text


def _replace_latex_sqrt(text: str) -> str:
    pattern = re.compile(r"\\sqrt\{([^{}]+)\}")
    previous = None
    while previous != text:
        previous = text
        text = pattern.sub(lambda m: f"sqrt({m.group(1)})", text)
    return text


def _strip_boxed(text: str) -> str:
    stripped = text.strip()
    previous = None
    pattern = re.compile(r"^\\boxed\{(.+)\}$", flags=re.DOTALL)
    while previous != stripped:
        previous = stripped
        match = pattern.match(stripped)
        if match:
            stripped = match.group(1).strip()
    return stripped


def normalize_math500_answer(text: str) -> str:
    normalized = text.strip()
    normalized = re.sub(r"(?is)^final answer\s*:\s*", "", normalized)
    normalized = re.sub(r"(?is)^answer\s*:\s*", "", normalized)
    normalized = normalized.strip().strip("`")
    normalized = _strip_boxed(normalized)
    normalized = normalized.replace("$", "")
    normalized = normalized.replace("\\left", "")
    normalized = normalized.replace("\\right", "")
    normalized = normalized.replace("\\!", "")
    normalized = normalized.replace("\\cdot", "*")
    normalized = normalized.replace("\\times", "*")
    normalized = normalized.replace("\\pi", "pi")
    normalized = normalized.replace("\\pm", "+/-")
    normalized = normalized.replace("^\\circ", "")
    normalized = re.sub(r"\\text\{([^{}]+)\}", r"\1", normalized)
    normalized = re.sub(r"\\operatorname\{([^{}]+)\}", r"\1", normalized)
    normalized = _replace_latex_frac(normalized)
    normalized = _replace_latex_sqrt(normalized)
    normalized = re.sub(r"\(([^()/]+)\)/\(([^()/]+)\)", r"\1/\2", normalized)
    normalized = normalized.replace("{", "")
    normalized = normalized.replace("}", "")
    normalized = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", normalized)
    normalized = re.sub(r"\s+", "", normalized)
    normalized = normalized.rstrip(".")
    return normalized.lower()


def split_top_level_commas(text: str) -> list[str]:
    parts = []
    current = []
    depth = 0
    for char in text:
        if char in "([":
            depth += 1
        elif char in ")]" and depth > 0:
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    parts.append("".join(current))
    return [part for part in parts if part]


def remove_leading_assignment(text: str) -> str:
    return re.sub(r"^[a-z]+\s*=\s*", "", text)


def expand_plus_minus(text: str) -> set[str]:
    if "+/-" not in text:
        return {text}
    return {text.replace("+/-", "+", 1), text.replace("+/-", "-", 1)}


def math500_candidate_forms(text: str) -> set[str]:
    candidates = {text}
    simplified = text
    for pattern in [
        r"^(?:the)?finalanswer(?:is)?[:=]?(.*)$",
        r"^(?:the)?answer(?:is)?[:=]?(.*)$",
    ]:
        match = re.match(pattern, simplified)
        if match and match.group(1):
            candidates.add(match.group(1))

    current = list(candidates)
    for item in current:
        if "=" in item:
            rhs = item.split("=")[-1]
            if rhs:
                candidates.add(rhs)
        stripped_assignment = remove_leading_assignment(item)
        if stripped_assignment:
            candidates.add(stripped_assignment)
    return {candidate for candidate in candidates if candidate}


def math500_answers_match(predicted: str, gold: str) -> bool:
    pred_norm = normalize_math500_answer(predicted)
    gold_norm = normalize_math500_answer(gold)
    candidates_pred = set()
    for item in math500_candidate_forms(pred_norm):
        candidates_pred |= expand_plus_minus(item)

    candidates_gold = set()
    for item in math500_candidate_forms(gold_norm):
        candidates_gold |= expand_plus_minus(item)

    if candidates_pred & candidates_gold:
        return True

    for pred_item in candidates_pred:
        for gold_item in candidates_gold:
            pred_parts = split_top_level_commas(pred_item)
            gold_parts = split_top_level_commas(gold_item)
            if (
                len(pred_parts) > 1
                and len(gold_parts) > 1
                and not pred_item.startswith("(")
                and not gold_item.startswith("(")
                and sorted(pred_parts) == sorted(gold_parts)
            ):
                return True
    return False


def evaluate_ifeval_response(task: TaskExample, response_text: str) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    ids = task.metadata["instruction_id_list"]
    kwargs_list = task.metadata["kwargs"]
    stripped = response_text.strip()
    lowered = stripped.lower()

    for instruction_id, kwargs_item in zip(ids, kwargs_list):
        passed = True
        detail: dict[str, Any] = {"instruction_id": instruction_id}

        if instruction_id == "punctuation:no_comma":
            passed = "," not in stripped

        elif instruction_id == "change_case:english_lowercase":
            passed = re.search(r"[A-Z]", stripped) is None

        elif instruction_id == "change_case:english_capital":
            passed = re.search(r"[a-z]", stripped) is None

        elif instruction_id == "change_case:capital_word_frequency":
            observed = count_capital_words(stripped)
            relation = kwargs_item.get("capital_relation")
            expected = kwargs_item.get("capital_frequency")
            if expected is not None:
                expected = int(expected)
            passed = relation_holds(observed, relation, expected)
            detail.update(
                {"observed": observed, "expected": expected, "relation": relation}
            )

        elif instruction_id == "detectable_format:json_format":
            candidate = strip_code_fences(stripped)
            try:
                json.loads(candidate)
                passed = True
            except json.JSONDecodeError:
                passed = False

        elif instruction_id == "detectable_format:number_bullet_lists":
            observed = count_bullets(stripped)
            expected = kwargs_item.get("num_bullets")
            if expected is not None:
                expected = int(expected)
            passed = relation_holds(observed, "exactly", expected)
            detail.update({"observed": observed, "expected": expected})

        elif instruction_id == "detectable_format:number_highlighted_sections":
            observed = count_highlighted_sections(stripped)
            expected = kwargs_item.get("num_highlights")
            if expected is not None:
                expected = int(expected)
            if expected is None:
                passed = observed >= 2
            else:
                passed = observed >= expected
            detail.update({"observed": observed, "expected": expected})

        elif instruction_id == "detectable_format:multiple_sections":
            splitter = kwargs_item.get("section_spliter")
            expected = kwargs_item.get("num_sections")
            if expected is not None:
                expected = int(expected)
            if splitter:
                observed = len(
                    re.findall(rf"(?mi)^\s*{re.escape(str(splitter))}\s+\d+", stripped)
                )
            else:
                observed = count_paragraphs(stripped)
            passed = relation_holds(observed, "exactly", expected)
            detail.update(
                {"observed": observed, "expected": expected, "splitter": splitter}
            )

        elif instruction_id == "detectable_format:title":
            passed = re.search(r"<<[^<>\n]+>>", stripped) is not None

        elif instruction_id == "detectable_content:number_placeholders":
            observed = count_placeholders(stripped)
            expected = kwargs_item.get("num_placeholders")
            if expected is not None:
                expected = int(expected)
            passed = relation_holds(observed, "at least", expected)
            detail.update({"observed": observed, "expected": expected})

        elif instruction_id == "detectable_content:postscript":
            marker = kwargs_item.get("postscript_marker")
            if marker:
                passed = bool(
                    re.search(rf"(?mi)^\s*{re.escape(str(marker))}", stripped)
                )
            else:
                passed = bool(re.search(r"(?mi)^P(?:\.P\.)?S\.", stripped))
            detail.update({"postscript_marker": marker})

        elif instruction_id == "startend:end_checker":
            target = (
                kwargs_item.get("end_phrase")
                or kwargs_item.get("end_text")
                or kwargs_item.get("keywords")
            )
            if isinstance(target, list):
                target = target[-1] if target else None
            if not target:
                match = re.search(r'"([^"]+)"\s*$', task.prompt)
                target = match.group(1) if match else None
            if target is None:
                raise ValueError(
                    f"Missing end checker target for task {task.example_id}"
                )
            passed = stripped.endswith(str(target))
            detail.update({"target": target})

        elif instruction_id == "startend:quotation":
            passed = (
                stripped.startswith('"')
                and stripped.endswith('"')
                and len(stripped) >= 2
            )

        elif instruction_id == "keywords:existence":
            keywords = kwargs_item.get("keywords") or []
            if not keywords and kwargs_item.get("keyword"):
                keywords = [kwargs_item["keyword"]]
            passed = all(keyword_count(stripped, keyword) >= 1 for keyword in keywords)
            detail.update({"keywords": keywords})

        elif instruction_id == "keywords:frequency":
            keywords = kwargs_item.get("keywords") or []
            if not keywords and kwargs_item.get("keyword"):
                keywords = [kwargs_item["keyword"]]
            relation = kwargs_item.get("relation") or "at least"
            expected = kwargs_item.get("frequency")
            if expected is None:
                expected = kwargs_item.get("capital_frequency")
            if expected is None:
                match = re.search(
                    r"at least\s+(\d+)\s+times", task.prompt, flags=re.IGNORECASE
                )
                expected = int(match.group(1)) if match else None
            if expected is not None:
                expected = int(expected)
            observed_map = {
                keyword: keyword_count(stripped, keyword) for keyword in keywords
            }
            passed = all(
                relation_holds(observed_map[keyword], relation, expected)
                for keyword in keywords
            )
            detail.update(
                {
                    "keywords": keywords,
                    "observed": observed_map,
                    "expected": expected,
                    "relation": relation,
                }
            )

        elif instruction_id == "keywords:forbidden_words":
            forbidden_words = kwargs_item.get("forbidden_words") or []
            if not forbidden_words:
                quoted = re.findall(r'"([^"]+)"', task.prompt)
                if quoted:
                    forbidden_words = quoted[-3:]
            passed = all(keyword_count(stripped, word) == 0 for word in forbidden_words)
            detail.update({"forbidden_words": forbidden_words})

        elif instruction_id == "length_constraints:number_words":
            observed = count_words(stripped)
            expected = kwargs_item.get("num_words")
            if expected is not None:
                expected = int(expected)
            relation = kwargs_item.get("relation")
            passed = relation_holds(observed, relation, expected)
            detail.update(
                {"observed": observed, "expected": expected, "relation": relation}
            )

        elif instruction_id == "length_constraints:number_sentences":
            observed = count_sentences(stripped)
            expected = kwargs_item.get("num_sentences")
            if expected is None:
                expected = infer_expected_count(kwargs_item)
            if expected is not None:
                expected = int(expected)
            relation = kwargs_item.get("relation")
            passed = relation_holds(observed, relation, expected)
            detail.update(
                {"observed": observed, "expected": expected, "relation": relation}
            )

        elif instruction_id == "length_constraints:number_paragraphs":
            observed = count_paragraphs(stripped)
            expected = kwargs_item.get("num_paragraphs")
            if expected is None:
                expected = infer_expected_count(kwargs_item)
            if expected is not None:
                expected = int(expected)
            relation = kwargs_item.get("relation")
            passed = relation_holds(observed, relation, expected)
            detail.update(
                {"observed": observed, "expected": expected, "relation": relation}
            )

        elif instruction_id == "combination:two_responses":
            parts = [part.strip() for part in stripped.split("******")]
            passed = len(parts) == 2 and all(parts)

        elif instruction_id == "combination:repeat_prompt":
            prompt_to_repeat = (
                kwargs_item.get("prompt_to_repeat")
                or task.prompt.split("\n")[0].strip()
            )
            passed = stripped.startswith(str(prompt_to_repeat))
            detail.update({"prompt_to_repeat": prompt_to_repeat})

        else:
            raise ValueError(
                f"Unsupported IFEval instruction at runtime: {instruction_id}"
            )

        detail["passed"] = passed
        checks.append(detail)

    return {
        "score": int(all(item["passed"] for item in checks)),
        "checks": checks,
        "response_text": stripped,
    }


def evaluate_mmlu_response(task: TaskExample, response_text: str) -> dict[str, Any]:
    match = re.search(r"\b([A-J])\b", response_text.upper())
    predicted = match.group(1) if match else None
    gold = task.metadata["answer"]
    return {
        "score": int(predicted == gold),
        "predicted": predicted,
        "gold": gold,
        "response_text": response_text.strip(),
    }


def evaluate_math500_response(task: TaskExample, response_text: str) -> dict[str, Any]:
    gold = task.metadata["answer"]
    predicted = response_text.strip()
    return {
        "score": int(math500_answers_match(predicted, gold)),
        "predicted": predicted,
        "gold": gold,
        "predicted_normalized": normalize_math500_answer(predicted),
        "gold_normalized": normalize_math500_answer(gold),
        "response_text": predicted,
    }


def evaluate_task(task: TaskExample, response_text: str) -> dict[str, Any]:
    if task.benchmark == "ifeval":
        return evaluate_ifeval_response(task, response_text)
    if task.benchmark == "mmlu_pro":
        return evaluate_mmlu_response(task, response_text)
    if task.benchmark == "math500":
        return evaluate_math500_response(task, response_text)
    raise ValueError(f"Unsupported benchmark: {task.benchmark}")


def request_completion(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float = 0.0,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            request_kwargs: dict[str, Any] = {
                "model": model,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
            if model.startswith("gpt-5"):
                request_kwargs["max_completion_tokens"] = max_tokens
            else:
                request_kwargs["max_tokens"] = max_tokens

            response = client.chat.completions.create(**request_kwargs)
            content = response.choices[0].message.content or ""
            usage = response.usage
            if usage is None:
                raise RuntimeError("API response missing usage information.")
            usage_dump = (
                usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
            )
            choice = response.choices[0]
            return {
                "text": content,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "raw_model": response.model,
                "finish_reason": getattr(choice, "finish_reason", None),
                "usage_details": usage_dump,
            }
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"Model request failed after retries: {last_error}")


def request_completion_transformers(
    model: Any,
    tokenizer: Any,
    resolved_model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    model_inputs = tokenizer(prompt_text, return_tensors="pt")
    model_inputs = {
        name: tensor.to(model.device) for name, tensor in model_inputs.items()
    }
    prompt_tokens = int(model_inputs["input_ids"].shape[1])
    generated = model.generate(
        **model_inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_token_ids = generated[0][prompt_tokens:]
    completion_tokens = int(new_token_ids.shape[0])
    text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    finish_reason = "length"
    if completion_tokens > 0 and tokenizer.eos_token_id is not None:
        last_token = int(new_token_ids[-1])
        if last_token == tokenizer.eos_token_id:
            finish_reason = "stop"
    return {
        "text": text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "raw_model": resolved_model,
        "finish_reason": finish_reason,
        "usage_details": None,
    }


def output_directory(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Path("log") / f"pilot-{timestamp}"


def load_completed(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    completed = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            completed.add(
                (record["benchmark"], record["example_id"], record["condition"])
            )
    return completed


def save_manifest(tasks: list[TaskExample], path: Path) -> None:
    manifest = [
        {
            "benchmark": task.benchmark,
            "example_id": task.example_id,
            "prompt": task.prompt,
            "metadata": task.metadata,
            "max_tokens": task.max_tokens,
        }
        for task in tasks
    ]
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_manifest(path: Path) -> list[TaskExample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = []
    for item in payload:
        tasks.append(
            TaskExample(
                benchmark=item["benchmark"],
                example_id=item["example_id"],
                prompt=item["prompt"],
                metadata=item["metadata"],
                max_tokens=item["max_tokens"],
            )
        )
    return tasks


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["benchmark"], row["condition"])].append(row)

    summary: dict[str, Any] = {"per_group": {}, "overall": {}}

    for (benchmark, condition), items in sorted(grouped.items()):
        scores = [item["score"] for item in items]
        prompt_tokens = [item["prompt_tokens"] for item in items]
        completion_tokens = [item["completion_tokens"] for item in items]
        total_tokens = [item["total_tokens"] for item in items]
        summary["per_group"][f"{benchmark}:{condition}"] = {
            "benchmark": benchmark,
            "condition": condition,
            "n": len(items),
            "score_mean": statistics.mean(scores),
            "prompt_tokens_mean": statistics.mean(prompt_tokens),
            "completion_tokens_mean": statistics.mean(completion_tokens),
            "total_tokens_mean": statistics.mean(total_tokens),
            "score_per_1k_tokens": 1000.0 * sum(scores) / sum(total_tokens),
        }

    overall_grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        overall_grouped[row["condition"]].append(row)

    for condition, items in sorted(overall_grouped.items()):
        scores = [item["score"] for item in items]
        prompt_tokens = [item["prompt_tokens"] for item in items]
        completion_tokens = [item["completion_tokens"] for item in items]
        total_tokens = [item["total_tokens"] for item in items]
        summary["overall"][condition] = {
            "n": len(items),
            "score_mean": statistics.mean(scores),
            "prompt_tokens_mean": statistics.mean(prompt_tokens),
            "completion_tokens_mean": statistics.mean(completion_tokens),
            "total_tokens_mean": statistics.mean(total_tokens),
            "score_per_1k_tokens": 1000.0 * sum(scores) / sum(total_tokens),
        }

    base = summary["overall"].get("base")
    if base:
        for condition, metrics in summary["overall"].items():
            metrics["prompt_token_delta_vs_base"] = (
                metrics["prompt_tokens_mean"] - base["prompt_tokens_mean"]
            )
            metrics["total_token_delta_vs_base"] = (
                metrics["total_tokens_mean"] - base["total_tokens_mean"]
            )
            metrics["score_delta_vs_base"] = metrics["score_mean"] - base["score_mean"]

    return summary


def make_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Pilot Summary",
        "",
        "## Overall",
        "",
        "| Condition | N | Score | Prompt Tok | Completion Tok | Total Tok | Score/1k Tok | Prompt Delta vs Base | Score Delta vs Base |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition, metrics in summary["overall"].items():
        lines.append(
            "| {condition} | {n} | {score:.3f} | {prompt:.2f} | {completion:.2f} | {total:.2f} | {eff:.3f} | {pdelta:.2f} | {sdelta:.3f} |".format(
                condition=condition,
                n=metrics["n"],
                score=metrics["score_mean"],
                prompt=metrics["prompt_tokens_mean"],
                completion=metrics["completion_tokens_mean"],
                total=metrics["total_tokens_mean"],
                eff=metrics["score_per_1k_tokens"],
                pdelta=metrics.get("prompt_token_delta_vs_base", 0.0),
                sdelta=metrics.get("score_delta_vs_base", 0.0),
            )
        )

    lines.extend(
        [
            "",
            "## By Benchmark",
            "",
            "| Benchmark | Condition | N | Score | Prompt Tok | Completion Tok | Total Tok | Score/1k Tok |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, metrics in summary["per_group"].items():
        lines.append(
            "| {benchmark} | {condition} | {n} | {score:.3f} | {prompt:.2f} | {completion:.2f} | {total:.2f} | {eff:.3f} |".format(
                benchmark=metrics["benchmark"],
                condition=metrics["condition"],
                n=metrics["n"],
                score=metrics["score_mean"],
                prompt=metrics["prompt_tokens_mean"],
                completion=metrics["completion_tokens_mean"],
                total=metrics["total_tokens_mean"],
                eff=metrics["score_per_1k_tokens"],
            )
        )
    return "\n".join(lines) + "\n"


def list_models(client: OpenAI) -> None:
    models = client.models.list()
    for item in models.data:
        print(item.id)


def main() -> None:
    args = parse_args()
    for condition in args.conditions:
        if condition not in PROMPT_VARIANTS:
            raise ValueError(f"Unknown condition: {condition}")

    client = None
    local_model = None
    local_tokenizer = None
    if args.backend == "openai":
        client = build_client(args)
        if args.list_models:
            list_models(client)
            return
    elif args.list_models:
        raise ValueError("--list-models is only supported for the openai backend.")

    if args.backend == "transformers":
        local_model, local_tokenizer = build_transformers_backend(args)

    if args.manifest:
        tasks = load_manifest(Path(args.manifest))
    else:
        tasks = []
        tasks.extend(load_ifeval_tasks(args.ifeval_samples, args.seed))
        tasks.extend(load_mmlu_tasks(args.mmlu_samples, args.seed))
        if args.math500_samples > 0:
            tasks.extend(load_math500_tasks(args.math500_samples, args.seed))

    tasks = apply_max_tokens_override(tasks, args.max_tokens_override)

    out_dir = output_directory(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    results_path = out_dir / "results.jsonl"
    summary_json_path = out_dir / "summary.json"
    summary_md_path = out_dir / "summary.md"

    save_manifest(tasks, manifest_path)

    completed = load_completed(results_path) if args.resume else set()
    rows: list[dict[str, Any]] = []
    if args.resume and results_path.exists():
        with results_path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]

    with results_path.open("a", encoding="utf-8") as handle:
        for task in tqdm(tasks, desc="tasks"):
            for condition in args.conditions:
                key = (task.benchmark, task.example_id, condition)
                if key in completed:
                    continue
                if args.backend == "openai":
                    call = request_completion(
                        client=client,
                        model=args.model,
                        system_prompt=PROMPT_VARIANTS[condition],
                        user_prompt=task.prompt,
                        max_tokens=task.max_tokens,
                    )
                else:
                    call = request_completion_transformers(
                        model=local_model,
                        tokenizer=local_tokenizer,
                        resolved_model=args.local_model_path,
                        system_prompt=PROMPT_VARIANTS[condition],
                        user_prompt=task.prompt,
                        max_tokens=task.max_tokens,
                    )
                evaluation = evaluate_task(task, call["text"])
                row = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "benchmark": task.benchmark,
                    "example_id": task.example_id,
                    "condition": condition,
                    "model": args.model,
                    "backend": args.backend,
                    "resolved_model": call["raw_model"],
                    "configured_max_tokens": task.max_tokens,
                    "finish_reason": call.get("finish_reason"),
                    "score": evaluation["score"],
                    "prompt_tokens": call["prompt_tokens"],
                    "completion_tokens": call["completion_tokens"],
                    "total_tokens": call["total_tokens"],
                    "usage_details": call.get("usage_details"),
                    "response_text": evaluation["response_text"],
                    "evaluation": evaluation,
                    "task_metadata": task.metadata,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                rows.append(row)

    summary = aggregate_metrics(rows)
    summary_json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary_md_path.write_text(make_summary_markdown(summary), encoding="utf-8")

    print(summary_md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
