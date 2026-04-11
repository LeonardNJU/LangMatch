from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from openai import OpenAI


TARGET_CONDITIONS = ["base", "zh", "wy"]
TARGET_MODES = ["hidden", "compact"]

PROMPT_CONTRACTS = {
    "hidden": {
        "system": {
            "base": (
                "You are a careful reasoning assistant. Think in English, answer in English, "
                "and always place the final answer inside <answer>...</answer>. For multiple-choice tasks, "
                "put only the option letter inside the answer tag."
            ),
            "zh": (
                "你是一个严谨的推理助手。请用中文思考、用中文作答，并始终把最终答案放在<answer>...</answer>中。"
                "如果是选择题，answer标签里只能放选项字母。"
            ),
            "wy": (
                "汝为谨严推理之助手。思与答皆用文言，终须以<answer>...</answer>标其终答。"
                "若为选择题，answer标签中惟书选项字母。"
            ),
        },
        "suffix": {
            "mmlu_pro": {
                "base": "Think carefully in English. You do not need to show your reasoning. Put only the final option letter inside <answer>...</answer>.",
                "zh": "请用中文仔细思考，不必展示推理过程。只把最终选项字母写在<answer>...</answer>中。",
                "wy": "請以文言審思，不必顯其推理。惟以最終選項字母置於<answer>...</answer>之中。",
            },
            "math500": {
                "base": "Think carefully in English. You do not need to show your reasoning. Put only the final answer inside <answer>...</answer>. Do not use \\boxed.",
                "zh": "请用中文仔细思考，不必展示推理过程。只将最终答案放在<answer>...</answer>中。不要使用\\boxed。",
                "wy": "請以文言審思，不必顯其推理。惟最終答案須且僅須置於<answer>...</answer>之內。勿用\\boxed。",
            },
        },
    },
    "compact": {
        "system": {
            "base": (
                "You are a careful reasoning assistant. Think in English, answer in English, keep the response compact, "
                "and always place the final answer inside <answer>...</answer>. For multiple-choice tasks, put only the option letter inside the answer tag."
            ),
            "zh": (
                "你是一个严谨的推理助手。请用中文思考、用中文作答、整体保持简洁，并始终把最终答案放在<answer>...</answer>中。"
                "如果是选择题，answer标签里只能放选项字母。"
            ),
            "wy": (
                "汝为谨严推理之助手。思与答皆用文言，整體務求簡勁，終須以<answer>...</answer>標其終答。"
                "若為選擇題，answer標中惟書選項字母。"
            ),
        },
        "suffix": {
            "mmlu_pro": {
                "base": "Think carefully in English. You may include a brief reasoning sketch if helpful, but keep it compact and no longer than one short paragraph. Then put only the final option letter inside <answer>...</answer>.",
                "zh": "请用中文仔细思考。如有必要，可以写一小段简短推理，但整体必须简洁，且不要超过一个短段。然后只把最终选项字母写在<answer>...</answer>中。",
                "wy": "請以文言審思。若有必要，得略陳其理，然務須簡勁，且毋過一短段。然後但以最終選項字母置於<answer>...</answer>之中。",
            },
            "math500": {
                "base": "Think carefully in English. You may include a brief reasoning sketch if helpful, but keep it compact and no longer than one short paragraph. Put only the final answer inside <answer>...</answer>. Do not use \\boxed.",
                "zh": "请用中文仔细思考。如有必要，可以写一小段简短推理，但整体必须简洁，且不要超过一个短段。只将最终答案放在<answer>...</answer>中。不要使用\\boxed。",
                "wy": "請以文言審思。若有必要，得略陳其理，然務須簡勁，且毋過一短段。惟最終答案須且僅須置於<answer>...</answer>之內。勿用\\boxed。",
            },
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare language-matched manifests.")
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=TARGET_MODES, required=True)
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--translation-model", default="gpt-4o")
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


def get_generation_token_kwargs(model: str, max_tokens: int) -> dict[str, int]:
    if model.startswith("gpt-5"):
        return {"max_completion_tokens": max_tokens}
    return {"max_tokens": max_tokens}


def strip_existing_answer_instruction(text: str) -> str:
    text = re.sub(
        r"\n\nRespond with the single correct option letter only\.\s*$", "", text
    )
    text = re.sub(
        r"\n\nSolve the problem\.\s*只输出最终答案，不要解释。Use compact ASCII math when possible: fractions as a/b, sqrt as sqrt\(\.\.\.\), tuples as \(a,b\), lists comma-separated, and do not use \\boxed\.\s*$",
        "",
        text,
    )
    return text.strip()


def get_prompt_contract(benchmark: str, condition: str, mode: str) -> tuple[str, str]:
    try:
        contract = PROMPT_CONTRACTS[mode]
        return contract["system"][condition], contract["suffix"][benchmark][condition]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported prompt contract: benchmark={benchmark}, condition={condition}, mode={mode}"
        ) from exc


def append_condition_instruction(
    core_prompt: str, benchmark: str, condition: str, mode: str
) -> str:
    _, suffix = get_prompt_contract(benchmark, condition, mode)
    return f"{core_prompt}\n\n{suffix}"


def translation_prompt(target_language: str, benchmark: str, text: str) -> str:
    benchmark_hint = {
        "mmlu_pro": "multiple-choice question with fixed option letters",
        "math500": "math problem with symbolic notation",
    }[benchmark]
    return (
        f"Translate the following task prompt into {target_language}.\n"
        "Requirements:\n"
        "1. Preserve task meaning exactly.\n"
        "2. Preserve all math notation, numbers, XML tags like <answer>...</answer>, and option letters A-J exactly.\n"
        "3. Do not solve the problem.\n"
        "4. The translated prompt must be natural and idiomatic in the target language.\n"
        f"5. This is a {benchmark_hint}.\n"
        "6. Return only the translated prompt text, with no commentary.\n\n"
        f"SOURCE PROMPT:\n{text}"
    )


def translate_text(
    client: OpenAI, model: str, target_language: str, benchmark: str, text: str
) -> str:
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                **get_generation_token_kwargs(model, 2048),
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise translator for benchmark prompts.",
                    },
                    {
                        "role": "user",
                        "content": translation_prompt(target_language, benchmark, text),
                    },
                ],
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(min(60, 2**attempt))
    raise RuntimeError(f"Translation failed after retries: {last_error}")


def build_condition_prompt(
    client: OpenAI, model: str, item: dict[str, Any], condition: str, mode: str
) -> str:
    source_core = strip_existing_answer_instruction(item["prompt"])
    if condition == "base":
        return append_condition_instruction(
            source_core, item["benchmark"], condition, mode
        )
    target_language = (
        "Modern Chinese" if condition == "zh" else "Classical Chinese (Wenyanwen)"
    )
    translated_core = translate_text(
        client, model, target_language, item["benchmark"], source_core
    )
    return append_condition_instruction(
        translated_core, item["benchmark"], condition, mode
    )


def main() -> None:
    args = parse_args()
    client = build_client(args)
    source_manifest = json.loads(Path(args.source_manifest).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"langmatch_{args.mode}_manifest.json"
    existing_lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    if out_path.exists():
        existing_rows = json.loads(out_path.read_text(encoding="utf-8"))
        existing_lookup = {
            (row["benchmark"], row["example_id"], row["condition"]): row
            for row in existing_rows
        }

    translated_rows = []
    for item in source_manifest:
        for condition in TARGET_CONDITIONS:
            key = (item["benchmark"], item["example_id"], condition)
            if key in existing_lookup:
                translated_rows.append(existing_lookup[key])
                continue
            user_prompt = build_condition_prompt(
                client, args.translation_model, item, condition, args.mode
            )
            system_prompt, _ = get_prompt_contract(
                benchmark=item["benchmark"], condition=condition, mode=args.mode
            )
            translated_rows.append(
                {
                    "benchmark": item["benchmark"],
                    "example_id": item["example_id"],
                    "mode": args.mode,
                    "condition": condition,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "source_prompt": item["prompt"],
                    "metadata": item["metadata"],
                    "max_tokens": 2048,
                }
            )
            out_path.write_text(
                json.dumps(translated_rows, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    out_path.write_text(
        json.dumps(translated_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(out_path)
    print(f"rows={len(translated_rows)}")


if __name__ == "__main__":
    main()
