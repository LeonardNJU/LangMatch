from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run language-matched thinking experiment."
    )
    parser.add_argument("--backend", choices=["openai", "transformers"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--local-model-path", default=None)
    parser.add_argument("--resume", action="store_true")
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
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not args.local_model_path:
        raise ValueError("Missing local model path for transformers backend.")
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


def load_manifest(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_math_answer(text: str) -> str:
    normalized = text.strip()
    normalized = re.sub(r"\\boxed\{(.+?)\}", r"\1", normalized)
    normalized = normalized.replace("$", "")
    normalized = normalized.replace("\\left", "")
    normalized = normalized.replace("\\right", "")
    normalized = normalized.replace("\\pi", "pi")
    normalized = re.sub(
        r"\\(?:dfrac|tfrac|frac)\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", normalized
    )
    normalized = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", normalized)
    normalized = re.sub(r"\\text\{([^{}]+)\}", r"\1", normalized)
    normalized = normalized.replace("{", "").replace("}", "")
    normalized = re.sub(r"\s+", "", normalized)
    normalized = normalized.rstrip(".")
    normalized = normalized.replace("°", "")
    return normalized.lower()


def extract_answer_tag(text: str) -> str | None:
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def split_think_output(text: str) -> tuple[str, str]:
    match = re.search(r"<think>(.*?)</think>", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return "", text.strip()
    hidden = match.group(1).strip()
    visible = (text[: match.start()] + text[match.end() :]).strip()
    return hidden, visible


def evaluate_row(row_manifest: dict[str, Any], response_text: str) -> dict[str, Any]:
    answer = extract_answer_tag(response_text)
    benchmark = row_manifest["benchmark"]
    if benchmark == "mmlu_pro":
        predicted = None
        if answer:
            match = re.search(r"\b([A-J])\b", answer.upper())
            predicted = match.group(1) if match else None
        gold = row_manifest["metadata"]["answer"]
        return {
            "score": int(predicted == gold),
            "predicted": predicted,
            "gold": gold,
            "answer_slot": answer,
        }
    if benchmark == "math500":
        gold = row_manifest["metadata"]["answer"]
        predicted_norm = normalize_math_answer(answer or "")
        gold_norm = normalize_math_answer(gold)
        return {
            "score": int(predicted_norm == gold_norm),
            "predicted": answer,
            "gold": gold,
            "predicted_normalized": predicted_norm,
            "gold_normalized": gold_norm,
            "answer_slot": answer,
        }
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def request_openai(
    client: OpenAI, model: str, system_prompt: str, user_prompt: str, max_tokens: int
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            if model.startswith("gpt-5"):
                response = client.chat.completions.create(
                    model=model,
                    temperature=0.0,
                    reasoning_effort="medium",
                    max_completion_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    temperature=0.0,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
            usage = response.usage
            choice = response.choices[0]
            return {
                "text": choice.message.content or "",
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "raw_model": response.model,
                "finish_reason": getattr(choice, "finish_reason", None),
                "usage_details": usage.model_dump()
                if hasattr(usage, "model_dump")
                else dict(usage),
            }
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"OpenAI request failed after retries: {last_error}")


def request_transformers(
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
        enable_thinking=True,
    )
    model_inputs = tokenizer(prompt_text, return_tensors="pt")
    model_inputs = {
        name: tensor.to(model.device) for name, tensor in model_inputs.items()
    }
    prompt_tokens = int(model_inputs["input_ids"].shape[1])
    generated = model.generate(
        **model_inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_token_ids = generated[0][prompt_tokens:]
    completion_tokens = int(new_token_ids.shape[0])
    raw_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    hidden_reasoning_text, visible_text = split_think_output(raw_text)
    finish_reason = "length"
    if (
        completion_tokens > 0
        and tokenizer.eos_token_id is not None
        and int(new_token_ids[-1]) == tokenizer.eos_token_id
    ):
        finish_reason = "stop"
    return {
        "text": visible_text,
        "raw_text": raw_text,
        "hidden_reasoning_text": hidden_reasoning_text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "raw_model": resolved_model,
        "finish_reason": finish_reason,
        "usage_details": None,
    }


def make_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["benchmark"], row["condition"])].append(row)
    summary = {"per_group": {}, "overall": {}}
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
        }
    overall = defaultdict(list)
    for row in rows:
        overall[row["condition"]].append(row)
    for condition, items in sorted(overall.items()):
        scores = [item["score"] for item in items]
        summary["overall"][condition] = {
            "n": len(items),
            "score_mean": statistics.mean(scores),
            "prompt_tokens_mean": statistics.mean(
                [item["prompt_tokens"] for item in items]
            ),
            "completion_tokens_mean": statistics.mean(
                [item["completion_tokens"] for item in items]
            ),
            "total_tokens_mean": statistics.mean(
                [item["total_tokens"] for item in items]
            ),
        }
    return summary


def main() -> None:
    args = parse_args()
    manifest = load_manifest(Path(args.manifest))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    results_path = out_dir / "results.jsonl"

    completed = set()
    rows: list[dict[str, Any]] = []
    if args.resume and results_path.exists():
        with results_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows.append(row)
                completed.add((row["benchmark"], row["example_id"], row["condition"]))

    client = build_client(args) if args.backend == "openai" else None
    local_model, local_tokenizer = (None, None)
    if args.backend == "transformers":
        local_model, local_tokenizer = build_transformers_backend(args)

    with results_path.open("a", encoding="utf-8") as handle:
        for item in manifest:
            key = (item["benchmark"], item["example_id"], item["condition"])
            if key in completed:
                continue
            if args.backend == "openai":
                call = request_openai(
                    client,
                    args.model,
                    item["system_prompt"],
                    item["user_prompt"],
                    item["max_tokens"],
                )
            else:
                call = request_transformers(
                    local_model,
                    local_tokenizer,
                    args.local_model_path,
                    item["system_prompt"],
                    item["user_prompt"],
                    item["max_tokens"],
                )
            evaluation = evaluate_row(item, call["text"])
            row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "benchmark": item["benchmark"],
                "example_id": item["example_id"],
                "condition": item["condition"],
                "model": args.model,
                "backend": args.backend,
                "resolved_model": call["raw_model"],
                "configured_max_tokens": item["max_tokens"],
                "system_prompt": item["system_prompt"],
                "user_prompt": item["user_prompt"],
                "source_prompt": item["source_prompt"],
                "mode": item.get("mode"),
                "score": evaluation["score"],
                "prompt_tokens": call["prompt_tokens"],
                "completion_tokens": call["completion_tokens"],
                "total_tokens": call["total_tokens"],
                "finish_reason": call.get("finish_reason"),
                "usage_details": call.get("usage_details"),
                "response_text": call["text"],
                "raw_response_text": call.get("raw_text", call["text"]),
                "hidden_reasoning_text": call.get("hidden_reasoning_text", ""),
                "evaluation": evaluation,
                "task_metadata": item["metadata"],
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            rows.append(row)

    summary = make_summary(rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
