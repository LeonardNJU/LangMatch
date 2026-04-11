from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export consolidated per-sample matrix outputs."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec in the form model_label=path_to_run_dir",
    )
    parser.add_argument("--output", required=True, help="Output JSONL path")
    return parser.parse_args()


def parse_runs(run_specs: list[str]) -> list[tuple[str, Path]]:
    runs = []
    for spec in run_specs:
        if "=" not in spec:
            raise ValueError(f"Invalid run spec: {spec}")
        label, raw_path = spec.split("=", 1)
        runs.append((label, Path(raw_path)))
    return runs


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    runs = parse_runs(args.run)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exported_rows: list[dict[str, Any]] = []

    for model_label, run_dir in runs:
        manifest = load_json(run_dir / "manifest.json")
        manifest_lookup = {
            (item["benchmark"], item["example_id"]): item for item in manifest
        }
        results = [
            json.loads(line)
            for line in (run_dir / "results.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]

        for row in results:
            manifest_item = manifest_lookup[(row["benchmark"], row["example_id"])]
            exported_rows.append(
                {
                    "source_run": run_dir.name,
                    "model_label": model_label,
                    "model": row["model"],
                    "backend": row["backend"],
                    "resolved_model": row["resolved_model"],
                    "benchmark": row["benchmark"],
                    "example_id": row["example_id"],
                    "condition": row["condition"],
                    "system_prompt": PROMPT_VARIANTS[row["condition"]],
                    "user_prompt": manifest_item["prompt"],
                    "configured_max_tokens": row.get(
                        "configured_max_tokens", manifest_item["max_tokens"]
                    ),
                    "score": row["score"],
                    "prompt_tokens": row["prompt_tokens"],
                    "completion_tokens": row["completion_tokens"],
                    "total_tokens": row["total_tokens"],
                    "finish_reason": row.get("finish_reason"),
                    "response_text": row["response_text"],
                    "evaluation": row["evaluation"],
                    "task_metadata": row["task_metadata"],
                    "usage_details": row.get("usage_details"),
                    "timestamp": row["timestamp"],
                }
            )

    exported_rows.sort(
        key=lambda item: (
            item["model_label"],
            item["benchmark"],
            item["example_id"],
            item["condition"],
        )
    )

    output_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in exported_rows) + "\n",
        encoding="utf-8",
    )
    print(output_path)
    print(f"rows={len(exported_rows)}")


if __name__ == "__main__":
    main()
