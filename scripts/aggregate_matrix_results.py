from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate matrix evaluation outputs.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec in the form label=path_to_output_dir",
    )
    parser.add_argument("--output", required=True, help="Markdown report path")
    return parser.parse_args()


def load_summary(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_runs(run_specs: list[str]) -> list[tuple[str, Path]]:
    runs = []
    for spec in run_specs:
        if "=" not in spec:
            raise ValueError(f"Invalid run spec: {spec}")
        label, raw_path = spec.split("=", 1)
        runs.append((label, Path(raw_path)))
    return runs


def format_float(value: float) -> str:
    return f"{value:.3f}"


def main() -> None:
    args = parse_args()
    runs = parse_runs(args.run)

    summaries = [(label, load_summary(path)) for label, path in runs]
    benchmark_keys = sorted(
        {
            metrics["benchmark"]
            for _, summary in summaries
            for metrics in summary["per_group"].values()
        }
    )
    conditions = ["base", "zh_compact", "wy"]

    lines = ["# Matrix Evaluation Summary", ""]

    lines.extend(
        [
            "## Overall by Model and Prompt",
            "",
            "| Model | Prompt | N | Score | Prompt Tok | Completion Tok | Total Tok | Score/1k Tok |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, summary in summaries:
        overall = summary["overall"]
        for condition in conditions:
            metrics = overall[condition]
            lines.append(
                f"| {label} | {condition} | {metrics['n']} | {format_float(metrics['score_mean'])} | "
                f"{metrics['prompt_tokens_mean']:.2f} | {metrics['completion_tokens_mean']:.2f} | "
                f"{metrics['total_tokens_mean']:.2f} | {format_float(metrics['score_per_1k_tokens'])} |"
            )

    for benchmark in benchmark_keys:
        lines.extend(
            [
                "",
                f"## {benchmark}",
                "",
                "| Model | Prompt | N | Score | Prompt Tok | Completion Tok | Total Tok | Score/1k Tok |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for label, summary in summaries:
            per_group = summary["per_group"]
            for condition in conditions:
                key = f"{benchmark}:{condition}"
                if key not in per_group:
                    continue
                metrics = per_group[key]
                lines.append(
                    f"| {label} | {condition} | {metrics['n']} | {format_float(metrics['score_mean'])} | "
                    f"{metrics['prompt_tokens_mean']:.2f} | {metrics['completion_tokens_mean']:.2f} | "
                    f"{metrics['total_tokens_mean']:.2f} | {format_float(metrics['score_per_1k_tokens'])} |"
                )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
