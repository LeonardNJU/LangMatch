from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate language-matched rerun outputs."
    )
    parser.add_argument("--run", action="append", required=True, help="label=run_dir")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def parse_runs(run_specs: list[str]) -> list[tuple[str, Path]]:
    runs = []
    for spec in run_specs:
        label, raw_path = spec.split("=", 1)
        runs.append((label, Path(raw_path)))
    return runs


def load_summary(path: Path) -> dict:
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    runs = parse_runs(args.run)
    summaries = [(label, load_summary(path)) for label, path in runs]
    benchmarks = sorted(
        {v["benchmark"] for _, s in summaries for v in s["per_group"].values()}
    )
    conditions = ["base", "zh", "wy"]
    lines = ["# Language-matched thinking rerun summary", ""]
    lines.extend(
        [
            "## Overall by model and condition",
            "",
            "| Model | Condition | N | Score | Prompt Tok | Completion Tok | Total Tok |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, summary in summaries:
        for cond in conditions:
            m = summary["overall"][cond]
            lines.append(
                f"| {label} | {cond} | {m['n']} | {m['score_mean']:.3f} | {m['prompt_tokens_mean']:.2f} | {m['completion_tokens_mean']:.2f} | {m['total_tokens_mean']:.2f} |"
            )
    for bench in benchmarks:
        lines.extend(
            [
                "",
                f"## {bench}",
                "",
                "| Model | Condition | N | Score | Prompt Tok | Completion Tok | Total Tok |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for label, summary in summaries:
            for cond in conditions:
                m = summary["per_group"][f"{bench}:{cond}"]
                lines.append(
                    f"| {label} | {cond} | {m['n']} | {m['score_mean']:.3f} | {m['prompt_tokens_mean']:.2f} | {m['completion_tokens_mean']:.2f} | {m['total_tokens_mean']:.2f} |"
                )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
