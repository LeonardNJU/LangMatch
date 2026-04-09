from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CONDITIONS = ["base", "zh_compact", "wy"]
CONDITION_LABELS = {"base": "base", "zh_compact": "zh", "wy": "wy"}
CONDITION_MARKERS = {
    "base": "o",
    "zh_compact": "s",
    "wy": "^",
}
BENCHMARK_ORDER = ["ifeval", "math500", "mmlu_pro"]
MODEL_COLORS = {
    "gpt-4o": "#4C78A8",
    "gpt-5.4": "#F58518",
    "qwen3-1.7b": "#54A24B",
    "qwen3-4b": "#B279A2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def parse_runs(run_specs: list[str]) -> list[tuple[str, Path]]:
    runs = []
    for spec in run_specs:
        label, raw_path = spec.split("=", 1)
        runs.append((label, Path(raw_path)))
    return runs


def load_summary(output_dir: Path) -> dict[str, Any]:
    return json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))


def collect_benchmark_data(summaries, benchmark):
    benchmark_data = {}
    for model_label, summary in summaries:
        model_data = {}
        for condition in CONDITIONS:
            key = f"{benchmark}:{condition}"
            model_data[condition] = summary["per_group"][key]
        benchmark_data[model_label] = model_data
    return benchmark_data


def render_benchmark_plot(benchmark: str, benchmark_data: dict, output_path: Path):
    models = list(benchmark_data.keys())

    fig, ax = plt.subplots(figsize=(9.2, 6.4), layout="constrained")

    model_legend_handles = []
    prompt_legend_handles = []

    for model in models:
        model_legend_handles.append(
            Line2D(
                [0],
                [0],
                color=MODEL_COLORS[model],
                marker="o",
                linestyle="-",
                linewidth=1.4,
                markersize=8,
                label=model,
            )
        )

    for condition in CONDITIONS:
        prompt_legend_handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                marker=CONDITION_MARKERS[condition],
                linestyle="None",
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=1.0,
                markersize=8,
                label=CONDITION_LABELS[condition],
            )
        )

    for model in models:
        color = MODEL_COLORS[model]
        points = []
        for condition in CONDITIONS:
            x = benchmark_data[model][condition]["total_tokens_mean"]
            y = benchmark_data[model][condition]["score_mean"]
            points.append((x, y, condition))

        points.sort(key=lambda item: item[0])
        xs = [item[0] for item in points]
        ys = [item[1] for item in points]
        ax.plot(xs, ys, color=color, linewidth=1.4, alpha=0.35, zorder=1)

        for x, y, condition in points:
            ax.scatter(
                x,
                y,
                color=color,
                marker=CONDITION_MARKERS[condition],
                s=110,
                edgecolor="black",
                linewidth=0.7,
                alpha=0.95,
                zorder=3,
            )

    ax.set_xlabel("Total Tok (↓ better)")
    ax.set_ylabel("Score / SR (↑ better)")
    ax.set_title(f"{benchmark}: Pareto-style score vs. total token tradeoff")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    ax.set_axisbelow(True)

    y_max = max(1.0, ax.get_ylim()[1])
    ax.set_ylim(0, min(1.05, y_max + 0.03))

    leg1 = ax.legend(
        handles=model_legend_handles,
        title="Models",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        frameon=True,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=prompt_legend_handles,
        title="Prompt modes",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.52),
        frameon=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    runs = parse_runs(args.run)
    summaries = [(label, load_summary(path)) for label, path in runs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for benchmark in BENCHMARK_ORDER:
        data = collect_benchmark_data(summaries, benchmark)
        render_benchmark_plot(benchmark, data, output_dir / f"{benchmark}.png")
        print(output_dir / f"{benchmark}.png")


if __name__ == "__main__":
    main()
