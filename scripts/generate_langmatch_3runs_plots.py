from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


WORKTREE_ROOT = Path(__file__).resolve().parents[1]
INTERACTIONS = WORKTREE_ROOT / "docs" / "langmatch_3runs_interaction_logs.jsonl"
SUMMARY_JSON = WORKTREE_ROOT / "docs" / "langmatch_3runs_plot_summary.json"
FIGURE_ROOT = WORKTREE_ROOT / "docs" / "figures" / "langmatch_3runs"
PUBLIC_SOURCE_LABEL = "publish-safe aggregated LangMatch 3-run summary"

SETTING_GROUPS = ["explicit_process", "hidden", "compact_visible"]
REQUESTED_METRICS = ["overall", "ifeval", "math500", "mmlu_pro"]
CONDITIONS = ["base", "zh", "wy"]
CONDITION_LABELS = {"base": "base", "zh": "zh", "wy": "wy"}
CONDITION_MARKERS = {"base": "o", "zh": "s", "wy": "^"}
MODEL_COLORS = {
    "gpt-4o": "#4C78A8",
    "gpt-5.4": "#F58518",
    "qwen3-4b": "#B279A2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactions", default=str(INTERACTIONS))
    parser.add_argument("--summary-json", default=str(SUMMARY_JSON))
    parser.add_argument("--figure-root", default=str(FIGURE_ROOT))
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def aggregate(rows: list[dict[str, object]]) -> dict[str, float | int]:
    scores = [float(row["score"]) for row in rows]
    prompt_tokens = [float(row["prompt_tokens"]) for row in rows]
    completion_tokens = [float(row["completion_tokens"]) for row in rows]
    total_tokens = [float(row["total_tokens"]) for row in rows]
    total_tok_mean = mean(total_tokens)
    sr = mean(scores)
    return {
        "n": len(rows),
        "sr": sr,
        "prompt_tok": mean(prompt_tokens),
        "completion_tok": mean(completion_tokens),
        "total_tok": total_tok_mean,
        "score_per_1k_tok": sr / (total_tok_mean / 1000.0) if total_tok_mean else 0.0,
    }


def build_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {
        "source": PUBLIC_SOURCE_LABEL,
        "setting_groups": SETTING_GROUPS,
        "requested_metrics": REQUESTED_METRICS,
        "condition_order": CONDITIONS,
        "model_colors": MODEL_COLORS,
        "by_setting_group": {},
    }

    available_benchmarks = sorted({str(row["benchmark"]) for row in rows})
    summary["available_benchmarks"] = available_benchmarks
    summary["missing_benchmarks"] = [
        metric
        for metric in REQUESTED_METRICS
        if metric != "overall" and metric not in available_benchmarks
    ]

    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(
        list
    )
    for row in rows:
        grouped[
            (
                str(row["setting_group"]),
                str(row["benchmark"]),
                str(row["model"]),
                str(row["condition"]),
            )
        ].append(row)

    for setting_group in SETTING_GROUPS:
        setting_payload: dict[str, object] = {}
        setting_rows = [row for row in rows if row["setting_group"] == setting_group]

        for metric in REQUESTED_METRICS:
            metric_payload: dict[str, dict[str, object]] = {}
            for model in sorted({str(row["model"]) for row in setting_rows}):
                model_payload: dict[str, object] = {}
                for condition in CONDITIONS:
                    if metric == "overall":
                        selected = [
                            row
                            for row in setting_rows
                            if row["model"] == model and row["condition"] == condition
                        ]
                    else:
                        selected = grouped.get(
                            (setting_group, metric, model, condition), []
                        )
                    if selected:
                        model_payload[condition] = aggregate(selected)
                if model_payload:
                    metric_payload[model] = model_payload
            setting_payload[metric] = metric_payload
        summary["by_setting_group"][setting_group] = setting_payload

    summary["axes"] = build_axes(summary)
    return summary


def build_axes(summary: dict[str, object]) -> dict[str, dict[str, list[float]]]:
    axes: dict[str, dict[str, list[float]]] = {}
    by_setting_group = summary["by_setting_group"]
    for metric in REQUESTED_METRICS:
        xs: list[float] = []
        ys: list[float] = []
        for setting_group in SETTING_GROUPS:
            metric_payload = by_setting_group[setting_group][metric]
            for model_payload in metric_payload.values():
                for stats in model_payload.values():
                    xs.append(float(stats["total_tok"]))
                    ys.append(float(stats["sr"]))
        if xs:
            x_min = min(xs)
            x_max = max(xs)
            pad = max(20.0, (x_max - x_min) * 0.08)
            axes[metric] = {
                "x": [max(0.0, x_min - pad), x_max + pad],
                "y": [0.0, min(1.05, max(1.0, max(ys) + 0.04))],
            }
        else:
            axes[metric] = {"x": [0.0, 1.0], "y": [0.0, 1.05]}
    return axes


def render_plot(
    setting_group: str,
    metric: str,
    metric_payload: dict[str, dict[str, object]],
    axis_limits: dict[str, list[float]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 6.4), layout="constrained")

    model_legend_handles = [
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
        for model in MODEL_COLORS
    ]
    prompt_legend_handles = [
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
        for condition in CONDITIONS
    ]

    if metric_payload:
        for model, conditions in metric_payload.items():
            points = []
            for condition in CONDITIONS:
                stats = conditions.get(condition)
                if not stats:
                    continue
                points.append(
                    (float(stats["total_tok"]), float(stats["sr"]), condition)
                )

            if not points:
                continue

            line_points = sorted(points, key=lambda item: item[0])
            ax.plot(
                [item[0] for item in line_points],
                [item[1] for item in line_points],
                color=MODEL_COLORS[model],
                linewidth=1.4,
                alpha=0.35,
                zorder=1,
            )

            for x, y, condition in points:
                ax.scatter(
                    x,
                    y,
                    color=MODEL_COLORS[model],
                    marker=CONDITION_MARKERS[condition],
                    s=110,
                    edgecolor="black",
                    linewidth=0.7,
                    alpha=0.95,
                    zorder=3,
                )
    else:
        ax.text(
            0.5,
            0.54,
            "No exported data for this metric",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            0.46,
            PUBLIC_SOURCE_LABEL,
            ha="center",
            va="center",
            fontsize=10,
            alpha=0.75,
            transform=ax.transAxes,
        )

    ax.set_xlim(axis_limits["x"])
    ax.set_ylim(axis_limits["y"])
    ax.set_xlabel("Total Tok (↓ better)")
    ax.set_ylabel("Score / SR (↑ better)")
    ax.set_title(f"{setting_group} / {metric}: Pareto-style score vs. token tradeoff")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    ax.set_axisbelow(True)

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
        title="Conditions",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.52),
        frameon=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    interactions_path = Path(args.interactions)
    summary_path = Path(args.summary_json)
    figure_root = Path(args.figure_root)

    if interactions_path.exists():
        rows = load_rows(interactions_path)
        summary = build_summary(rows)
        summary["source"] = PUBLIC_SOURCE_LABEL
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    elif summary_path.exists():
        summary = load_summary(summary_path)
    else:
        raise FileNotFoundError(
            "Need either local aggregate inputs or a prebuilt public plot summary JSON."
        )

    for setting_group in SETTING_GROUPS:
        for metric in REQUESTED_METRICS:
            render_plot(
                setting_group=setting_group,
                metric=metric,
                metric_payload=summary["by_setting_group"][setting_group][metric],
                axis_limits=summary["axes"][metric],
                output_path=figure_root / setting_group / f"{metric}.png",
            )

    print(
        json.dumps(
            {
                "summary_json": str(summary_path),
                "figure_root": str(figure_root),
                "missing_benchmarks": summary["missing_benchmarks"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
