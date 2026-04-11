from __future__ import annotations

import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path


WORKTREE_ROOT = Path(__file__).resolve().parents[1]


def get_repo_root() -> Path:
    common_dir = subprocess.check_output(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=WORKTREE_ROOT,
        text=True,
    ).strip()
    return (WORKTREE_ROOT / common_dir).resolve().parent


REPO_ROOT = get_repo_root()
LOG_SOURCES = [
    ("explicit_process", "log/langmatch-qwen3-4b-v1/results.jsonl"),
    ("hidden", "log/langmatch-hidden-qwen3-4b-v2/results.jsonl"),
    (
        "compact_visible",
        "log/langmatch-compact-qwen3-4b-v2/results.jsonl",
    ),
    ("explicit_process", "log/langmatch-gpt4o-v1/results.jsonl"),
    ("explicit_process", "log/langmatch-gpt54-v1/results.jsonl"),
    ("hidden", "log/langmatch-hidden-gpt4o-v2/results.jsonl"),
    ("hidden", "log/langmatch-hidden-gpt54-v2/results.jsonl"),
    (
        "compact_visible",
        "log/langmatch-compact-gpt4o-v2/results.jsonl",
    ),
    (
        "compact_visible",
        "log/langmatch-compact-gpt54-v2/results.jsonl",
    ),
]

OUTPUT_DIR = WORKTREE_ROOT / "docs"
SUMMARY_MD = OUTPUT_DIR / "langmatch_3runs_tok_sr_table.md"
SUMMARY_CSV = OUTPUT_DIR / "langmatch_3runs_tok_sr_table.csv"
INTERACTIONS = OUTPUT_DIR / "langmatch_3runs_interaction_logs.jsonl"
GAPS = OUTPUT_DIR / "langmatch_3runs_data_gaps.md"


def norm_condition(condition: str) -> str:
    if condition == "zh_compact":
        return "zh"
    if condition not in {"base", "zh", "wy"}:
        raise ValueError(f"unexpected condition: {condition}")
    return condition


def load_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def resolve_source(relpath: str) -> Path:
    for base in (WORKTREE_ROOT, REPO_ROOT):
        candidate = base / relpath
        if candidate.exists():
            return candidate
    raise FileNotFoundError(relpath)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def source_base_label(path: Path) -> str:
    try:
        path.relative_to(WORKTREE_ROOT)
        return "worktree"
    except ValueError:
        return "repo_root"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    used_sources: list[tuple[str, str, Path]] = []

    for setting_group, relpath in LOG_SOURCES:
        path = resolve_source(relpath)
        used_sources.append((setting_group, relpath, path))
        for row in load_rows(path):
            condition = norm_condition(str(row["condition"]))
            model = str(row["model"])
            key = (setting_group, model, condition)
            enriched = dict(row)
            enriched["setting_group"] = setting_group
            enriched["condition"] = condition
            raw_response_field = (
                "raw_response_text"
                if row.get("raw_response_text") is not None
                else "response_text"
            )
            enriched["raw_response_field"] = raw_response_field
            enriched["raw_response"] = row.get(raw_response_field)
            enriched["source_relpath"] = relpath
            enriched["source_location"] = source_base_label(path)
            all_rows.append(enriched)
            grouped[key].append(enriched)

    csv_rows = []
    for (setting_group, model, condition), rows in sorted(grouped.items()):
        csv_rows.append(
            {
                "setting_group": setting_group,
                "model": model,
                "condition": condition,
                "n": len(rows),
                "sr": mean([float(r["score"]) for r in rows]),
                "prompt_tok": mean([float(r["prompt_tokens"]) for r in rows]),
                "completion_tok": mean([float(r["completion_tokens"]) for r in rows]),
                "total_tok": mean([float(r["total_tokens"]) for r in rows]),
            }
        )

    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "setting_group",
                "model",
                "condition",
                "n",
                "sr",
                "prompt_tok",
                "completion_tok",
                "total_tok",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    sections = []
    for setting_group in ["explicit_process", "hidden", "compact_visible"]:
        sections.append(f"## {setting_group}")
        sections.append("")
        sections.append(
            "| Model | Setting | N | SR | Prompt Tok | Completion Tok | Total Tok |"
        )
        sections.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for row in csv_rows:
            if row["setting_group"] == setting_group:
                sections.append(
                    f"| {row['model']} | {row['condition']} | {row['n']} | {row['sr']:.3f} | {row['prompt_tok']:.2f} | {row['completion_tok']:.2f} | {row['total_tok']:.2f} |"
                )
        sections.append("")
    SUMMARY_MD.write_text(
        "# LangMatch 3-run Summary\n\n"
        "Token columns come from source-native backends (`openai` for GPT logs, `transformers` for qwen logs); they are reported as recorded and should not be interpreted as tokenizer-identical across providers. Full source provenance is listed in `langmatch_3runs_data_gaps.md`.\n\n"
        + "\n".join(sections).rstrip()
        + "\n",
        encoding="utf-8",
    )

    with INTERACTIONS.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(
                json.dumps(
                    {
                        k: row.get(k)
                        for k in [
                            "setting_group",
                            "model",
                            "condition",
                            "benchmark",
                            "example_id",
                            "backend",
                            "system_prompt",
                            "user_prompt",
                            "raw_response",
                            "raw_response_field",
                            "response_text",
                            "hidden_reasoning_text",
                            "source_relpath",
                            "source_location",
                            "score",
                            "prompt_tokens",
                            "completion_tokens",
                            "total_tokens",
                        ]
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    source_lines = []
    for setting_group, relpath, path in used_sources:
        source_lines.append(
            f"- {setting_group}: `{relpath}` ({source_base_label(path)})"
        )

    gaps = [
        "# LangMatch 3-run Data Gaps",
        "",
        "For the currently exported 3-run artifacts, coverage is complete for all 3 models (gpt-4o, gpt-5.4, qwen3-4b) across all 3 settings (base/zh/wy) in `overall`, `math500`, and `mmlu_pro`.",
        "`ifeval` is currently absent from the exported 3-run interaction logs, so it is intentionally treated as missing rather than complete.",
        "",
        "Notes:",
        "- qwen3-4b hidden/compact logs were synced from prior remote HPC runs before export.",
        "- qwen3-4b explicit_process logs were likewise synced from the remote project copy before export.",
        "- Token counts come from mixed backends (`openai` vs `transformers`), so cross-provider token comparisons are source-native rather than tokenizer-identical.",
        "- `raw_response` preserves the source-native raw field when present (`raw_response_text`), otherwise falls back to `response_text`; hidden-trace availability therefore differs by backend/provider.",
        "",
        "Source provenance:",
        *source_lines,
    ]
    GAPS.write_text("\n".join(gaps) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {"csv_rows": len(csv_rows), "interaction_rows": len(all_rows)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
