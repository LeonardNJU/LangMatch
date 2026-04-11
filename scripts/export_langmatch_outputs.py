from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export consolidated language-matched outputs."
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


def main() -> None:
    args = parse_args()
    runs = parse_runs(args.run)
    rows: list[dict[str, Any]] = []
    for label, run_dir in runs:
        results = [
            json.loads(line)
            for line in (run_dir / "results.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        for row in results:
            row = dict(row)
            row["model_label"] = label
            row["source_run"] = run_dir.name
            rows.append(row)
    rows.sort(
        key=lambda r: (
            r["model_label"],
            r["benchmark"],
            r["example_id"],
            r["condition"],
        )
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    print(output)
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()
