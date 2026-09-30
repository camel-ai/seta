from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    labels = [row.get("label") for row in rows]
    if not rows or any(not isinstance(label, str) or not label for label in labels):
        raise ValueError(f"every row in {path} must have a non-empty string label")
    if any(Path(label).name != label or label in (".", "..") for label in labels):
        raise ValueError(f"task labels in {path} must be safe directory names")
    if len(labels) != len(set(labels)):
        raise ValueError(f"duplicate labels in {path}")
    return rows


def select_rows(
    rows: list[dict[str, Any]],
    include: list[str],
    match: str | None,
    exclude: list[str],
) -> list[dict[str, Any]]:
    include_set = set(include)
    exclude_set = set(exclude)
    pattern = re.compile(match) if match else None
    selected = [
        row
        for row in rows
        if (not include_set or row["label"] in include_set)
        and (pattern is None or pattern.search(row["label"]))
        and row["label"] not in exclude_set
    ]
    missing = include_set - {row["label"] for row in rows}
    if missing:
        raise ValueError(f"unknown included tasks: {', '.join(sorted(missing))}")
    if not selected:
        raise ValueError("task filter selected no rows")
    return sorted(selected, key=lambda row: row["label"])


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"wrote {len(rows)} prompts to {path}")


def build_rows(tasks_dir: Path, agent_name: str) -> list[dict[str, Any]]:
    rows = []
    for task_dir in sorted(
        path
        for path in tasks_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ):
        instruction = task_dir / "instruction.md"
        if not instruction.is_file():
            raise FileNotFoundError(instruction)
        rows.append(
            {
                "label": task_dir.name,
                "prompt": [{"role": "user", "content": instruction.read_text()}],
                "metadata": {"agent_name": agent_name, "instance_id": task_dir.name},
            }
        )
    return rows


def add_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--include", action="append", default=[], metavar="TASK")
    parser.add_argument("--match", help="regular expression matched against task labels")
    parser.add_argument("--exclude", action="append", default=[], metavar="TASK")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--tasks-dir", type=Path, required=True)
    build.add_argument("--agent-name", default="terminus-2")
    build.add_argument("--output", type=Path, required=True)
    add_filters(build)
    filtered = commands.add_parser("filter")
    filtered.add_argument("--input", type=Path, required=True)
    filtered.add_argument("--output", type=Path, required=True)
    add_filters(filtered)
    names = commands.add_parser("names")
    names.add_argument("--input", type=Path, required=True)
    names.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "names":
        rows = sorted(read_jsonl(args.input), key=lambda row: row["label"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("".join(f'{row["label"]}\n' for row in rows))
        print(f"wrote {len(rows)} task names to {args.output}")
        return
    rows = build_rows(args.tasks_dir, args.agent_name) if args.command == "build" else read_jsonl(args.input)
    write_jsonl(args.output, select_rows(rows, args.include, args.match, args.exclude))


if __name__ == "__main__":
    main()
