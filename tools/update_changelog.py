#!/usr/bin/env python3
"""Update docs/changelog.md based on recent git commits."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
CHANGELOG = ROOT / "docs" / "changelog.md"

CATEGORY_MAP = [
    ("feat", "Added"),
    ("add", "Added"),
    ("fix", "Fixed"),
    ("bug", "Fixed"),
    ("docs", "Changed"),
    ("doc", "Changed"),
    ("refactor", "Changed"),
    ("chore", "Changed"),
    ("perf", "Changed"),
]


def fetch_commits(limit: int = 20) -> List[str]:
    try:
        output = subprocess.check_output(
            ["git", "log", f"-n{limit}", "--pretty=%s"],
            cwd=str(ROOT),
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError):
        print("Cannot read git log. Skipping changelog update.", file=sys.stderr)
        return []
    return [line.strip() for line in output.decode("utf-8").splitlines() if line.strip()]


def categorise(message: str) -> str:
    lower = message.lower()
    for prefix, category in CATEGORY_MAP:
        if lower.startswith(f"{prefix}:") or lower.startswith(f"{prefix}("):
            return category
    return "Changed"


def normalise_message(message: str) -> str:
    # Drop leading conventional-commit prefix if present.
    if ":" in message.split(" ")[0]:
        head, tail = message.split(":", 1)
        if any(head.lower().startswith(prefix) for prefix, _ in CATEGORY_MAP):
            return tail.strip().capitalize()
    return message.strip().capitalize()


def build_unreleased_section(messages: List[str]) -> List[str]:
    grouped: Dict[str, List[str]] = {"Added": [], "Changed": [], "Fixed": []}
    for msg in messages:
        category = categorise(msg)
        if category not in grouped:
            grouped[category] = []
        cleaned = normalise_message(msg)
        if cleaned not in grouped[category]:
            grouped[category].append(cleaned)

    lines: List[str] = []
    for category in ("Added", "Changed", "Fixed"):
        lines.append(f"### {category}")
        if grouped.get(category):
            lines.extend(f"- {entry}" for entry in grouped[category])
        else:
            lines.append("- (no recent entries)")
        lines.append("")
    return lines


def replace_unreleased(contents: str, new_block: List[str]) -> str:
    lines = contents.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip().lower() == "## [unreleased]")
    except StopIteration:
        # If the marker is missing, append it at the end.
        if lines and lines[-1] != "":
            lines.append("")
        lines.append("## [Unreleased]")
        start = len(lines) - 1

    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## [")), len(lines))
    new_lines = lines[: start + 1]
    new_lines.append("")
    new_lines.extend(new_block)
    new_lines.extend(lines[end:])
    return "\n".join(new_lines).rstrip() + "\n"


def ensure_changelog_exists() -> None:
    if CHANGELOG.exists():
        return
    CHANGELOG.parent.mkdir(parents=True, exist_ok=True)
    CHANGELOG.write_text(
        "# Changelog\n\nAll notable changes to this project will be documented in this file.\n\n"
        "## [Unreleased]\n\n### Added\n- (pending)\n\n### Changed\n- (pending)\n\n"
        "### Fixed\n- (pending)\n",
        encoding="utf-8",
    )


def main() -> None:
    ensure_changelog_exists()
    commits = fetch_commits()
    if not commits:
        return
    current = CHANGELOG.read_text(encoding="utf-8")
    new_section = build_unreleased_section(commits)
    updated = replace_unreleased(current, new_section)
    CHANGELOG.write_text(updated, encoding="utf-8")
    print("Updated docs/changelog.md")


if __name__ == "__main__":
    main()
