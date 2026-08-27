#!/usr/bin/env python3
"""Fail on comments that are misaligned or wider than the configured limit.

The check covers Python and fenced code in Markdown. Tool directives and
unbreakable tokens such as URLs are exempt because wrapping them would not be
useful. With no arguments, the script scans the repository's maintained source
and documentation paths. It is stdlib-only for use from pre-commit.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = ("dreamerv3", "embodied", "scores", "scripts")
TOP_LEVEL_FILES = ("README.md", "plot.py", "setup.py")
DEFAULT_LIMIT = 88
FENCE = "```"

# Directives bind to their source line, so moving them would change meaning.
DIRECTIVE = re.compile(r"^#\s*(type|noqa|pyright|mypy|pragma|fmt|isort|ruff|nosec)\b")


def comment_limit() -> int:
    """Return the configured Ruff or Black line limit, falling back to 88."""
    try:
        text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    except OSError:
        return DEFAULT_LIMIT
    for tool in ("ruff", "black"):
        section = re.search(rf"\[tool\.{tool}\](.*?)(?:\n\[|\Z)", text, re.S)
        if section is not None:
            match = re.search(r"^line-length\s*=\s*(\d+)", section.group(1), re.M)
            if match:
                return int(match.group(1))
    return DEFAULT_LIMIT


def _comment_column(line: str) -> int | None:
    """Return the comment column, ignoring hash characters inside strings."""
    quote: str | None = None
    index = 0
    while index < len(line):
        char = line[index]
        if quote is not None:
            if char == "\\":
                index += 2
                continue
            if char == quote:
                quote = None
        elif char in ("'", '"'):
            quote = char
        elif char == "#":
            return index
        index += 1
    return None


def _is_wrappable(line: str, column: int, limit: int) -> bool:
    """Return whether wrapping the comment could bring the line in bounds."""
    if column >= limit:
        return False
    text = line[column:].strip()
    if DIRECTIVE.match(text):
        return False
    room = limit - column - 2
    tokens = text.lstrip("#").split()
    return bool(tokens) and max(len(token) for token in tokens) <= room


def _next_code_indent(lines: Sequence[str], start: int) -> int:
    """Return indentation of the next nonblank, non-comment line."""
    for line in lines[start:]:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return len(line) - len(line.lstrip())
    return 0


def scan_lines(
    lines: Sequence[str], markdown: bool, limit: int
) -> list[tuple[int, str]]:
    """Return line-numbered comment violations in the supplied content."""
    violations: list[tuple[int, str]] = []
    in_fence = not markdown
    anchor: int | None = None

    for number, line in enumerate(lines, start=1):
        if markdown and line.lstrip().startswith(FENCE):
            in_fence = not in_fence
            anchor = None
            continue
        if not in_fence:
            continue

        column = _comment_column(line)
        if column is None:
            anchor = None
            continue

        if len(line) > limit and _is_wrappable(line, column, limit):
            violations.append(
                (number, f"comment past column {limit} ({len(line)} wide)")
            )

        if line[:column].strip() == "":
            if (
                anchor is not None
                and column != anchor
                and column > _next_code_indent(lines, number)
            ):
                violations.append(
                    (
                        number,
                        f"continuation comment at column {column}, expected {anchor}",
                    )
                )
            anchor = anchor if column == anchor else None
        else:
            anchor = column

    return violations


def find_violations(
    paths: Iterable[Path], limit: int | None = None
) -> list[tuple[Path, int, str]]:
    """Return every violation across the supplied Python and Markdown files."""
    if limit is None:
        limit = comment_limit()
    violations: list[tuple[Path, int, str]] = []
    for path in paths:
        if path.suffix not in (".py", ".md"):
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        violations.extend(
            (path, number, message)
            for number, message in scan_lines(
                content.split("\n"), path.suffix == ".md", limit
            )
        )
    return violations


def default_paths() -> list[Path]:
    """Return maintained Python and Markdown files covered by the policy."""
    nested = (
        path
        for root in SCAN_ROOTS
        for suffix in ("*.py", "*.md")
        for path in (REPO_ROOT / root).rglob(suffix)
    )
    top_level = (REPO_ROOT / name for name in TOP_LEVEL_FILES)
    return sorted(path for path in (*nested, *top_level) if path.is_file())


def main(argv: Sequence[str] | None = None) -> int:
    """Report comment-hygiene violations.

    Returns:
        One when violations are present, otherwise zero.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    paths = [Path(arg) for arg in args] if args else default_paths()
    violations = find_violations(paths)
    if not violations:
        return 0

    print(
        "Wrapped comments must align with their anchor, and wrappable comments "
        "must not push a line past the configured limit.",
        file=sys.stderr,
    )
    for path, number, message in violations:
        print(f"  {path}:{number}: {message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
