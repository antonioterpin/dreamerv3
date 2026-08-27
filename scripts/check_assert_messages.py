#!/usr/bin/env python3
"""Fail on an ``assert`` that carries no message.

Every assertion states the invariant it protects so failures remain useful
without reconstructing intent from the surrounding source. The checker is
stdlib-only and can therefore run in the code-style pre-commit environment.
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = ("dreamerv3", "embodied", "scores")
TOP_LEVEL_FILES = ("plot.py", "setup.py")


def default_paths() -> list[Path]:
    """Return every Python file covered by the assert-message policy."""
    nested = (path for root in SCAN_ROOTS for path in (REPO_ROOT / root).rglob("*.py"))
    top_level = (REPO_ROOT / name for name in TOP_LEVEL_FILES)
    return sorted(path for path in (*nested, *top_level) if path.is_file())


def _in_scope(path: Path) -> bool:
    """Return whether a path belongs to DreamerV3's checked Python sources."""
    try:
        relative = path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return False
    return relative in TOP_LEVEL_FILES or any(
        relative.startswith(f"{root}/") for root in SCAN_ROOTS
    )


def find_violations(paths: Iterable[Path]) -> list[tuple[Path, int]]:
    """Return the file and line of every message-free assert.

    Args:
        paths: Candidate files to scan. Non-Python and out-of-scope paths are
            ignored so pre-commit can pass a mixed staged-file list.

    Returns:
        Message-free assert locations in scan order.
    """
    violations: list[tuple[Path, int]] = []
    for path in paths:
        if path.suffix != ".py" or not _in_scope(path):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        violations.extend(
            (path, node.lineno)
            for node in ast.walk(tree)
            if isinstance(node, ast.Assert) and node.msg is None
        )
    return violations


def main(argv: Sequence[str] | None = None) -> int:
    """Report message-free asserts.

    Args:
        argv: Optional command-line paths to check.

    Returns:
        One when violations are present, otherwise zero.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    paths = [Path(arg) for arg in args] if args else default_paths()
    violations = sorted(find_violations(paths))
    if not violations:
        return 0

    print(
        "Every assert must state the invariant it expected. These asserts "
        "carry no message:",
        file=sys.stderr,
    )
    for path, line in violations:
        print(f"  {path}:{line}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
