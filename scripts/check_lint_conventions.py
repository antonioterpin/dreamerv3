#!/usr/bin/env python3
"""Enforce narrow lint conventions Ruff cannot express.

Every ``noqa`` must name its rule and state a reason, either inline or in a
comment directly above. Literal module-and-symbol lookups through ``importlib``
are rejected because a normal import preserves type information; genuinely
dynamic module paths and symbol names remain supported. The checker is
stdlib-only for use from pre-commit.
"""

from __future__ import annotations

import ast
import re
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = ("dreamerv3", "embodied", "scores", "scripts")
TOP_LEVEL_FILES = ("plot.py", "setup.py")
NOQA = re.compile(r"#\s*noqa:\s*[A-Z]+[0-9]+(?:\s*,\s*[A-Z]+[0-9]+)*(.*)$")
BARE_NOQA = re.compile(r"#\s*noqa(?!:)")


def default_paths() -> list[Path]:
    """Return maintained Python files covered by the policy."""
    nested = (path for root in SCAN_ROOTS for path in (REPO_ROOT / root).rglob("*.py"))
    top_level = (REPO_ROOT / name for name in TOP_LEVEL_FILES)
    return sorted(path for path in (*nested, *top_level) if path.is_file())


def _relative(path: Path) -> str:
    """Return a repository-relative POSIX path when possible."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _is_import_module_call(node: ast.AST) -> bool:
    """Return whether a node calls import_module with one literal path."""
    if not isinstance(node, ast.Call) or len(node.args) != 1:
        return False
    func = node.func
    named = isinstance(func, ast.Attribute) and func.attr == "import_module"
    bare = isinstance(func, ast.Name) and func.id == "import_module"
    argument = node.args[0]
    return (
        (named or bare)
        and isinstance(argument, ast.Constant)
        and isinstance(argument.value, str)
    )


def _static_importlib_lookup(node: ast.AST) -> bool:
    """Return whether a node resolves a literal symbol from a literal module."""
    if isinstance(node, ast.Attribute):
        return _is_import_module_call(node.value)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
    ):
        attribute = node.args[1]
        return (
            _is_import_module_call(node.args[0])
            and isinstance(attribute, ast.Constant)
            and isinstance(attribute.value, str)
        )
    return False


def noqa_violations(source: str) -> list[tuple[int, str]]:
    """Return line-numbered bare or unmotivated noqa directives."""
    lines = source.splitlines()
    violations: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        if BARE_NOQA.search(line):
            violations.append((index + 1, "uses noqa without explicit rule codes"))
            continue
        match = NOQA.search(line)
        if match is None:
            continue
        inline = match.group(1).strip(" -\t")
        above = lines[index - 1].strip() if index else ""
        if not inline and not above.startswith("#"):
            violations.append((index + 1, "suppresses a rule without saying why"))
    return violations


def find_violations(paths: Iterable[Path]) -> list[tuple[str, int, str]]:
    """Return all convention breaches in the supplied Python files."""
    violations: list[tuple[str, int, str]] = []
    for path in paths:
        relative = _relative(path)
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        violations.extend(
            (relative, line, message) for line, message in noqa_violations(source)
        )
        violations.extend(
            (
                relative,
                node.lineno,
                "resolves a statically-known symbol through importlib",
            )
            for node in ast.walk(tree)
            if _static_importlib_lookup(node)
        )
    return violations


def main(argv: Sequence[str] | None = None) -> int:
    """Report convention breaches.

    Returns:
        One when violations are present, otherwise zero.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    paths = [Path(arg) for arg in args] if args else default_paths()
    paths = [path for path in paths if path.suffix == ".py" and path.exists()]
    violations = find_violations(paths)
    if not violations:
        return 0

    print(
        "Every noqa must name its rules and explain why; literal module and "
        "symbol lookups must use normal imports.",
        file=sys.stderr,
    )
    for path, line, message in violations:
        print(f"  {path}:{line}: {message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
