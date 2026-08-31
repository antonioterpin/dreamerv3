"""Regression tests for Ninjax module annotation compatibility."""

import subprocess
import sys
from pathlib import Path


def test_embodied_import_resolves_ninjax_field_annotations() -> None:
    """Import annotated Ninjax modules in a fresh interpreter."""
    result = subprocess.run(
        [sys.executable, "-c", "import embodied; import dreamerv3.rssm"],
        check=False,
        capture_output=True,
        cwd=Path(__file__).parents[2],
        text=True,
    )

    assert result.returncode == 0, result.stderr
