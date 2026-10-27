"""Run the STAT 248 homework scripts in examples/ as integration tests.

They are excluded from the default test run, use ``pixi run examples`` or
``pytest -m examples``. Sample sizes are reduced via ``YASSM_EXAMPLES_FAST=1``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parent.parent / "examples"
SCRIPTS = sorted(EXAMPLES.glob("*/hw*.py"))


@pytest.mark.examples
@pytest.mark.parametrize("script", SCRIPTS, ids=[p.stem for p in SCRIPTS])
def test_example(script: Path, tmp_path: Path) -> None:
    env = os.environ | {"MPLBACKEND": "Agg", "YASSM_EXAMPLES_FAST": "1", "PYTHONWARNINGS": "ignore"}
    res = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert res.returncode == 0, res.stderr[-5000:]
