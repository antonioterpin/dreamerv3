"""Provide test configs functionality."""

from __future__ import annotations


import pathlib

import elements
import ruamel.yaml as yaml

CONFIGS = pathlib.Path(__file__).resolve().parents[2] / "dreamerv3" / "configs.yaml"


def test_run_args_declare_the_resume_regex() -> None:
    """Verify run args declare the resume regex."""
    configs = yaml.YAML(typ="safe").load(CONFIGS.read_text())
    config = elements.Config(configs["defaults"])
    args = elements.Config(**config.run, logdir=config.logdir)
    assert args.from_checkpoint == "", 'Expected args from checkpoint to equal "".'
    assert (
        args.from_checkpoint_regex == ""
    ), "parallel/train pass it to agent.load; empty loads every parameter"
