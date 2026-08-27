import io
import contextlib

import pytest

from embodied.jax.agent import Agent, Options


def _agent(**options):
    agent = object.__new__(Agent)
    agent.jaxcfg = Options(**options)
    return agent


def test_options_default_to_verbose():
    assert Options().verbose is True


@pytest.mark.parametrize("verbose", [True, False])
def test_stdout_is_suppressed_only_when_not_verbose(verbose):
    agent = _agent(verbose=verbose)
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        with agent._stdout_unless_verbose():
            print("inside")
        print("outside")
    lines = captured.getvalue().split()
    assert ("inside" in lines) == verbose, lines
    assert "outside" in lines, "Suppression must end with the context"
