"""Provide driver functionality."""

from __future__ import annotations
from typing import Any


import time

import cloudpickle
import elements
import numpy as np
import portal


class Driver:
    """Represent driver."""

    def __init__(self, make_env_fns: Any, parallel: bool = True, **kwargs: Any) -> None:
        """Initialize the driver.

        Args:
            make_env_fns: Make environment fns value.
            parallel: Parallel value.
            kwargs: Keyword arguments forwarded to the wrapped callable.
        """
        assert (
            len(make_env_fns) >= 1
        ), "Expected number of make env fns to be at least 1."
        self.parallel = parallel
        self.kwargs = kwargs
        self.length = len(make_env_fns)
        if parallel:
            import multiprocessing as mp

            context = mp.get_context()
            self.pipes, pipes = zip(*[context.Pipe() for _ in range(self.length)])
            self.stop = context.Event()
            fns = [cloudpickle.dumps(fn) for fn in make_env_fns]
            self.procs = [
                portal.Process(self._env_server, self.stop, i, pipe, fn, start=True)
                for i, (fn, pipe) in enumerate(zip(fns, pipes))
            ]
            self.pipes[0].send(("act_space",))
            self.act_space = self._receive(self.pipes[0])
        else:
            self.envs = [fn() for fn in make_env_fns]
            self.act_space = self.envs[0].act_space
        self.callbacks = []
        self.acts = None
        self.carry = None
        self.reset()

    def reset(self, init_policy: Any | None = None) -> None:
        """Reset state.

        Args:
            init_policy: Init policy value.
        """
        self.acts = {
            k: np.zeros((self.length,) + v.shape, v.dtype)
            for k, v in self.act_space.items()
        }
        self.acts["reset"] = np.ones(self.length, bool)
        self.carry = init_policy and init_policy(self.length)

    def close(self) -> None:
        """Close state."""
        if self.parallel:
            [proc.kill() for proc in self.procs]
        else:
            [env.close() for env in self.envs]

    def on_step(self, callback: Any) -> None:
        """Handle on step.

        Args:
            callback: Function invoked when the operation completes.
        """
        self.callbacks.append(callback)

    def __call__(self, policy: Any, steps: int = 0, episodes: int = 0) -> None:
        """Apply the driver.

        Args:
            policy: Policy to process.
            steps: Steps to process.
            episodes: Episodes to process.
        """
        step, episode = 0, 0
        while step < steps or episode < episodes:
            step, episode = self._step(policy, step, episode)

    def _step(self, policy: Any, step: Any, episode: Any) -> tuple[Any, ...]:
        acts = self.acts
        assert acts is not None, "Driver actions must be initialized before stepping."
        assert all(
            len(x) == self.length for x in acts.values()
        ), "Expected every element to satisfy that number of x to equal self.length."
        assert all(
            isinstance(v, np.ndarray) for v in acts.values()
        ), "Expected all elements to satisfy the invariant."
        acts = [{k: v[i] for k, v in acts.items()} for i in range(self.length)]
        if self.parallel:
            [pipe.send(("step", act)) for pipe, act in zip(self.pipes, acts)]
            obs = [self._receive(pipe) for pipe in self.pipes]
        else:
            obs = [env.step(act) for env, act in zip(self.envs, acts)]
        obs = {k: np.stack([x[k] for x in obs]) for k in obs[0].keys()}
        logs = {k: v for k, v in obs.items() if k.startswith("log/")}
        obs = {k: v for k, v in obs.items() if not k.startswith("log/")}
        assert all(len(x) == self.length for x in obs.values()), obs
        self.carry, acts, outs = policy(self.carry, obs, **self.kwargs)
        assert all(k not in acts for k in outs), (list(outs.keys()), list(acts.keys()))
        if obs["is_last"].any():
            mask = ~obs["is_last"]
            acts = {k: self._mask(v, mask) for k, v in acts.items()}
        self.acts = {**acts, "reset": obs["is_last"].copy()}
        trans = {**obs, **acts, **outs, **logs}
        for i in range(self.length):
            trn = elements.tree.map(lambda x: x[i], trans)
            [fn(trn, i, **self.kwargs) for fn in self.callbacks]
        step += len(obs["is_first"])
        episode += obs["is_last"].sum()
        return step, episode

    def _mask(self, value: Any, mask: Any) -> Any:
        while mask.ndim < value.ndim:
            mask = mask[..., None]
        return value * mask.astype(value.dtype)

    def _receive(self, pipe: Any) -> Any:
        try:
            msg, arg = pipe.recv()
            if msg == "error":
                raise RuntimeError(arg)
            assert msg == "result", 'Expected msg to equal "result".'
            return arg
        except Exception:
            print("Terminating workers due to an exception.")
            [proc.kill() for proc in self.procs]
            raise

    @staticmethod
    def _env_server(stop: Any, envid: Any, pipe: Any, ctor: Any) -> None:
        try:
            ctor = cloudpickle.loads(ctor)
            env = ctor()
            while not stop.is_set():
                if not pipe.poll(0.1):
                    time.sleep(0.1)
                    continue
                try:
                    msg, *args = pipe.recv()
                except EOFError:
                    return
                if msg == "step":
                    assert len(args) == 1, "Expected number of args to equal 1."
                    act = args[0]
                    obs = env.step(act)
                    pipe.send(("result", obs))
                elif msg == "obs_space":
                    assert len(args) == 0, "Expected number of args to equal 0."
                    pipe.send(("result", env.obs_space))
                elif msg == "act_space":
                    assert len(args) == 0, "Expected number of args to equal 0."
                    pipe.send(("result", env.act_space))
                else:
                    raise ValueError(f"Invalid message {msg}")
        except ConnectionResetError:
            print("Connection to driver lost")
        except Exception as e:
            pipe.send(("error", e))
            raise
        finally:
            try:
                env.close()
            except Exception:
                pass
            pipe.close()
