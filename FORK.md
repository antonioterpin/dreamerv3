# Changes relative to danijar/dreamerv3

This repository is a fork of [danijar/dreamerv3](https://github.com/danijar/dreamerv3).
It carries a small set of adaptations needed to drive DreamerV3 from an
external, real-time training harness. Every change is generic (nothing in
this fork knows about any particular downstream project) and, unless stated
otherwise, defaults to the upstream behavior.

Each entry names the files it touches and the configuration knob (if any)
that enables it, so the fork can be re-based onto upstream and the list
doubles as a migration checklist.

## Compatibility fixes

- **Keyword-only `jax.jit` sharding arguments** (`embodied/jax/agent.py`,
  `embodied/jax/internal.py`, `embodied/jax/transform.py`): recent JAX
  releases reject positional `in_shardings` / `out_shardings` /
  `donate_argnums`. All call sites pass them by keyword, which is accepted by
  every JAX version upstream supports.
- **Gym `>= 0.26` step / reset tuples** (`embodied/envs/from_gym.py`):
  `FromGym` accepts `reset()` returning `(obs, info)` and `step()` returning
  the five-tuple `(obs, reward, terminated, truncated, info)`, in addition to
  the legacy forms. `is_last` is `terminated or truncated`; `is_terminal`
  follows `terminated` unless the info dict overrides it.
- **`reset` not forwarded to Dict-action Gym envs** (`embodied/envs/from_gym.py`):
  `reset` is embodied's own action key and `FromGym` consumes it; with a
  `Dict` action space it used to be passed on to the wrapped environment,
  which strict environments reject.
- **`run.from_checkpoint_regex` declared** (`dreamerv3/configs.yaml`): the
  runners pass `args.from_checkpoint_regex` to `agent.load`, but the key was
  never declared, so resuming with `run.from_checkpoint` raised
  `AttributeError`. Empty (the default) loads every parameter
  (`embodied/tests/test_configs.py`).
- **`np.ranodm` typo** in `embodied/core/streams.py` (`Mixture.__next__`).

## Floating point image observations

Upstream requires every image observation (rank-3 space) to be `uint8`.
The encoder (`dreamerv3/rssm.py`) now also accepts floating point images,
which are assumed to lie in `[0, 1]` (the same range the decoder's sigmoid
output and reconstruction loss already use; `uint8` images are still divided
by 255). The open-loop video summaries in `Agent.report`
(`dreamerv3/agent.py`) are rendered for `uint8` images only, since a float
image has no pixel range to draw. No configuration is needed; the dtype of
the observation space selects the behavior.

## Quiet initialization (`jax.verbose`)

`embodied.jax.Agent` prints the ninjax module tree, parameter summaries,
partition groupings, and compilation chatter while creating parameters and
compiling `train` / `report`. Setting `jax.verbose: False`
(`dreamerv3/configs.yaml`, default `True`) redirects that output away while
those two phases run; the `elements.print` status lines and the cost /
memory analysis stay visible. `jax.profiler` is also declared in
`configs.yaml` (it was only an `Options` default before), so both knobs can
be overridden from a config block or the command line.

## Running the fork's tests

The tests added by this fork live next to the upstream ones under
`embodied/tests/` and need only `elements`, `portal`, `jax`, `gym`, and
`numpy` (the upstream test modules predate the `zerofun` to `portal` rename
and are not collected by this command):

```sh
python -m pytest embodied/tests/test_from_gym.py \
    embodied/tests/test_float_images.py \
    embodied/tests/test_jax_agent_options.py
```
