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

## Graceful completion of finite runs (`run_done_error`)

Upstream's parallel runner assumes environments reset forever; a run ends
by killing the workers. Real-world harnesses have a finite number of
episodes. `embodied.run.parallel.combined(..., run_done_error=SomeError)`
(`embodied/run/parallel.py`) treats that exception type, raised by
`env.step` on the reset after the final episode, as normal completion:

    final terminal transition -> replay / logger flush
    -> final agent checkpoint -> replay and logger checkpoints
    -> worker shutdown -> clean portal exit

The handoff uses a process-safe semaphore (`requested`, released once by
every environment on its own final reset) and two events (`actor_flushed`,
`checkpoint_persisted`). The actor closes its server only after all
environments reported exhaustion, so none is cut off mid-episode; it resolves
the replay and logger RPCs of a terminal transition before `actor_flushed` is
set; the replay server releases a learner blocked below the minimum fill; the
learner's replay streams treat the shutdown disconnect as completion, so its
prefetch threads exit instead of taking the agent process down; the replay
worker waits for its asynchronous chunk writes before exiting; and every
worker is joined once `portal.run` returned normally (a crash still kills the
others and raises, without joining: a thread blocked in a wait cannot be
killed and would hang the join). Any other exception still crashes the
worker as before. Two changes apply regardless of `run_done_error`:
`embodied.streams.Prefetch` (`embodied/core/streams.py`)
propagates source exhaustion as `StopIteration` instead of a worker crash,
and the learner opens no eval stream when `eval_envs <= 0`. Without
`run_done_error` the behavior is unchanged.

## Checkpoints at episode boundaries (`run.save_on_episode`)

Upstream checkpoints on a wall-clock timer (`run.save_every`). When a run is
a sequence of physical episodes, the natural moment to persist is the
episode boundary. With `run.save_on_episode: True` (`dreamerv3/configs.yaml`,
default `False`) the parallel actor (`embodied/run/parallel.py`) tracks the
mean reward per step of every environment's episode and, on each
`is_last`, asks the learner to save `ckpt/agent`; when that mean improves on
every episode seen so far it also asks for `ckpt/agent_best` (episodes that
end before the learner's first train step, i.e. before the replay reached
its minimum fill, are covered by the run's final checkpoint). The learner
services these requests between train steps, so a save lands after the
train step in flight. The wall-clock timer keeps working independently; set
`run.save_every: 0` to rely on episode boundaries only.

## Policy precompiled before the actor serves

`embodied.jax.Agent` compiles the policy lazily, so upstream's first real
`policy()` call pays the JAX compilation while an environment is already
waiting for an action. `Agent.precompile_policy(batch_size)`
(`embodied/jax/agent.py`) runs one dummy train-mode batch through the public
policy path and blocks until it executed; the dummy carry and actions are
discarded and the action counter is restored, so nothing observable
changes. `parallel_actor` (`embodied/run/parallel.py`) calls it between two
actor/learner rendezvous:

    restore checkpoint -> barrier -> precompile policy -> barrier
    -> start actor server -> environments connect

The warmup always runs for `embodied.jax.Agent`; there is no knob, since it
only moves the compilation ahead of the first request and changes nothing
observable.

The learner neither trains nor stages a policy sync until the second
rendezvous, and a failing warmup aborts the barrier so the learner cannot
deadlock. Agents without `precompile_policy` skip the step.

## Policy sync schedule (`jax.policy_sync_mode`, `jax.policy_sync_steps`)

In parallel training the learner stages fresh policy parameters after every
train step and upstream's actor installs them after its next `policy()`
call, so a policy may change on any environment step. `jax.policy_sync_mode`
(`dreamerv3/configs.yaml`, `embodied/jax/agent.py`) chooses when the actor
installs staged parameters:

| mode | installs after |
| --- | --- |
| `immediate` (default) | every policy call; upstream behavior |
| `episode` | a policy call in which any environment of the actor batch reports `is_last`; with one environment per actor batch (`run.envs: 1`, `run.actor_batch: 1`) each episode runs exactly one policy |
| `steps` | every `jax.policy_sync_steps`-th policy call (each call serves one actor batch) |

The installing call already dispatched with the previous parameters; the
new ones take effect from the following call. In the scheduled modes the
learner keeps the newest parameters staged (and releases the ones they
replace), whereas `immediate` keeps upstream's rule of holding the first
staged set until the actor takes it. The schedule counts actor policy calls,
so with `run.actor_batch: 1` a step is one environment step.

## Running the fork's tests

The tests added by this fork live next to the upstream ones under
`embodied/tests/` and need only `elements`, `portal`, `jax`, `gym`, and
`numpy` (the upstream test modules predate the `zerofun` to `portal` rename
and are not collected by this command):

```sh
python -m pytest embodied/tests/test_configs.py embodied/tests/test_from_gym.py \
    embodied/tests/test_float_images.py \
    embodied/tests/test_jax_agent_options.py \
    embodied/tests/test_parallel_lifecycle.py \
    embodied/tests/test_policy_precompile.py \
    embodied/tests/test_policy_sync.py
```
