"""Provide transform functionality."""

from __future__ import annotations
from typing import Any


import threading
import re
from collections import Counter

import jax
from jax.sharding import PartitionSpec as P
import ninjax as nj

from . import nets as nn

LOCK = threading.Lock()


# Add tracer_sharding attribute to abstract values. This allows us to use
# shard_map based on layer callback shardings, even though JAX does not
# currently expose the shardings of tracer objects.
TRACER_SHARDINGS = {}


def init(
    fn: Any,
    mesh: Any,
    arg_shardings: Any,
    param_partition_rules: tuple[Any, ...] = (),
    act_partition_rules: tuple[Any, ...] = (),
    static_argnums: tuple[Any, ...] = (),
    dummy_inputs: tuple[Any, ...] = (),
    print_partition: bool = False,
) -> tuple[Any, ...]:
    """Handle init.

    Args:
        fn: Function to apply.
        mesh: Mesh value.
        arg_shardings: Arg shardings value.
        param_partition_rules: Param partition rules value.
        act_partition_rules: Act partition rules value.
        static_argnums: Static argnums value.
        dummy_inputs: Dummy inputs value.
        print_partition: Print partition value.

    Returns:
        Result of the operation.
    """

    def init(fun: Any, **jit_kwargs: Any) -> Any:
        """Handle init.

        Args:
            fun: Fun value.
            jit_kwargs: Jit kwargs value.

        Returns:
            Result of the operation.
        """
        if not getattr(fun, "_is_pure", False):
            fun = nj.pure(fun)

        def wrapper(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
            """Handle wrapper.

            Args:
                args: Positional arguments forwarded to the wrapped callable.
                kwargs: Keyword arguments forwarded to the wrapped callable.

            Returns:
                Result of the operation.
            """
            state, out = fun(*args, create=True, modify=True, ignore=True, **kwargs)
            del out
            return state, ()

        return wrapper

    fn = init(fn)

    def fn(*args: Any, inner: Any = fn) -> Any:
        """Handle function.

        Args:
            inner: Inner value.
            args: Positional arguments forwarded to the wrapped callable.

        Returns:
            Result of the operation.
        """
        params, seed, *args = args
        old = nn.LAYER_CALLBACK
        nn.LAYER_CALLBACK = create_layer_callback(mesh, act_partition_rules)
        params, _ = inner(params, *args, seed=seed)
        nn.LAYER_CALLBACK = old
        return params

    fn = jax.jit(fn, static_argnums=static_argnums)

    params_shapes = fn.eval_shape(*dummy_inputs)
    params_sharding, grouping = resolve_rules(
        params_shapes, param_partition_rules, mesh
    )
    if print_partition:
        print_grouping(grouping)

    fn = jax.jit(
        fn,
        in_shardings=arg_shardings,
        out_shardings=params_sharding,
        static_argnums=static_argnums,
        donate_argnums=None,
    )
    params = fn(*dummy_inputs)

    return params, params_sharding


def apply(
    fn: Any,
    mesh: Any,
    in_shardings: Any,
    out_shardings: Any,
    partition_rules: tuple[Any, ...] = (),
    static_argnums: tuple[Any, ...] = (),
    single_output: bool = False,
    return_params: bool = False,
    donate_params: bool = False,
    # shard_map specific
    split_rng: bool = True,
    use_shardmap: bool = False,
    first_outnums: tuple[Any, ...] = (),
) -> Any:
    """Apply state.

    Args:
        fn: Function to apply.
        mesh: Mesh value.
        in_shardings: In shardings value.
        out_shardings: Out shardings value.
        partition_rules: Partition rules value.
        static_argnums: Static argnums value.
        single_output: Single output value.
        return_params: Return parameters value.
        donate_params: Donate parameters value.
        split_rng: Split random number generator value.
        use_shardmap: Use shardmap value.
        first_outnums: First outnums value.

    Returns:
        Result of the operation.
    """
    if single_output:
        assert len(out_shardings) == 1, "Expected number of out shardings to equal 1."

    def fn(*args: Any, inner: Any = fn) -> Any:
        """Handle function.

        Args:
            inner: Inner value.
            args: Positional arguments forwarded to the wrapped callable.

        Returns:
            Result of the operation.
        """
        if donate_params:
            donated, allocated, seed, *args = args
            params = {**donated, **allocated}
        else:
            params, seed, *args = args
        if use_shardmap and len(mesh.devices) > 1 and split_rng:
            seed = jax.random.fold_in(seed, jax.lax.axis_index("d"))
        params, outs = inner(params, *args, seed=seed)
        outs = (outs,) if single_output else outs
        assert isinstance(outs, tuple), "Expected outs to have type tuple."
        return (params, *outs) if return_params else outs

    if use_shardmap and len(mesh.devices) > 1:

        def fn(*args: Any, inner: Any = fn) -> Any:
            outs = list(inner(*args))
            for i in first_outnums:
                outs[i] = jax.tree.map(lambda x: x[None], outs[i])
            return tuple(outs)

        from jax.experimental.shard_map import shard_map

        ispecs = list(jax.tree.map(lambda s: s.spec, in_shardings))
        for i in sorted(static_argnums):
            ispecs.insert(i, None)
        ispecs = tuple(ispecs)
        ospecs = jax.tree.map(lambda s: s.spec, out_shardings)
        fn = shard_map(fn, mesh, ispecs, ospecs, check_rep=False)

        def fn(*args: Any, inner: Any = fn) -> Any:
            outs = list(inner(*args))
            for i in first_outnums:
                outs[i] = jax.tree.map(lambda x: x[0], outs[i])
            return tuple(outs)

    if single_output:

        def fn(*args: Any, inner: Any = fn) -> Any:
            outs = inner(*args)
            assert len(outs) == 1, "Expected number of outs to equal 1."
            return outs[0]

    if single_output:
        out_shardings = out_shardings[0]
    donate = [0] if donate_params else []

    if not use_shardmap:

        def fn(*args: Any, inner: Any = fn) -> Any:
            with LOCK:
                old = nn.LAYER_CALLBACK
                nn.LAYER_CALLBACK = create_layer_callback(mesh, partition_rules)
                outs = inner(*args)
                nn.LAYER_CALLBACK = old
            return outs

    fn = jax.jit(
        fn,
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        static_argnums=static_argnums,
        donate_argnums=donate,
    )

    return fn


def create_layer_callback(mesh: Any, partition_rules: Any) -> Any:
    """Create layer callback.

    Args:
        mesh: Mesh value.
        partition_rules: Partition rules value.

    Returns:
        Result of the operation.

    Raises:
        Exception: If the operation cannot be completed.
    """

    def layer_callback(y: Any, name: Any) -> Any:
        """Handle layer callback.

        Args:
            y: Y value.
            name: Name value.

        Returns:
            Result of the operation.

        Raises:
            Exception: If the operation cannot be completed.
        """
        name = f"{nj.ninjax.SCOPE}/{name}"
        for rule, spec in partition_rules:
            if re.search(rule, name):
                sharding = jax.sharding.NamedSharding(mesh, spec)

                def apply(y: Any) -> Any:
                    y = jax.lax.with_sharding_constraint(y, sharding)
                    if not hasattr(type(y), "tracer_shardings"):
                        type(y).tracer_sharding = property(
                            lambda self: TRACER_SHARDINGS[id(self)]
                        )
                    TRACER_SHARDINGS[id(y)] = sharding
                    return y

                return jax.tree.map(apply, y)
        else:
            raise Exception(f"No matching rule found for activation key: {name}")

    return layer_callback


def resolve_rules(params: Any, partition_rules: Any, mesh: Any) -> tuple[Any, ...]:
    """Handle resolve rules.

    Args:
        params: Parameters value.
        partition_rules: Partition rules value.
        mesh: Mesh value.

    Returns:
        Result of the operation.

    Raises:
        Exception: If the operation cannot be completed.
    """
    if len(partition_rules) == 0:
        partition_rules = [(".*", P())]
    params_spec, grouping = dict(), dict()
    for k in params.keys():
        for rule, spec in partition_rules:
            if re.search(rule, k):
                params_spec[k] = spec
                if rule not in grouping:
                    grouping[rule] = []
                grouping[rule].append(k)
                break
        else:
            raise Exception(f"No matching rule found for param key: {k}")
    assert set(params.keys()) == set(
        params_spec.keys()
    ), "Expected keys in params keys() to equal set(params_spec.keys())."
    sharding = jax.tree.map(
        lambda spec: jax.sharding.NamedSharding(mesh, spec), params_spec
    )
    return sharding, grouping


def print_grouping(grouping: Any) -> None:
    """Handle print grouping.

    Args:
        grouping: Grouping value.
    """
    for rule, ps in grouping.items():
        if len(ps) == 0:
            continue
        print(f'Partition rule "{rule}" matches {len(ps)} param tensors')
        ks = ["/".join(p.split("/")[-2:]) for p in ps]
        ks = Counter(ks)
        ks = ks.most_common(len(ks))
        ks = [f"- .../{k}: {v}" for k, v in ks]
        print("\n".join(ks))
