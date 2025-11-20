import os
import pytest
import jax

# Setup JAX
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jaxmg import potrs


def test_in_specs_not_sequence_raises_assertion():
    A = jnp.eye(4)
    b = jnp.ones((4, 1))
    T_A = 1
    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    with pytest.raises(AssertionError, match="expected `in_specs` to be a tuple or list"):
        potrs(A, b, T_A, mesh=mesh, in_specs=P(None, "x"))


def test_in_specs_wrong_length_raises_assertion():
    A = jnp.eye(4)
    b = jnp.ones((4, 1))
    T_A = 1
    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    # single-element tuple should trigger the length assertion
    with pytest.raises(AssertionError, match="expected two `in_specs`"):
        potrs(A, b, T_A, mesh=mesh, in_specs=(P("x", None),))
def test_spec_a_valueerror_when_first_none_or_second_not_none():
    A = jnp.eye(4)
    b = jnp.ones((4, 1))
    T_A = 1
    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    # First entry None -> invalid
    with pytest.raises(ValueError, match="A must be sharded along the columns"):
        potrs(A, b, T_A, mesh=mesh, in_specs=(P(None, "x"), P(None, None)))
    # Second entry not None -> invalid
    with pytest.raises(ValueError, match="A must be sharded along the columns"):
        potrs(A, b, T_A, mesh=mesh, in_specs=(P("x", "y"), P(None, None)))


def test_spec_b_valueerror_when_not_replicated():
    A = jnp.eye(4)
    b = jnp.ones((4, 1))
    T_A = 1
    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    with pytest.raises(ValueError, match="b must be replicated along all shards"):
        potrs(A, b, T_A, mesh=mesh, in_specs=(P("x", None), P("x", None)))


def test_shape_mismatch_raises_assertion():
    A = jnp.eye(4)
    b = jnp.ones((5, 1))
    T_A = 1
    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    with pytest.raises(AssertionError, match="A and b must have the same number of columns"):
        potrs(A, b, T_A, mesh=mesh, in_specs=(P("x", None), P(None, None)))


def test_a_1d_raises_indexerror():
    # Due to the current implementation ordering, passing a 1D `a` raises an
    # IndexError when the code attempts to access a.shape[1]. This documents
    # the current behavior (should be tightened in future).
    A = jnp.ones((4,))
    b = jnp.ones((4, 1))
    T_A = 1
    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    with pytest.raises(IndexError):
        potrs(A, b, T_A, mesh=mesh, in_specs=(P("x", None), P(None, None)))


def test_b_1d_raises_assertion():
    A = jnp.eye(4)
    b = jnp.ones((4,))
    T_A = 1
    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    with pytest.raises(AssertionError, match="b must be a 2D array"):
        potrs(A, b, T_A, mesh=mesh, in_specs=(P("x", None), P(None, None)))
