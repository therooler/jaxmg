import os
import pytest
import jax
# Setup JAX before anything else imports it
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding
from jaxmg import potrs
from jaxmg.utils import random_psd

from functools import partial

devices = jax.devices()
ndev = len(devices)

pytestmark = pytest.mark.skipif(
    ndev not in {1},
    reason="Tests requires 1 device only"
)

mesh = jax.make_mesh((ndev,), ("x",))

@pytest.fixture
def setup_potrs_test():
    def _setup(N, T_A, dtype, a_shape=None, b_shape=None, a_sharding=None, b_sharding=None):
        A = jnp.eye(N, dtype=dtype) if a_shape is None else jnp.ones(a_shape, dtype=dtype)
        b = jnp.ones((N, 1), dtype=dtype) if b_shape is None else jnp.ones(b_shape, dtype=dtype)
        A = jax.device_put(A, NamedSharding(mesh, a_sharding or P(None, "x")))
        b = jax.device_put(b, NamedSharding(mesh, b_sharding or P(None, None)))
        return A, b, T_A, mesh
    return _setup

def test_potrs_assert_shape_rows(setup_potrs_test):
    A, b, T_A, mesh = setup_potrs_test(4, 1, jnp.float32, b_shape=(5, 1))
    with pytest.raises(AssertionError, match="A and b must have the same number of rows"):
        potrs(A, b, T_A, mesh=mesh, in_specs=(P(None, "x"), P(None, None)))

def test_potrs_assert_a_2d(setup_potrs_test):
    A, b, T_A, mesh = setup_potrs_test(4, 1, jnp.float32, a_shape=(4,), a_sharding=P("x",))
    with pytest.raises(AssertionError, match="a must be a 2D array"):
        potrs(A, b, T_A, mesh=mesh, in_specs=(P("x",), P(None, None)))

def test_potrs_assert_b_2d(setup_potrs_test):
    A, b, T_A, mesh = setup_potrs_test(4, 1, jnp.float32, b_shape=(4,), b_sharding=P(None,))
    with pytest.raises(AssertionError, match="b must be a 2D array"):
        potrs(A, b, T_A, mesh=mesh, in_specs=(P(None, "x"), P(None,)))

def test_potrs_assert_in_specs_len(setup_potrs_test):
    A, b, T_A, mesh = setup_potrs_test(4, 1, jnp.float32)
    with pytest.raises(AssertionError, match="expected two `in_specs`"):
        potrs(A, b, T_A, mesh=mesh, in_specs=(P("x", None),))

def test_potrs_valueerror_spec_a(setup_potrs_test):
    A, b, T_A, mesh = setup_potrs_test(4, 1, jnp.float32, a_sharding=P("x", None))
    with pytest.raises(ValueError, match="A must be sharded along the columns"):
        potrs(A, b, T_A, mesh=mesh, in_specs=(P("x", None), P(None, None)))

def test_potrs_valueerror_spec_b(setup_potrs_test):
    A, b, T_A, mesh = setup_potrs_test(4, 1, jnp.float32, b_sharding=P("x", None))
    with pytest.raises(ValueError, match="b must be replicated along all shards"):
        potrs(A, b, T_A, mesh=mesh, in_specs=(P(None, "x"), P("x", None)))

def test_potrs_valueerror_spec_b(setup_potrs_test):
    A, b, T_A, mesh = setup_potrs_test(4, 1, jnp.float32)
    with pytest.raises(AssertionError, match="expected `in_specs` to be a tuple or list"):
        potrs(A, b, T_A, mesh=mesh, in_specs=P(None, "x"))