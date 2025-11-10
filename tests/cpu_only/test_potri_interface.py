import sys
import os

this_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(this_dir, "..")
print(src_path)
sys.path.append(src_path)
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding
import pytest
from jaxmg import potri
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
def setup_potri_test():
    def _setup(N=4, T_A=1, dtype=jnp.float32, a_shape=None, a_sharding=None):
        A = jnp.eye(N, dtype=dtype) if a_shape is None else jnp.ones(a_shape, dtype=dtype)
        A = jax.device_put(A, NamedSharding(mesh, a_sharding or P(None, "x")))
        return A, T_A, mesh
    return _setup

def test_potri_assert_a_2d(setup_potri_test):
    A, T_A, mesh= setup_potri_test(a_shape=(4,), a_sharding=P("x",))
    with pytest.raises(AssertionError, match="a must be a 2D array"):
        potri(A, T_A=T_A, mesh=mesh, in_specs=(P("x"),))

def test_potri_assert_in_specs_len(setup_potri_test):
    A, T_A, mesh = setup_potri_test(a_sharding=P(None, "x"))
    # Pass tuple of length 2
    with pytest.raises(AssertionError, match="expected only one `in_specs`"):
        potri(A, T_A=T_A, mesh=mesh, in_specs=(P(None, "x"), P(None, "x")))

def test_potri_valueerror_spec_a(setup_potri_test):
    A, T_A, mesh = setup_potri_test(a_sharding=P("x", None))  # Wrong sharding: first axis sharded
    with pytest.raises(ValueError, match="must be sharded along the columns"):
        potri(A, T_A=T_A, mesh=mesh, in_specs=(P("x", None),))
