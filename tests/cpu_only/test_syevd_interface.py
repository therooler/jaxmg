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
from functools import partial
from jaxmg import syevd
from jaxmg.utils import random_psd

devices = jax.devices()
ndev = len(devices)

pytestmark = pytest.mark.skipif(
    ndev not in {1},
    reason="Tests requires 1 device only"
)

mesh = jax.make_mesh((ndev,), ("x",))

@pytest.fixture
def setup_syevd_test():
    def _setup(N, T_A, dtype, a_shape=None, a_sharding=None):
        a = jnp.ones((N, N), dtype=dtype) if a_shape is None else jnp.ones(a_shape, dtype=dtype)
        a = jax.device_put(a, NamedSharding(mesh, a_sharding or P(None, "x")))
        return a, T_A, mesh
    return _setup

def test_a_not_2d(setup_syevd_test):
    a, T_A, mesh = setup_syevd_test(4, 32, jnp.float32, a_shape=(3,), a_sharding=P(None))
    with pytest.raises(AssertionError, match="a must be a 2D array"):
        syevd(a, T_A, mesh, (P(None, "x"),))

def test_in_specs_wrong_length(setup_syevd_test):
    a, T_A, mesh = setup_syevd_test(4, 32, jnp.float32)
    with pytest.raises(AssertionError, match="expected only one `in_specs`"):
        syevd(a, T_A, mesh, (P(None, "x"), P(None, "x")))

def test_spec_a_not_sharded_columns(setup_syevd_test):
    a, T_A, mesh = setup_syevd_test(4, 32, jnp.float32)
    bad_specs = [P("x", None), P("x", "x"), P(None, None)]
    for spec in bad_specs:
        with pytest.raises(ValueError, match="A must be sharded along the columns"):
            syevd(a, T_A, mesh, (spec,))

def test_T_A_too_large(setup_syevd_test):
    a, T_A, mesh = setup_syevd_test(4, 2048, jnp.float32)
    with pytest.raises(ValueError, match="T_A has a maximum value of 1024"):
        syevd(a, T_A, mesh, (P(None, "x"),))

def test_correct_call(setup_syevd_test):
    a, T_A, mesh = setup_syevd_test(4, 32, jnp.float32)
    out = syevd(a, T_A, mesh, (P(None, "x"),))
    assert isinstance(out, tuple)