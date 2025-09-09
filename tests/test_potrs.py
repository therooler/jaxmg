import sys
import os
import pytest

this_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(this_dir, "..")
sys.path.append(src_path)
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding
from jaxmg import potrs
from jaxmg.utils import random_psd

from functools import partial

devices = [d for d in jax.devices() if d.platform == "gpu"]
ndev = len(devices)

pytestmark = pytest.mark.skipif(
    ndev not in {1, 2, 4},
    reason="Tests require 1, 2, or 4 GPUs"
)

mesh = jax.make_mesh((ndev,), ("x",))


def cusolver_solve_arange(N, T_A, dtype):
    A = jnp.diag(jnp.arange(N, dtype=dtype) + 1)
    b = jnp.ones((N, 1), dtype=dtype)
    ndev = len(devices)
    # Make mesh and place data
    mesh = jax.make_mesh((ndev,), ("x",))
    A = jax.device_put(A, NamedSharding(mesh, P(None, "x")))
    b = jax.device_put(b, NamedSharding(mesh, P(None, None)))

    out = jax.jit(
        partial(potrs, mesh=mesh, in_specs=(P(None, "x"), P(None, None))),
        static_argnums=2,
    )(A, b, T_A)
    expected_out = 1.0 / (jnp.arange(N, dtype=dtype) + 1)
    assert jnp.allclose(out.flatten(), expected_out)

def cusolver_solve_psd(N, T_A, dtype):
    A = random_psd(N, dtype=dtype, seed=1234)
    b = jnp.ones((N, 1), dtype=dtype)
    cfac = jax.scipy.linalg.cho_factor(A)
    expected_out = jax.scipy.linalg.cho_solve(cfac, b)
    ndev = len(devices)
    # Make mesh and place data
    mesh = jax.make_mesh((ndev,), ("x",))
    _A = jax.device_put(A, NamedSharding(mesh, P(None, "x")))
    _b = jax.device_put(b, NamedSharding(mesh, P(None, None)))

    out = jax.jit(
        partial(potrs, mesh=mesh, in_specs=(P(None, "x"), P(None, None))),
        static_argnums=2,
    )(_A, _b, T_A)
    norm_scipy = jnp.linalg.norm(b - A @ expected_out)
    norm_potrf = jnp.linalg.norm(b - A @ out)
    assert jnp.isclose(norm_scipy, norm_potrf, rtol=10, atol=0.0)

devices = jax.devices()
ndev = len(devices)
mesh = jax.make_mesh((ndev,), ("x",))


@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128))
@pytest.mark.parametrize("T_A", (1, 2, 3))
@pytest.mark.parametrize("N", (4, 8, 10, 12))
def test_cusolver_solve_arange_dev_1(N, T_A, dtype):
    if ndev != 1:
        pytest.skip("This case is for exactly 1 GPU")
    cusolver_solve_arange(N, T_A, dtype)

@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128))
@pytest.mark.parametrize("T_A", (1, 2, 3))
@pytest.mark.parametrize("N", (4, 8, 10, 12))
def test_cusolver_solve_psd_dev_1(N, T_A, dtype):
    if ndev != 1:
        pytest.skip("This case is for exactly 1 GPU")
    cusolver_solve_psd(N, T_A, dtype)


@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128))
@pytest.mark.parametrize("T_A", (1, 2, 3))
@pytest.mark.parametrize("N", (8, 10, 12))
def test_cusolver_solve_arange_dev_2(N, T_A, dtype):
    if ndev != 2:
        pytest.skip("This case is for exactly 2 GPUs")
    cusolver_solve_arange(N, T_A, dtype)

@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128))
@pytest.mark.parametrize("T_A", (1, 2, 3))
@pytest.mark.parametrize("N", (8, 10, 12))
def test_cusolver_solve_psd_dev_2(N, T_A, dtype):
    if ndev != 2:
        pytest.skip("This case is for exactly 2 GPUs")
    cusolver_solve_psd(N, T_A, dtype)


@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128))
@pytest.mark.parametrize("T_A", (1, 2, 4))
@pytest.mark.parametrize("N", (48, 60))
def test_cusolver_solve_arange_dev_4(N, T_A, dtype):
    if ndev != 4:
        pytest.skip("This case is for exactly 4 GPUs")
    cusolver_solve_arange(N, T_A, dtype)

@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128))
@pytest.mark.parametrize("T_A", (1, 2, 4))
@pytest.mark.parametrize("N", (48, 60))
def test_cusolver_solve_psd_dev_4(N, T_A, dtype):
    if ndev != 4:
        pytest.skip("This case is for exactly 4 GPUs")
    cusolver_solve_psd(N, T_A, dtype)