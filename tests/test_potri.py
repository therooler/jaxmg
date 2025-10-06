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

devices = [d for d in jax.devices() if d.platform == "gpu"]
ndev = len(devices)

pytestmark = pytest.mark.skipif(
    ndev not in {1, 2, 4},
    reason="Tests require 1, 2, or 4 GPUs"
)

mesh = jax.make_mesh((ndev,), ("x",))


def cusolver_solve_arange(N, T_A, dtype):
    A = jnp.diag(jnp.arange(N, dtype=dtype) + 1)
    # Make mesh and place data
    A = jax.device_put(A, NamedSharding(mesh, P(None, "x")))

    out = jax.jit(
        partial(potri, mesh=mesh, in_specs=(P(None, "x"),)), static_argnums=1
    )(A, T_A=T_A)
    expected_out = jnp.diag(1.0 / (jnp.arange(N, dtype=dtype) + 1))
    assert jnp.allclose(out, expected_out)

def cusolver_solve_psd(N, T_A, dtype):
    A = random_psd(N, dtype=dtype, seed=1234)
    expected_out = jnp.linalg.inv(A)
    # Make mesh and place data
    _A = jax.device_put(A, NamedSharding(mesh, P(None, "x")))
    out = jax.jit(
        partial(potri, mesh=mesh, in_specs=(P(None, "x"),)), static_argnums=1
    )(_A, T_A=T_A)
    assert jnp.allclose(A.conj().T, A)
    norm_potri = jnp.linalg.norm(A @ out - jnp.eye(N, dtype=dtype))
    norm_lax = jnp.linalg.norm(A @ expected_out - jnp.eye(N, dtype=dtype))
    assert jnp.isclose(norm_potri, norm_lax, rtol=10, atol=0.0)


def cusolver_solve_non_psd(N, T_A, dtype):
    A = jnp.diag(jnp.arange(N, dtype=dtype) - 1)
    # Make mesh and place data
    _A = jax.device_put(A, NamedSharding(mesh, P(None, "x")))
    out, status = jax.jit(
        partial(potri, mesh=mesh, in_specs=(P(None, "x"),), return_status=True), static_argnums=1,
    )(_A, T_A=T_A)
    status.block_until_ready()
    out.block_until_ready()
    assert status==7
    assert jnp.all(jnp.isnan(out))

def cusolver_solve_non_symm(N, T_A, dtype):
    A = jnp.diag(jnp.arange(N, dtype=dtype) + 1)
    #TODO: For some reason the solver does not fail when we set this to 1.0.
    A = A.at[1,0].set(2.0)
    # Make mesh and place data
    _A = jax.device_put(A, NamedSharding(mesh, P(None, "x")))
    out, status = jax.jit(
        partial(potri, mesh=mesh, in_specs=(P(None, "x"),), return_status=True), static_argnums=1,
    )(_A, T_A=T_A)
    status.block_until_ready()
    out.block_until_ready()
    assert status==7
    assert jnp.all(jnp.isnan(out))

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
@pytest.mark.parametrize("N", (4, 8, 10, 12))
def test_cusolver_solve_non_psd_dev_1(N, T_A, dtype):
    if ndev != 1:
        pytest.skip("This case is for exactly 1 GPU")
    cusolver_solve_non_psd(N, T_A, dtype)

@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128))
@pytest.mark.parametrize("T_A", (1, 2, 3))
@pytest.mark.parametrize("N", (4, 8, 10, 12))
def test_cusolver_solve_non_symm_dev_1(N, T_A, dtype):
    if ndev != 1:
        pytest.skip("This case is for exactly 1 GPU")
    cusolver_solve_non_symm(N, T_A, dtype)


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
@pytest.mark.parametrize("T_A", (1, 2, 3))
@pytest.mark.parametrize("N", (8, 10, 12))
def test_cusolver_solve_non_psd_dev_2(N, T_A, dtype):
    if ndev != 2:
        pytest.skip("This case is for exactly 2 GPUs")
    cusolver_solve_non_psd(N, T_A, dtype)

@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128))
@pytest.mark.parametrize("T_A", (1, 2, 3))
@pytest.mark.parametrize("N", (8, 10, 12))
def test_cusolver_solve_non_symm_dev_2(N, T_A, dtype):
    if ndev != 2:
        pytest.skip("This case is for exactly 2 GPUs")
    cusolver_solve_non_symm(N, T_A, dtype)

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

@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128))
@pytest.mark.parametrize("T_A", (1, 2, 4))
@pytest.mark.parametrize("N", (48, 60))
def test_cusolver_solve_non_psd_dev_4(N, T_A, dtype):
    if ndev != 4:
        pytest.skip("This case is for exactly 4 GPUs")
    cusolver_solve_non_psd(N, T_A, dtype)

@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128))
@pytest.mark.parametrize("T_A", (1, 2, 4))
@pytest.mark.parametrize("N", (48, 60))
def test_cusolver_solve_non_symm_dev_4(N, T_A, dtype):
    if ndev != 4:
        pytest.skip("This case is for exactly 4 GPUs")
    cusolver_solve_non_symm(N, T_A, dtype)

def test_potri_assert_a_2d():
    N = 4
    T_A = 1
    dtype = jnp.float32
    A = jnp.ones((N,), dtype=dtype)  # Not 2D
    A = jax.device_put(A, NamedSharding(mesh, P("x")))
    with pytest.raises(AssertionError, match="a must be a 2D array"):
        potri(A, T_A=T_A, mesh=mesh, in_specs=(P("x"),))

def test_potri_assert_in_specs_len():
    N = 4
    T_A = 1
    dtype = jnp.float32
    A = jnp.eye(N, dtype=dtype)
    A = jax.device_put(A, NamedSharding(mesh, P(None, "x")))
    # Pass tuple of length 2
    with pytest.raises(AssertionError, match="expected only one `in_specs`"):
        potri(A, T_A=T_A, mesh=mesh, in_specs=(P(None, "x"), P(None, "x")))

def test_potri_valueerror_spec_a():
    N = 4
    T_A = 1
    dtype = jnp.float32
    A = jnp.eye(N, dtype=dtype)
    # Wrong sharding: first axis sharded
    A = jax.device_put(A, NamedSharding(mesh, P("x", None)))
    with pytest.raises(ValueError, match="must be sharded along the columns"):
        potri(A, T_A=T_A, mesh=mesh, in_specs=(P("x", None),))