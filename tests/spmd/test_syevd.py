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

devices = [d for d in jax.devices() if d.platform == "gpu"]
ndev = len(devices)

pytestmark = pytest.mark.skipif(
    ndev not in {1, 2, 4}, reason="Tests require 1, 2, or 4 GPUs"
)

mesh = jax.make_mesh((ndev,), ("x",))


def cusolver_solve_arange(N, T_A, dtype):
    A = jnp.diag(jnp.arange(N, dtype=dtype) + 1)
    eigenvalues_expected = jnp.diag(A)
    # Make mesh and place data
    A = jax.device_put(A, NamedSharding(mesh, P(None, "x")))
    eigenvalues, V = jax.jit(
        partial(syevd, mesh=mesh, in_specs=(P(None, "x"),)), static_argnums=1
    )(A, T_A=T_A)
    assert jnp.allclose(eigenvalues_expected, eigenvalues)
    eigenvalus_VtAV = jnp.diag(V.T @ A @ V)
    assert jnp.allclose(eigenvalus_VtAV, eigenvalues_expected)


def cusolver_solve_psd(N, T_A, dtype):
    A = random_psd(N, dtype=dtype, seed=1234)
    eigenvalues_expected, V_expected = jnp.linalg.eigh(A)
    # Make mesh and place data
    A = jax.device_put(A, NamedSharding(mesh, P(None, "x")))
    eigenvalues, V = jax.jit(
        partial(syevd, mesh=mesh, in_specs=(P(None, "x"),)), static_argnums=1
    )(A, T_A=T_A)
    norm_syevd = jnp.linalg.norm(V.T @ A - jnp.diag(eigenvalues) @ V)
    norm_lax = jnp.linalg.norm(
        V_expected.T @ A - jnp.diag(eigenvalues_expected) @ V_expected
    )
    assert jnp.isclose(norm_syevd, norm_lax, rtol=10, atol=0.0)


def cusolver_solve_arange_no_V(N, T_A, dtype):
    A = jnp.diag(jnp.arange(N, dtype=dtype) + 1)
    eigenvalues_expected = jnp.diag(A)
    # Make mesh and place data
    A = jax.device_put(A, NamedSharding(mesh, P(None, "x")))

    eigenvalues = jax.jit(
        partial(syevd, mesh=mesh, in_specs=(P(None, "x"),), return_eigenvectors=False),
        static_argnums=1,
    )(A, T_A=T_A)
    assert jnp.allclose(eigenvalues_expected, eigenvalues, rtol=10, atol=0.0)


def cusolver_solve_psd_no_V(N, T_A, dtype):
    A = random_psd(N, dtype=dtype, seed=1234)
    eigenvalues_expected = jnp.linalg.eigvalsh(A)
    # Make mesh and place data
    A = jax.device_put(A, NamedSharding(mesh, P(None, "x")))
    eigenvalues = jax.jit(
        partial(syevd, mesh=mesh, in_specs=(P(None, "x"),), return_eigenvectors=False),
        static_argnums=1,
    )(A, T_A=T_A)

    assert jnp.allclose(eigenvalues, eigenvalues_expected, rtol=10, atol=0.0)


def cusolver_solve_psd_sol_copy(N, T_A, dtype):
    A = random_psd(N, dtype=dtype, seed=1234)
    # Make mesh and place data
    A = jax.device_put(A, NamedSharding(mesh, P(None, "x")))
    out, V = jax.jit(
        partial(syevd, mesh=mesh, in_specs=(P(None, "x"),), return_eigenvectors=True),
        static_argnums=1,
    )(A, T_A=T_A)
    str_shards = []
    for shard in out.addressable_shards:
        str_shards.append(str(shard.data))
    assert all(l == str_shards[0] for l in str_shards)


def cusolver_solve_psd_sol_no_V_copy(N, T_A, dtype):
    A = random_psd(N, dtype=dtype, seed=1234)
    # Make mesh and place data
    A = jax.device_put(A, NamedSharding(mesh, P(None, "x")))
    out = jax.jit(
        partial(syevd, mesh=mesh, in_specs=(P(None, "x"),), return_eigenvectors=False),
        static_argnums=1,
    )(A, T_A=T_A)
    str_shards = []
    for shard in out.addressable_shards:
        str_shards.append(str(shard.data))
    assert all(l == str_shards[0] for l in str_shards)


@pytest.mark.parametrize(
    "dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128)
)
@pytest.mark.parametrize("T_A", (1, 2, 3))
@pytest.mark.parametrize("N", (4, 8, 10, 12))
def test_cusolver_solve_arange_dev_1(N, T_A, dtype):
    if ndev != 1:
        pytest.skip("This case is for exactly 1 GPU")
    cusolver_solve_arange(N, T_A, dtype)


@pytest.mark.parametrize(
    "dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128)
)
@pytest.mark.parametrize("T_A", (1, 2, 3))
@pytest.mark.parametrize("N", (4, 8, 10, 12))
def test_cusolver_solve_psd_dev_1(N, T_A, dtype):
    if ndev != 1:
        pytest.skip("This case is for exactly 1 GPU")
    cusolver_solve_psd(N, T_A, dtype)


@pytest.mark.parametrize(
    "dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128)
)
@pytest.mark.parametrize("T_A", (1, 2, 3))
@pytest.mark.parametrize("N", (4, 8, 10, 12))
def test_cusolver_solve_arange_no_v_dev_1(N, T_A, dtype):
    if ndev != 1:
        pytest.skip("This case is for exactly 1 GPU")
    cusolver_solve_arange_no_V(N, T_A, dtype)


@pytest.mark.parametrize(
    "dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128)
)
@pytest.mark.parametrize("T_A", (1, 2, 3))
@pytest.mark.parametrize("N", (4, 8, 10, 12))
def test_cusolver_solve_psd_no_v_dev_1(N, T_A, dtype):
    if ndev != 1:
        pytest.skip("This case is for exactly 1 GPU")
    cusolver_solve_psd_no_V(N, T_A, dtype)


@pytest.mark.parametrize(
    "dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128)
)
@pytest.mark.parametrize("T_A", (1, 2, 3))
@pytest.mark.parametrize("N", (8, 10, 12))
def test_cusolver_solve_arange_dev_2(N, T_A, dtype):
    if ndev != 2:
        pytest.skip("This case is for exactly 1 GPU")
    cusolver_solve_arange(N, T_A, dtype)


@pytest.mark.parametrize(
    "dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128)
)
@pytest.mark.parametrize("T_A", (1, 2, 3))
@pytest.mark.parametrize("N", (8, 10, 12))
def test_cusolver_solve_psd_dev_2(N, T_A, dtype):
    if ndev != 2:
        pytest.skip("This case is for exactly 2 GPUs")
    cusolver_solve_psd(N, T_A, dtype)


@pytest.mark.parametrize(
    "dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128)
)
@pytest.mark.parametrize("T_A", (1, 2, 3))
@pytest.mark.parametrize("N", (4, 8, 10, 12))
def test_cusolver_solve_psd_sol_copy_dev_2(N, T_A, dtype):
    if ndev != 2:
        pytest.skip("This case is for exactly 2 GPUs")
    cusolver_solve_psd_sol_copy(N, T_A, dtype)


@pytest.mark.parametrize(
    "dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128)
)
@pytest.mark.parametrize("T_A", (1, 2, 3))
@pytest.mark.parametrize("N", (4, 8, 10, 12))
def test_cusolver_solve_psd_sol_no_V_copy_dev_2(N, T_A, dtype):
    if ndev != 2:
        pytest.skip("This case is for exactly 2 GPUs")
    cusolver_solve_psd_sol_no_V_copy(N, T_A, dtype)


@pytest.mark.parametrize(
    "dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128)
)
@pytest.mark.parametrize("T_A", (1, 2, 3))
@pytest.mark.parametrize("N", (8, 10, 12))
def test_cusolver_solve_arange_no_v_dev_2(N, T_A, dtype):
    if ndev != 2:
        pytest.skip("This case is for exactly 2 GPUs")
    cusolver_solve_arange_no_V(N, T_A, dtype)


@pytest.mark.parametrize(
    "dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128)
)
@pytest.mark.parametrize("T_A", (1, 2, 3))
@pytest.mark.parametrize("N", (8, 10, 12))
def test_cusolver_solve_psd_no_v_dev_2(N, T_A, dtype):
    if ndev != 2:
        pytest.skip("This case is for exactly 2 GPUs")
    cusolver_solve_psd_no_V(N, T_A, dtype)


@pytest.mark.parametrize(
    "dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128)
)
@pytest.mark.parametrize("T_A", (1, 2, 4))
@pytest.mark.parametrize("N", (48, 60))
def test_cusolver_solve_arange_dev_4(N, T_A, dtype):
    if ndev != 4:
        pytest.skip("This case is for exactly 4 GPUs")
    cusolver_solve_arange(N, T_A, dtype)


@pytest.mark.parametrize(
    "dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128)
)
@pytest.mark.parametrize("T_A", (1, 2, 4))
@pytest.mark.parametrize("N", (48, 60))
def test_cusolver_solve_psd_dev_4(N, T_A, dtype):
    if ndev != 4:
        pytest.skip("This case is for exactly 4 GPUs")
    cusolver_solve_psd(N, T_A, dtype)


@pytest.mark.parametrize(
    "dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128)
)
@pytest.mark.parametrize("T_A", (1, 2, 4))
@pytest.mark.parametrize("N", (48, 60))
def test_cusolver_solve_arange_no_v_dev_4(N, T_A, dtype):
    if ndev != 4:
        pytest.skip("This case is for exactly 4 GPUs")
    cusolver_solve_arange_no_V(N, T_A, dtype)


@pytest.mark.parametrize(
    "dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128)
)
@pytest.mark.parametrize("T_A", (1, 2, 4))
@pytest.mark.parametrize("N", (48, 60))
def test_cusolver_solve_arange_no_v_dev_4(N, T_A, dtype):
    if ndev != 4:
        pytest.skip("This case is for exactly 4 GPUs")
    cusolver_solve_psd_no_V(N, T_A, dtype)
