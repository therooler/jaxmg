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
mesh = jax.make_mesh((ndev,), ("x",))

# Test cases
N_list = list(i * ndev for i in [1, 3, 4, 10])
T_A_list = [1, 2, 3, 5]


def cusolver_solve_arange(N, T_A, dtype):
    A = jnp.diag(jnp.arange(N, dtype=dtype) + 1)
    eigenvalues_expected = jnp.diag(A)
    # Make mesh and place data
    A = jax.device_put(A, NamedSharding(mesh, P("x", None)))
    eigenvalues, V = jax.jit(
        partial(syevd, mesh=mesh, in_specs=(P("x", None),)), static_argnums=1
    )(A, T_A=T_A)
    assert jnp.allclose(eigenvalues_expected, eigenvalues)
    eigenvalus_VtAV = jnp.diag(V.T @ A @ V)
    assert jnp.allclose(eigenvalus_VtAV, eigenvalues_expected)


def cusolver_solve_psd(N, T_A, dtype):
    A = random_psd(N, dtype=dtype, seed=1234)
    eigenvalues_expected, V_expected = jnp.linalg.eigh(A)
    # Make mesh and place data
    A = jax.device_put(A, NamedSharding(mesh, P("x", None)))
    eigenvalues, V = jax.jit(
        partial(syevd, mesh=mesh, in_specs=(P("x", None),)), static_argnums=1
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
    A = jax.device_put(A, NamedSharding(mesh, P("x", None)))

    eigenvalues = jax.jit(
        partial(syevd, mesh=mesh, in_specs=(P("x", None),), return_eigenvectors=False),
        static_argnums=1,
    )(A, T_A=T_A)
    assert jnp.allclose(eigenvalues_expected, eigenvalues, rtol=10, atol=0.0)


def cusolver_solve_psd_no_V(N, T_A, dtype):
    A = random_psd(N, dtype=dtype, seed=1234)
    eigenvalues_expected = jnp.linalg.eigvalsh(A)
    # Make mesh and place data
    A = jax.device_put(A, NamedSharding(mesh, P("x", None)))
    eigenvalues = jax.jit(
        partial(syevd, mesh=mesh, in_specs=(P("x", None),), return_eigenvectors=False),
        static_argnums=1,
    )(A, T_A=T_A)

    assert jnp.allclose(eigenvalues, eigenvalues_expected, rtol=10, atol=0.0)


@pytest.mark.parametrize(
    "dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128)
)
@pytest.mark.parametrize("T_A", T_A_list)
@pytest.mark.parametrize("N", N_list)
def test_cusolver_solve_arange(N, T_A, dtype):
    cusolver_solve_arange(N, T_A, dtype)


@pytest.mark.parametrize(
    "dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128)
)
@pytest.mark.parametrize("T_A", T_A_list)
@pytest.mark.parametrize("N", N_list)
def test_cusolver_solve_psd(N, T_A, dtype):
    cusolver_solve_psd(N, T_A, dtype)


@pytest.mark.parametrize(
    "dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128)
)
@pytest.mark.parametrize("T_A", T_A_list)
@pytest.mark.parametrize("N", N_list)
def test_cusolver_solve_arange_no_v(N, T_A, dtype):
    cusolver_solve_arange_no_V(N, T_A, dtype)


@pytest.mark.parametrize(
    "dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128)
)
@pytest.mark.parametrize("T_A", T_A_list)
@pytest.mark.parametrize("N", N_list)
def test_cusolver_solve_psd_no_v(N, T_A, dtype):
    cusolver_solve_psd_no_V(N, T_A, dtype)
