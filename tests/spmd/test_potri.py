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
mesh = jax.make_mesh((ndev,), ("x",))
# Test cases
N_list = list(i * ndev for i in [2, 3, 4, 10])
T_A_list = [1, 2, 3, 5]


def cusolver_solve_arange(N, T_A, dtype):
    A = jnp.diag(jnp.arange(N, dtype=dtype) + 1)
    # Make mesh and place data
    A = jax.device_put(A, NamedSharding(mesh, P("x", None)))

    out = jax.jit(
        partial(potri, mesh=mesh, in_specs=(P("x", None),)), static_argnums=1
    )(A, T_A=T_A)
    expected_out = jnp.diag(1.0 / (jnp.arange(N, dtype=dtype) + 1))
    assert jnp.allclose(out, expected_out)


def cusolver_solve_psd(N, T_A, dtype):
    A = random_psd(N, dtype=dtype, seed=1234)
    expected_out = jnp.linalg.inv(A)
    # Make mesh and place data
    _A = jax.device_put(A, NamedSharding(mesh, P("x", None)))
    out = jax.jit(
        partial(potri, mesh=mesh, in_specs=(P("x", None),)), static_argnums=1
    )(_A, T_A=T_A)
    assert jnp.allclose(A.conj().T, A)
    norm_potri = jnp.linalg.norm(A @ out - jnp.eye(N, dtype=dtype))
    norm_lax = jnp.linalg.norm(A @ expected_out - jnp.eye(N, dtype=dtype))
    assert jnp.isclose(norm_potri, norm_lax, rtol=10, atol=0.0)


def cusolver_solve_non_psd(N, T_A, dtype):
    A = jnp.diag(jnp.arange(N, dtype=dtype) - 1)
    # Make mesh and place data
    _A = jax.device_put(A, NamedSharding(mesh, P("x", None)))
    out, status = jax.jit(
        partial(potri, mesh=mesh, in_specs=(P("x", None),), return_status=True),
        static_argnums=1,
    )(_A, T_A=T_A)
    status.block_until_ready()
    out.block_until_ready()
    assert status == 7
    assert jnp.all(jnp.isnan(out))


def cusolver_solve_non_symm(N, T_A, dtype):
    A = jnp.diag(jnp.arange(N, dtype=dtype) + 1)
    # TODO: For some reason the solver does not fail when we set this to 1.0.
    A = A.at[0,1].set(2.0)
    # Make mesh and place data
    _A = jax.device_put(A, NamedSharding(mesh, P("x", None)))
    out, status = jax.jit(
        partial(potri, mesh=mesh, in_specs=(P("x", None),), return_status=True),
        static_argnums=1,
    )(_A, T_A=T_A)
    status.block_until_ready()
    out.block_until_ready()
    assert status == 7
    assert jnp.all(jnp.isnan(out))


@pytest.mark.parametrize(
    "dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128)
)
@pytest.mark.parametrize("T_A", T_A_list)
@pytest.mark.parametrize("N", N_list)
def test_cusolver_solve_arange_dev_1(N, T_A, dtype):
    cusolver_solve_arange(N, T_A, dtype)


@pytest.mark.parametrize(
    "dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128)
)
@pytest.mark.parametrize("T_A", T_A_list)
@pytest.mark.parametrize("N", N_list)
def test_cusolver_solve_psd_dev_1(N, T_A, dtype):
    cusolver_solve_psd(N, T_A, dtype)


@pytest.mark.parametrize(
    "dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128)
)
@pytest.mark.parametrize("T_A", T_A_list)
@pytest.mark.parametrize("N", N_list)
def test_cusolver_solve_non_psd_dev_1(N, T_A, dtype):
    cusolver_solve_non_psd(N, T_A, dtype)


@pytest.mark.parametrize(
    "dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128)
)
@pytest.mark.parametrize("T_A", T_A_list)
@pytest.mark.parametrize("N", N_list)
def test_cusolver_solve_non_symm_dev_1(N, T_A, dtype):
    cusolver_solve_non_symm(N, T_A, dtype)
