import sys
import os
import pytest
import io

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
N_list = list(i * ndev for i in [1, 3, 4, 10])
T_A_list = [1, 2, 3, 5]
mesh = jax.make_mesh((ndev,), ("x",))


def cusolver_solve_arange(N, T_A, dtype):
    A = jnp.diag(jnp.arange(N, dtype=dtype) + 1)
    b = jnp.ones((N, 1), dtype=dtype)
    # Make mesh and place data
    A = jax.device_put(A, NamedSharding(mesh, P("x", None)))
    b = jax.device_put(b, NamedSharding(mesh, P(None, None)))

    out = jax.jit(
        partial(potrs, mesh=mesh, in_specs=(P("x", None), P(None, None)), pad=True),
        static_argnums=2,
    )(A, b, T_A)
    expected_out = 1.0 / (jnp.arange(N, dtype=dtype) + 1)
    assert jnp.allclose(out.flatten(), expected_out)


def cusolver_solve_psd(N, T_A, dtype):
    A = random_psd(N, dtype=dtype, seed=1234)
    b = jnp.ones((N, 1), dtype=dtype)
    cfac = jax.scipy.linalg.cho_factor(A)
    expected_out = jax.scipy.linalg.cho_solve(cfac, b)
    # Make mesh and place data
    _A = jax.device_put(A, NamedSharding(mesh, P("x", None)))
    _b = jax.device_put(b, NamedSharding(mesh, P(None, None)))

    out = jax.jit(
        partial(potrs, mesh=mesh, in_specs=(P("x", None), P(None, None)), pad=True),
        static_argnums=2,
    )(_A, _b, T_A)
    norm_scipy = jnp.linalg.norm(b - A @ expected_out)
    norm_potrf = jnp.linalg.norm(b - A @ out)
    assert jnp.isclose(norm_scipy, norm_potrf, rtol=10, atol=0.0)


def cusolver_solve_non_psd(N, T_A, dtype):
    A = jnp.diag(jnp.arange(N, dtype=dtype) - 1)
    b = jnp.ones((N, 1), dtype=dtype)
    # Make mesh and place data
    A = jax.device_put(A, NamedSharding(mesh, P("x", None)))
    b = jax.device_put(b, NamedSharding(mesh, P(None, None)))

    out, status = jax.jit(
        partial(
            potrs,
            mesh=mesh,
            in_specs=(P("x", None), P(None, None)),
            return_status=True,
            pad=True,
        ),
        static_argnums=2,
    )(A, b, T_A)
    status.block_until_ready()
    out.block_until_ready()
    assert status == 7
    assert jnp.all(jnp.isnan(out))


def cusolver_solve_non_symm(N, T_A, dtype):
    A = jnp.diag(jnp.arange(N, dtype=dtype) + 1)
    # TODO: For some reason the solver does not fail when we set this to 1.0.
    A = A.at[0, 1].set(2.0)
    b = jnp.ones((N, 1), dtype=dtype)
    # Make mesh and place data
    A = jax.device_put(A, NamedSharding(mesh, P("x", None)))
    b = jax.device_put(b, NamedSharding(mesh, P(None, None)))

    out, status = jax.jit(
        partial(
            potrs,
            mesh=mesh,
            in_specs=(P("x", None), P(None, None)),
            return_status=True,
            pad=True,
        ),
        static_argnums=2,
    )(A, b, T_A)
    status.block_until_ready()
    out.block_until_ready()
    assert status == 7
    assert jnp.all(jnp.isnan(out))


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
def test_cusolver_solve_non_psd(N, T_A, dtype):
    cusolver_solve_non_psd(N, T_A, dtype)


@pytest.mark.parametrize(
    "dtype", (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128)
)
@pytest.mark.parametrize("T_A", T_A_list)
@pytest.mark.parametrize("N", N_list)
def test_cusolver_solve_non_symm(N, T_A, dtype):
    cusolver_solve_non_symm(N, T_A, dtype)
