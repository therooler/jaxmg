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

if any("gpu" == d.platform for d in jax.devices()):
    print("Running on GPU")
    jax.config.update("jax_platforms", "gpu")

    def cusolver_solve_arange(N, T_A, dtype):
        A = jnp.diag(jnp.arange(N, dtype=dtype) + 1)
        ndev = len(devices)
        # Make mesh and place data
        mesh = jax.make_mesh((ndev,), ("x",))
        A = jax.device_put(A, NamedSharding(mesh, P(None, "x")))

        out = jax.jit(
            partial(potri, mesh=mesh, in_specs=(P(None, "x"),)), static_argnums=1
        )(A, T_A=T_A)
        expected_out = jnp.diag(1.0 / (jnp.arange(N, dtype=dtype) + 1))
        assert jnp.allclose(out, expected_out)

    def cusolver_solve_psd(N, T_A, dtype):
        A = random_psd(N, dtype=dtype, seed=1234)
        expected_out = jnp.linalg.inv(A)
        ndev = len(devices)
        # Make mesh and place data
        mesh = jax.make_mesh((ndev,), ("x",))
        _A = jax.device_put(A, NamedSharding(mesh, P(None, "x")))

        out = jax.jit(
            partial(potri, mesh=mesh, in_specs=(P(None, "x"),)), static_argnums=1
        )(A, T_A=T_A)
        norm_potri = jnp.linalg.norm(A @ out - jnp.eye(N, dtype=dtype))
        norm_lax = jnp.linalg.norm(A @ expected_out - jnp.eye(N, dtype=dtype))
        assert jnp.isclose(norm_potri, norm_lax, rtol=10, atol=0.0)

    devices = jax.devices()
    ndev = len(devices)
    mesh = jax.make_mesh((ndev,), ("x",))

    if ndev == 1:
        print("Running block_cyclic test with 1 device.")

        @pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
        @pytest.mark.parametrize("T_A", (1, 2, 3))
        @pytest.mark.parametrize("N", (4, 8, 10, 12))
        def test_cusolver_solve_arange_dev_1(N, T_A, dtype):
            cusolver_solve_arange(N, T_A, dtype)

        @pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
        @pytest.mark.parametrize("T_A", (1, 2, 3))
        @pytest.mark.parametrize("N", (4, 8, 10, 12))
        def test_cusolver_solve_psd_dev_1(N, T_A, dtype):
            cusolver_solve_psd(N, T_A, dtype)

    elif ndev == 2:

        @pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
        @pytest.mark.parametrize("T_A", (1, 2, 3))
        @pytest.mark.parametrize("N", (8, 10, 12))
        def test_cusolver_solve_arange_dev_2(N, T_A, dtype):
            cusolver_solve_arange(N, T_A, dtype)

        @pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
        @pytest.mark.parametrize("T_A", (1, 2, 3))
        @pytest.mark.parametrize("N", (8, 10, 12))
        def test_cusolver_solve_psd_dev_2(N, T_A, dtype):
            cusolver_solve_psd(N, T_A, dtype)

    elif ndev == 4:

        @pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
        @pytest.mark.parametrize("T_A", (1, 2, 4))
        @pytest.mark.parametrize("N", (48, 60))
        def test_cusolver_solve_arange_dev_4(N, T_A, dtype):
            cusolver_solve_arange(N, T_A, dtype)

        @pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
        @pytest.mark.parametrize("T_A", (1, 2, 4))
        @pytest.mark.parametrize("N", (48, 60))
        def test_cusolver_solve_psd_dev_4(N, T_A, dtype):
            cusolver_solve_psd(N, T_A, dtype)

    else:
        print("Test only works for 1,2 and 4 GPUs")
        assert True


else:

    def test_skip_gpu():
        print("Skipping GPU tests, no GPU available.")
        assert True
