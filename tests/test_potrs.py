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

if any("gpu" == d.platform for d in jax.devices()):
    print("Running on GPU")
    jax.config.update("jax_platforms", "gpu")

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

    def test_potrs_assert_shape_rows():
        N = 4
        T_A = 1
        dtype = jnp.float32
        devices = jax.devices()
        mesh = jax.make_mesh((len(devices),), ("x",))
        A = jnp.eye(N, dtype=dtype)
        b = jnp.ones((N+1, 1), dtype=dtype)
        A = jax.device_put(A, NamedSharding(mesh, P(None, "x")))
        b = jax.device_put(b, NamedSharding(mesh, P(None, None)))
        with pytest.raises(AssertionError, match="A and b must have the same number of rows"):
            potrs(A, b, T_A, mesh=mesh, in_specs=(P(None, "x"), P(None, None)))

    def test_potrs_assert_a_2d():
        N = 4
        T_A = 1
        dtype = jnp.float32
        devices = jax.devices()
        mesh = jax.make_mesh((len(devices),), ("x",))
        A = jnp.ones((N,), dtype=dtype)  # Not 2D
        b = jnp.ones((N, 1), dtype=dtype)
        A = jax.device_put(A, NamedSharding(mesh, P("x",)))
        b = jax.device_put(b, NamedSharding(mesh, P(None, None)))
        with pytest.raises(AssertionError, match="a must be a 2D array"):
            potrs(A, b, T_A, mesh=mesh, in_specs=(P("x",), P(None, None)))

    def test_potrs_assert_b_2d():
        N = 4
        T_A = 1
        dtype = jnp.float32
        devices = jax.devices()
        mesh = jax.make_mesh((len(devices),), ("x",))
        A = jnp.eye(N, dtype=dtype)
        b = jnp.ones((N,), dtype=dtype)  # Not 2D
        A = jax.device_put(A, NamedSharding(mesh, P(None, "x")))
        b = jax.device_put(b, NamedSharding(mesh, P(None,)))
        with pytest.raises(AssertionError, match="b must be a 2D array"):
            potrs(A, b, T_A, mesh=mesh, in_specs=(P(None, "x"), P(None,)))

    def test_potrs_assert_in_specs_len():
        N = 4
        T_A = 1
        dtype = jnp.float32
        devices = jax.devices()
        mesh = jax.make_mesh((len(devices),), ("x",))
        A = jnp.eye(N, dtype=dtype)
        b = jnp.ones((N, 1), dtype=dtype)
        A = jax.device_put(A, NamedSharding(mesh, P(None, "x")))
        b = jax.device_put(b, NamedSharding(mesh, P(None, None)))
        with pytest.raises(AssertionError, match="expected two `in_specs`"):
            potrs(A, b, T_A, mesh=mesh, in_specs=(P("x", None),))

    def test_potrs_valueerror_spec_a():
        N = 4
        T_A = 1
        dtype = jnp.float32
        devices = jax.devices()
        mesh = jax.make_mesh((len(devices),), ("x",))
        A = jnp.eye(N, dtype=dtype)
        b = jnp.ones((N, 1), dtype=dtype)
        # Wrong sharding: first axis sharded
        A = jax.device_put(A, NamedSharding(mesh, P("x", None)))
        b = jax.device_put(b, NamedSharding(mesh, P(None, None)))
        with pytest.raises(ValueError, match="A must be sharded along the columns"):
            potrs(A, b, T_A, mesh=mesh, in_specs=(P("x", None), P(None, None)))

    def test_potrs_valueerror_spec_b():
        N = 4
        T_A = 1
        dtype = jnp.float32
        devices = jax.devices()
        mesh = jax.make_mesh((len(devices),), ("x",))
        A = jnp.eye(N, dtype=dtype)
        b = jnp.ones((N, 1), dtype=dtype)
        A = jax.device_put(A, NamedSharding(mesh, P(None, "x")))
        # Wrong sharding for b
        b = jax.device_put(b, NamedSharding(mesh, P("x", None)))
        with pytest.raises(ValueError, match="b must be replicated along all shards"):
            potrs(A, b, T_A, mesh=mesh, in_specs=(P(None, "x"), P("x", None)))

else:

    def test_skip_gpu():
        print("Skipping GPU tests, no GPU available.")
        assert True
