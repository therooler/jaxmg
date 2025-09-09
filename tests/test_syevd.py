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


if any("gpu" == d.platform for d in jax.devices()):
    print("Running on GPU")
    jax.config.update("jax_platforms", "gpu")

    def cusolver_solve_arange(N, T_A, dtype):
        A = jnp.diag(jnp.arange(N, dtype=dtype) + 1)
        eigenvalues_expected = jnp.diag(A)
        ndev = len(devices)
        # Make mesh and place data
        mesh = jax.make_mesh((ndev,), ("x",))
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
        ndev = len(devices)
        # Make mesh and place data
        mesh = jax.make_mesh((ndev,), ("x",))
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
        ndev = len(devices)
        # Make mesh and place data
        mesh = jax.make_mesh((ndev,), ("x",))
        A = jax.device_put(A, NamedSharding(mesh, P(None, "x")))

        eigenvalues = jax.jit(
            partial(
                syevd, mesh=mesh, in_specs=(P(None, "x"),), return_eigenvectors=False
            ),
            static_argnums=1,
        )(A, T_A=T_A)
        assert jnp.allclose(eigenvalues_expected, eigenvalues, rtol=10, atol=0.0)

    def cusolver_solve_psd_no_V(N, T_A, dtype):
        A = random_psd(N, dtype=dtype, seed=1234)
        eigenvalues_expected = jnp.linalg.eigvalsh(A)
        ndev = len(devices)
        # Make mesh and place data
        mesh = jax.make_mesh((ndev,), ("x",))
        A = jax.device_put(A, NamedSharding(mesh, P(None, "x")))
        eigenvalues = jax.jit(
            partial(
                syevd, mesh=mesh, in_specs=(P(None, "x"),), return_eigenvectors=False
            ),
            static_argnums=1,
        )(A, T_A=T_A)

        assert jnp.allclose(eigenvalues, eigenvalues_expected, rtol=10, atol=0.0)

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

        @pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
        @pytest.mark.parametrize("T_A", (1, 2, 3))
        @pytest.mark.parametrize("N", (4, 8, 10, 12))
        def test_cusolver_solve_arange_no_v_dev_1(N, T_A, dtype):
            cusolver_solve_arange_no_V(N, T_A, dtype)

        @pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
        @pytest.mark.parametrize("T_A", (1, 2, 3))
        @pytest.mark.parametrize("N", (4, 8, 10, 12))
        def test_cusolver_solve_psd_no_v_dev_1(N, T_A, dtype):
            cusolver_solve_psd_no_V(N, T_A, dtype)

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

        @pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
        @pytest.mark.parametrize("T_A", (1, 2, 3))
        @pytest.mark.parametrize("N", (8, 10, 12))
        def test_cusolver_solve_arange_no_v_dev_2(N, T_A, dtype):
            cusolver_solve_arange_no_V(N, T_A, dtype)

        @pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
        @pytest.mark.parametrize("T_A", (1, 2, 3))
        @pytest.mark.parametrize("N", (8, 10, 12))
        def test_cusolver_solve_psd_no_v_dev_2(N, T_A, dtype):
            cusolver_solve_psd_no_V(N, T_A, dtype)

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

        @pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
        @pytest.mark.parametrize("T_A", (1, 2, 4))
        @pytest.mark.parametrize("N", (48, 60))
        def test_cusolver_solve_arange_no_v_dev_4(N, T_A, dtype):
            cusolver_solve_arange_no_V(N, T_A, dtype)

        @pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
        @pytest.mark.parametrize("T_A", (1, 2, 4))
        @pytest.mark.parametrize("N", (48, 60))
        def test_cusolver_solve_arange_no_v_dev_4(N, T_A, dtype):
            cusolver_solve_psd_no_V(N, T_A, dtype)

    else:
        print("Test only works for 1,2 and 4 GPUs")
        assert True

    @pytest.fixture(scope="module")
    def mesh():
        devices = jax.devices("gpu")
        if not devices:
            pytest.skip("No GPU devices available")
        ndev = len(devices)
        return jax.make_mesh((ndev,), ("x",))

    @pytest.mark.parametrize("bad_shape", [(3,), (3, 3, 3)])
    def test_a_not_2d(mesh, bad_shape):
        a = jnp.ones(bad_shape)
        with pytest.raises(AssertionError, match="a must be a 2D array"):
            syevd(a, 32, mesh, (P(None, "x"),))

    def test_in_specs_wrong_length(mesh):
        a = jnp.eye(4)
        with pytest.raises(AssertionError, match="expected only one `in_specs`"):
            syevd(a, 32, mesh, (P(None, "x"), P(None, "x")))

    def test_spec_a_not_sharded_columns(mesh):
        a = jnp.eye(4)
        # PartitionSpec with both axes sharded or first axis sharded
        bad_specs = [P("x", None), P("x", "x"), P(None, None)]
        for spec in bad_specs:
            with pytest.raises(ValueError, match="A must be sharded along the columns"):
                syevd(a, 32, mesh, (spec,))

    def test_T_A_too_large(mesh):
        a = jnp.eye(4)
        with pytest.raises(ValueError, match="T_A has a maximum value of 1024"):
            syevd(a, 2048, mesh, (P(None, "x"),))

    def test_correct_call(mesh):
        N = 4
        a = jnp.eye(N)
        sharding = NamedSharding(mesh, P(None, "x"))
        a = jax.device_put(a, sharding)
        # Should not raise
        out = syevd(a, 32, mesh, (P(None, "x"),))
        assert isinstance(out, tuple)

else:

    def test_skip_gpu():
        print("Skipping GPU tests, no GPU available.")
        assert True
