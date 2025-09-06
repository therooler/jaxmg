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
from jaxmg import potrf


if any("gpu" == d.platform for d in jax.devices()):
    print("Running on GPU")
    jax.config.update("jax_platforms", "gpu")

    def cusolver_solve(N, T_A, dtype):
        A = jnp.diag(jnp.arange(N, dtype=dtype) + 1)
        b = jnp.ones((N, 1), dtype=dtype)
        ndev = len(devices)
        # Make mesh and place data
        mesh = jax.make_mesh((ndev,), ("x",))
        A = jax.device_put(A, NamedSharding(mesh, P(None, "x")))
        b = jax.device_put(b, NamedSharding(mesh, P(None, None)))

        # Reconstruct from getrf
        out = potrf(A, b, T_A=T_A)
        expected_out = 1.0 / (jnp.arange(N, dtype=dtype) + 1)
        # print(f"Output: {out}")
        # print(f"Expected: {expected_out}")
        assert jnp.allclose(out.flatten(), expected_out)

    devices = jax.devices()
    ndev = len(devices)
    mesh = jax.make_mesh((ndev,), ("x",))

    if ndev == 1:
        print("Running block_cyclic test with 1 device.")
        @pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
        @pytest.mark.parametrize("T_A", (1, 2, 3))
        @pytest.mark.parametrize("N", (4, 8, 10, 12))
        def test_cusolver_solve_dev_1(N, T_A, dtype):
            cusolver_solve(N, T_A, dtype)

    elif ndev == 2:

        @pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
        @pytest.mark.parametrize("T_A", (1, 2, 3))
        @pytest.mark.parametrize("N", (8, 10, 12))
        def test_cusolver_solve_dev_1(N, T_A, dtype):
            cusolver_solve(N, T_A, dtype)

    else:
        print("Test only works for 1 and 2 GPUs")
        assert True


else:

    def test_skip_gpu():
        print("Skipping GPU tests, no GPU available.")
        assert True
