import sys
import os

this_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(this_dir, "..")
print(src_path)
sys.path.append(src_path)
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding
import pytest
from src.python.block_cyclic import (
    block_cyclic_relayout,
    manual_block_cyclic_layout,
)


def test_device_count():
    devices = jax.devices()
    print(f"[test_device_count] Found {len(devices)} JAX devices.")
    assert len(devices) >= 1


devices = jax.devices()
ndev = len(devices)
if ndev == 1:
    print("Skipping block_cyclic test with 1 device.")
elif ndev == 2:
    print("Running block_cyclic test with 2 devices.")
    mesh = jax.make_mesh((ndev,), ("x",))

    @pytest.mark.parametrize("T_A", (1,2,3))
    @pytest.mark.parametrize("N", (4,8,10, 12))
    def test_block_cyclic_sharding(N, T_A):
        A = jnp.diag(jnp.arange(N, dtype=jnp.float64) + 1)
        A_bc_correct = manual_block_cyclic_layout(A, T_A, ndev)
        # Make mesh and place data
        A = jax.device_put(A, NamedSharding(mesh, P(None, "x")))
        A_bc = block_cyclic_relayout(A, T_A)
        assert jnp.allclose(A_bc[:,:A_bc_correct.shape[1]], A_bc_correct)

elif ndev == 4:
    print("Running block_cyclic test with 4 devices.")
    pass
else:
    raise ValueError("This test requires exactly 1, 2, or 4 devices.")
