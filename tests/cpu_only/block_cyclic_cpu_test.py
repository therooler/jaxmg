import sys
import os

this_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(this_dir, "..")

sys.path.append(src_path)
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding
import pytest
from jaxmg import (
    cyclic_1d_layout,
    manual_cyclic_1d_layout,
    calculate_padding,
    calculate_valid_T_A,
    undo_cyclic_1d_layout
)

devices = jax.devices()
ndev = len(devices)
mesh = jax.make_mesh((ndev,), ("x",))

def test_device_count():
    devices = jax.devices()
    print(f"[test_device_count] Found {len(devices)} JAX devices.")
    assert len(devices) >= 1


def cyclic_1d_sharding(N, T_A):
    A = jnp.diag(jnp.arange(N, dtype=jnp.float64) + 1)
    A_bc_correct = manual_cyclic_1d_layout(A, T_A, ndev)
    # Make mesh and place data
    A = jax.device_put(A, NamedSharding(mesh, P(None, "x")))
    try:
        A_bc = cyclic_1d_layout(A, T_A)
        assert jnp.allclose(A_bc, A_bc_correct)
    except ValueError as exc_info:
        with pytest.raises(
            ValueError, match="Attempting 1d cylic relayout with:"
        ):
            raise (exc_info)
        T_A_min, T_A_max = calculate_valid_T_A(N // ndev, T_A, ndev, N // ndev)
        if T_A_min>0:
            new_T_A = T_A_min
        else:
            new_T_A = T_A_max
        A_bc_correct = manual_cyclic_1d_layout(A, new_T_A, ndev)
        # Make mesh and place data
        A = jax.device_put(A, NamedSharding(mesh, P(None, "x")))
        A_bc = cyclic_1d_layout(A, new_T_A)
        assert jnp.allclose(A_bc, A_bc_correct)
        
def undo_cyclic_1d_sharding(N, T_A):
    A = jnp.diag(jnp.arange(N, dtype=jnp.float64) + 1)
    # A_bc = manual_cyclic_1d_layout(A, T_A, ndev)
    A = jax.device_put(A, NamedSharding(mesh, P(None, "x")))
    try:
        A_bc = cyclic_1d_layout(A, T_A)
        # Make mesh and place data
        A_bc = jax.device_put(A_bc, NamedSharding(mesh, P(None, "x")))
        A_bc_undone = undo_cyclic_1d_layout(A_bc, T_A)
        assert jnp.allclose(A, A_bc_undone)
    except ValueError as exc_info:
        with pytest.raises(
            ValueError, match="Attempting 1d cylic relayout with:"
        ):
            raise (exc_info)


def test_T_A_calculation():
    T_A_MAX = 128
    for N in range(ndev, 10000, ndev):
        shard_size = N // ndev
        padding = calculate_padding(shard_size, T_A_MAX, ndev)
        if (ndev - 1) * padding > shard_size:
            new_T_A = calculate_valid_T_A(shard_size, T_A_MAX, ndev, shard_size)


if ndev == 1:
    print("Running block_cyclic test with 1 device.")

    @pytest.mark.parametrize("T_A", (1, 2, 3))
    @pytest.mark.parametrize("N", (4, 8, 10, 12))
    def test_block_cyclic_sharding_dev_1(N, T_A):
        cyclic_1d_sharding(N, T_A)

    @pytest.mark.parametrize("T_A", (1, 2, 3))
    @pytest.mark.parametrize("N", (4, 8, 10, 12))
    def test_undo_block_cyclic_sharding_dev_1(N, T_A):
        undo_cyclic_1d_sharding(N, T_A)

elif ndev == 2:
    print("Running block_cyclic test with 2 devices.")

    @pytest.mark.parametrize("T_A", (1, 2, ))
    @pytest.mark.parametrize("N", (4, 8, 10, 12))
    def test_block_cyclic_sharding_dev_2(N, T_A):
        cyclic_1d_sharding(N, T_A)

    @pytest.mark.parametrize("T_A", (1, 2, 3))
    @pytest.mark.parametrize("N", (4, 8, 10, 12))
    def test_undo_block_cyclic_sharding_dev_2(N, T_A):
        undo_cyclic_1d_sharding(N, T_A)

elif ndev == 4:
    print("Running block_cyclic test with 4 devices.")

    @pytest.mark.parametrize("T_A", (1, 2, 3))
    @pytest.mark.parametrize("N", (16, 24))
    def test_block_cyclic_sharding_dev_4(N, T_A):
        cyclic_1d_sharding(N, T_A)

    @pytest.mark.parametrize("T_A", (1, 2, 3))
    @pytest.mark.parametrize("N", (16, 24))
    def test_undo_block_cyclic_sharding_dev_4(N, T_A):
        undo_cyclic_1d_sharding(N, T_A)

else:
    raise ValueError("This test requires exactly 1, 2, or 4 devices.")
