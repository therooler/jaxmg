# run_test_with_devices.py
import os

ndev = 4
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={ndev}"
import jax
from functools import partial

# Setup JAX before anything else imports it
jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

jnp.set_printoptions(linewidth=300)
from jax.sharding import PartitionSpec as P, NamedSharding


import numpy as np
import matplotlib.pyplot as plt
from src.jaxmg.block_cyclic import block_cyclic_relayout, manual_block_cyclic_layout
from src.jaxmg.block_cyclic import calculate_padding, calculate_valid_T_A

def visualize_sharded_matrix(sharded_array, T_A):
    """
    Visualizes a sharded JAX array highlighting shards and tiles.
    Each shard gets a distinct color; tile boundaries are shown as grid lines.
    """
    # Collect shards into a single full matrix and record shard ownership per column
    shards = [np.array(s.data) for s in sharded_array.addressable_shards]
    ndev = len(shards)
    full_rows = shards[0].shape[0]
    full_cols = sum(s.shape[1] for s in shards)

    # Build a "mask" where each column's value = shard index + 1
    mask = np.zeros((full_rows, full_cols), dtype=int)
    col_offset = 0
    for shard_idx, shard in enumerate(shards):
        mask[:, col_offset : col_offset + shard.shape[1]] = shard_idx + 1
        col_offset += shard.shape[1]

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab20", ndev)
    im = ax.imshow(mask, cmap=cmap, vmin=0.5, vmax=ndev + 0.5)

    # Draw vertical tile boundaries
    for col in range(0, full_cols, T_A):
        ax.axvline(col - 0.5, color="black", linewidth=0.5)
    # Draw horizontal tile boundaries (optional, for square tiles)
    for row in range(0, full_rows, T_A):
        ax.axhline(row - 0.5, color="black", linewidth=0.5)

    ax.set_title("Sharded Matrix Layout\nColors = Shards, Lines = Tiles")
    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows")

    cbar = fig.colorbar(im, ticks=range(1, ndev + 1))
    cbar.set_label("Shard Index")
    plt.savefig("mat.pdf")
    plt.show()


if __name__ == "__main__":
    T_A = 128
    ndev = 8
    for N in range(ndev,10000, ndev):
        shard_size = N // ndev
        padding = calculate_padding(shard_size, T_A, ndev)
        if (ndev - 1) * padding > shard_size:
            new_T_A = calculate_valid_T_A(shard_size, T_A, ndev)
        if new_T_A==0:
            print(N, new_T_A)

    # N = 4
    # T_A = 36
    # for i in range(10):
    #     print(N, T_A)
    #     N+=i*4

    #     A = jnp.diag(jnp.arange(N) + 1).astype(jnp.float64)
    #     mesh = jax.make_mesh((ndev,), ("x",))
    #     A = jax.device_put(A, NamedSharding(mesh, P(None, "x")))
    #     # for i, shard in enumerate(A.addressable_shards):
    #     #     print(f"Shard A {i} on device {shard.device}:")
    #     #     print(shard.data)
    #     # print("A_bc:\n", A)
    #     A_bc = block_cyclic_relayout(A, T_A)
    #     # for i, shard in enumerate(A_bc.addressable_shards):
    #     #     print(f"Shard A {i} on device {shard.device}:")
    #     #     print(shard.data)
    #     # print("A_bc:\n", A_bc)
    #     # visualize_sharded_matrix(A_bc, T_A)
    #     A_bc_manual = manual_block_cyclic_layout(A, T_A, ndev)

    #     # for row_i, row_j in zip(A_bc, A_bc_manual):
    #         # print(row_i[:24].astype(int))
    #         # print(row_j[:24].astype(int))
    #         # if not jnp.allclose(row_i[: len(row_j)], row_j):
    #         #     print(row_i)
    #         #     print(row_j)
    #         #     print("DIFF")
    #         #     shard_size = N // ndev
    #         #     target = T_A * ndev
    #         #     padding = (target - ((shard_size) % target)) % target
    #         #     print(shard_size)
    #         #     print(target)
    #         #     print(ndev * padding)
    #             # out = "\n".join(
    #             #     " ".join(f"{val:02d}" for val in row) for row in A_bc.astype(int)
    #             # )
    #             # print(out)

    #             # exit()
    #     assert jnp.allclose(A_bc, A_bc_manual)
