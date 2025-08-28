# run_test_with_devices.py
import os

ndev = 2
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={ndev}"
import jax
from functools import partial

# Setup JAX before anything else imports it
jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding

def get_mesh_and_spec_from_array(A):
    sharding = A.sharding
    if isinstance(sharding, NamedSharding):
        return sharding.mesh, sharding.spec
    else:
        raise ValueError(
            "Array is not sharded with a NamedSharding, cannot extract mesh and spec."
        )


def _make_block_cyclic(x_block, T_A, ndev, axis_name):
    N, shard_size = x_block.shape  # input is (N, N // ndev), columns
    # If the tile size is larger than the shard the matrix is already in 1D block cyclic form
    if T_A >= shard_size:
        return x_block
    else:
        # To ensure unique tile ownership per device we need to pad the matrices
        # with zeros and shift the columns over the devices. The target is therefore
        # A matrix that is a multiple of T_A * ndev
        target = T_A * ndev
        padding = (target - (shard_size % target)) % target
        shard_size_padded = shard_size + padding
        # We can shift the matrices to the right by shifting the leftmost columns
        # to the previous GPU, so cols 1:padding get shifted to gpu(i)->gpu(i-1)
        # We handle the last gpu specifically and pad the back with zeros.
        dev_nr = jax.lax.axis_index(axis_name)
        cond_first_gpu = dev_nr == 0
        x_left_chunk = jax.lax.cond(
            cond_first_gpu,
            lambda x: get_chunk_gpu_first(x, padding, axis_name),
            lambda x: get_chunk_gpu_other(x, padding),
            x_block,
        )
        # Collective call to exchange chunks
        x_left_chunk = jax.lax.ppermute(
            x=x_left_chunk,
            axis_name=axis_name,
            perm=[(i, (i - 1) % ndev) for i in range(ndev)],
        )
        print(x_block)
        # We have to cut off cols 1:padding in the last GPU
        cond_last_gpu = dev_nr == ndev - 1
        x_block = jax.lax.cond(
            cond_last_gpu,
            lambda x: concat_chunk_gpu_last(x, x_left_chunk, padding),
            lambda x: concat_chunk_gpu_other(x, x_left_chunk),
            x_block,
        )
        print(x_block)
        # We now arange the tiles so that the shard looks like 
        #   gpu0: (gpu_0_tiles_gpu_0, gpu_0_tiles_gpu_1,...)
        #   gpu1: (gpu_1_tiles_gpu_0, gpu_1_tiles_gpu_1,...)
        #     :
        perm = local_block_cyclic_permutation(shard_size_padded, T_A, ndev)
        x_block = x_block[:, perm]
        # We can now use a single all-to-all call to redistribute the tiles to
        #   gpu0: (gpu_0_tiles_gpu_0, gpu_0_tiles_gpu_0,...)
        #   gpu1: (gpu_1_tiles_gpu_1, gpu_1_tiles_gpu_1,...)
        #     :
        x_block = x_block.reshape(N, ndev, -1)
        x_block = jax.lax.all_to_all(
            axis_name=axis_name, x=x_block, split_axis=1, concat_axis=1, tiled=False
        )
        x_block = x_block.reshape(N, shard_size_padded)
        # We now remove the unneccesary padding that was required for the all-to-all
        shard_size_padded_necessary = shard_size + (T_A - (shard_size % T_A))
        return x_block[:, :shard_size_padded_necessary]


def get_chunk_gpu_first(x_block, padding, axis_name):
    N = x_block.shape[0]
    chunk = jnp.zeros((N, padding), dtype=x_block.dtype)
    return jax.lax.pvary(chunk, axis_name=axis_name)


def get_chunk_gpu_other(x_block, padding):
    return x_block[:, :padding]


def concat_chunk_gpu_other(x_block, x_left_chunk):
    return jnp.concatenate([x_block, x_left_chunk], axis=1)


def concat_chunk_gpu_last(x_block, x_left_chunk, padding):
    return jnp.concatenate([x_block[:, padding:], x_left_chunk, x_left_chunk], axis=1)


def block_cyclic_relayout(a, T_A):
    mesh_a, spec_a = get_mesh_and_spec_from_array(a)
    return jax.shard_map(
        partial(
            _make_block_cyclic,
            T_A=T_A,
            ndev=len(mesh_a.devices),
            axis_name=spec_a._partitions[1],
        ),
        mesh=mesh_a,
        in_specs=spec_a,
        out_specs=spec_a,
    )(a)


def local_block_cyclic_permutation(shard_cols, T_A, ndev):
    nblocks = shard_cols // T_A
    print(f"shard_cols: {shard_cols}")
    print(f"nblocks: {nblocks}")
    nblocks_per_device = nblocks // ndev
    interleaved = []
    for dev in range(ndev):
        for block in range(nblocks_per_device):
            shift = ndev * T_A * block + dev * T_A
            for i in range(T_A):
                idx = shift + i
                interleaved.append(idx)
    return jnp.array(interleaved)


def manual_block_cyclic_layout(A, T_A, ndev):
    N = A.shape[0]
    shard_size = N//ndev
    shards = []
    for dev in range(ndev):
        cols = []
        ncols = 0
        for tile_start in range(0, N, T_A):
            tile_end = min(tile_start + T_A, N)
            tile = A[:, tile_start:tile_end]

            # Pad if tile is smaller than T_A
            if tile.shape[1] < T_A:
                pad_width = T_A - tile.shape[1]
                tile = jnp.pad(
                    tile, ((0, 0), (0, pad_width)), mode="constant", constant_values=0
                )

            tile_idx = tile_start // T_A
            if tile_idx % ndev == dev:
                cols.append(tile)
                ncols+=1
        if ncols>0:
            shards.append(jnp.concatenate(cols, axis=1))
        if ncols != (shard_size + (T_A - (shard_size % T_A))):
            shards.append(jnp.zeros((N, ncols-shard_size + (T_A - (shard_size % T_A))), dtype=A.dtype))
    return jnp.concatenate(shards, axis=1)

import numpy as np
import matplotlib.pyplot as plt

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
        mask[:, col_offset:col_offset + shard.shape[1]] = shard_idx + 1
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
    plt.savefig('mat.pdf')
    plt.show()


if __name__ == "__main__":
    N = 10
    T_A = 2
    A = jnp.diag(jnp.arange(N) + 1).astype(jnp.float64)
    mesh = jax.make_mesh((ndev,), ("x",))
    A = jax.device_put(A, NamedSharding(mesh, P(None, "x")))
    for i, shard in enumerate(A.addressable_shards):
        print(f"Shard A {i} on device {shard.device}:")
        print(shard.data)
    print("A_bc:\n", A)
    A_bc = block_cyclic_relayout(A, T_A)
    for i, shard in enumerate(A_bc.addressable_shards):
        print(f"Shard A {i} on device {shard.device}:")
        print(shard.data)
    print("A_bc:\n", A_bc)
    # visualize_sharded_matrix(A_bc, T_A)
    A_bc_manual = manual_block_cyclic_layout(A, T_A,ndev)
    print(A_bc_manual)
    assert jnp.allclose(A_bc[:, : A_bc_manual.shape[1]], A_bc_manual)
