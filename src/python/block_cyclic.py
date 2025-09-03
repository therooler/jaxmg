import jax
import jax.numpy as jnp

from functools import partial
from .utils import get_mesh_and_spec_from_array


def _make_block_cyclic(x_block, T_A, ndev, axis_name):
    N, shard_size = x_block.shape  # input is (N, N // ndev), columns
    if shard_size < ndev:
        raise ValueError(
            f"We require shard_size >= ndev, but received shard_size = {shard_size} with {ndev} devices."
        )
    # If the tile size is larger than the shard the matrix is already in 1D block cyclic form
    if T_A >= shard_size:
        return x_block
    # To ensure unique tile ownership per device we need to pad the matrices
    # with zeros and shift the columns over the devices. The target is therefore
    # A matrix that is a multiple of T_A * ndev
    # target = T_A * ndev
    # padding = (target - ((shard_size) % target)) % target
    padding = calculate_padding(shard_size, T_A, ndev)
    if (ndev - 1) * padding > shard_size:
        new_T_A = calculate_valid_T_A(shard_size, T_A, ndev)
        if new_T_A > 0:
            suggested_padding_str = f"Largest T_A < {T_A} that would result in ndev * padding <= shard_size is T_A = {new_T_A}"
        else:
            suggested_padding_str = f"No valid T_A < {T_A} exists, a future release may support this case."
        raise ValueError(
            "Attempting 1d block cylic relayout with:\n"
            f"\t- N = {N}\n"
            f"\t- shard_size = {shard_size}\n"
            f"\t- ndev = {ndev}\n"
            f"\t- T_A = {T_A}\n"
            "In order to use an all-to-all call to remap the matrix, we would need to add zero padding of\n"
            f"\t- padding: {padding}\n"
            f"This would require a shift of the last matrix of (ndev - 1) * padding = {(ndev-1) * padding} cols,"
            f"which is larger than the shard_size {shard_size}\n"
            f"{suggested_padding_str}"
        )
    shard_size_padded = shard_size + padding
    # We can shift the matrices to the right by shifting the leftmost columns
    # to the previous GPU, so cols 1:padding get shifted to gpu(i)->gpu(i-1)
    # We handle the last gpu specifically and pad the back with zeros.
    dev_nr = jax.lax.axis_index(axis_name)
    cond_first_gpu = dev_nr == 0
    # print("x_block")
    # print(x_block)
    # print(ndev * padding)
    # print(target)
    # print(shard_size)
    # print(padding)

    x_left_chunk = jax.lax.cond(
        cond_first_gpu,
        lambda x: get_chunk_gpu_first(x, ndev * padding, axis_name),
        lambda x: get_chunk_gpu_other(x, ndev * padding, axis_name),
        x_block,
    )
    # print("x_left_chunk")
    # print(x_left_chunk)
    # Collective call to exchange chunks
    x_left_chunk = jax.lax.ppermute(
        x=x_left_chunk,
        axis_name=axis_name,
        perm=[(i, (i - 1) % ndev) for i in range(ndev)],
    )
    # print("x_left_chunk ppermute")
    # print(x_left_chunk)
    # print(x_block)
    branches = tuple(
        partial(concat_chunk_gpus, padding=padding, dev=i) for i in range(ndev)
    )
    x_block = jax.lax.switch(dev_nr, branches, x_block, x_left_chunk)
    # print("Out")
    # print(x_block)
    # We now arange the tiles so that the shard looks like
    #   gpu0: (gpu_0_tiles_gpu_0, gpu_0_tiles_gpu_1,...)
    #   gpu1: (gpu_1_tiles_gpu_0, gpu_1_tiles_gpu_1,...)
    #     :
    perm = local_block_cyclic_permutation(shard_size_padded, T_A, ndev)
    if perm:
        x_block = x_block[:, jnp.array(perm)]
    # We can now use a single all-to-all call to redistribute the tiles to
    #   gpu0: (gpu_0_tiles_gpu_0, gpu_0_tiles_gpu_0,...)
    #   gpu1: (gpu_1_tiles_gpu_1, gpu_1_tiles_gpu_1,...)
    #     :
    # print("perm", perm)
    # print(x_block)
    x_block = x_block.reshape(N, ndev, -1)
    x_block = jax.lax.all_to_all(
        axis_name=axis_name, x=x_block, split_axis=1, concat_axis=1, tiled=False
    )
    x_block = x_block.reshape(N, shard_size_padded)
    # We now remove the unneccesary padding that was required for the all-to-all
    shard_size_padded_necessary = shard_size + (T_A - (shard_size % T_A)) % T_A
    # print("before slice\n", x_block)
    return x_block[:, :shard_size_padded_necessary]


def calculate_padding(shard_size, T_A, ndev):
    target = T_A * ndev
    padding = (target - ((shard_size) % target)) % target
    return padding


def calculate_valid_T_A(shard_size, T_A, ndev):
    new_T_A = T_A
    while new_T_A > 0:
        suggested_padding = calculate_padding(shard_size, new_T_A, ndev)
        if (ndev - 1) * suggested_padding <= shard_size:
            break
        new_T_A -= 1
    return new_T_A


def get_chunk_gpu_first(x_block, padding, axis_name):
    N = x_block.shape[0]
    # Create zero chunk, use pvary to add sharding axis
    chunk = jnp.zeros((N, padding), dtype=x_block.dtype)
    return jax.lax.pvary(chunk, axis_name=axis_name)


def get_chunk_gpu_other(x_block, padding, axis_name):
    # Get the first 1:padding columns
    N, shard_size = x_block.shape
    if padding > shard_size:
        # Create zero chunk, use pvary to add sharding axis
        chunk = jnp.zeros((N, padding - shard_size), dtype=x_block.dtype)
        chunk_p = jax.lax.pvary(chunk, axis_name=axis_name)
        return jnp.concatenate([x_block, chunk_p], axis=1)
    else:
        return x_block[:, :padding]


def concat_chunk_gpu_first(x_block, x_left_chunk, padding):
    # Add columns from next gpu to the right
    return jnp.concatenate([x_block, x_left_chunk[:, :padding]], axis=1)


def concat_chunk_gpus(x_block, x_left_chunk, padding: int, dev: int):
    # Add columns from next gpu to the right
    # print("dev:", dev)
    # print("shardsize", x_block.shape[1])
    # print(x_block[:, dev * padding :].shape)
    # print(x_left_chunk[:, : (dev + 1) * padding].shape)
    x_block = jnp.concatenate(
        [x_block[:, dev * padding :], x_left_chunk[:, : (dev + 1) * padding]], axis=1
    )
    return x_block[:, : x_block.shape[1] + padding]


def concat_chunk_gpu_last(x_block, x_left_chunk, padding):
    # Chop of first columns and add padding
    # print("dev:", 3)
    # print(x_block[:, padding:].shape)
    # print(x_left_chunk.shape)
    return jnp.concatenate([x_block[:, padding:], x_left_chunk], axis=1)


def block_cyclic_relayout(a, T_A):
    """Perform block cyclic relayout"""
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
    # Previous padding ensures that both of these are integer
    nblocks = shard_cols // T_A
    nblocks_per_device = nblocks // ndev
    # Build permutation of indices
    interleaved = []
    for dev in range(ndev):
        for block in range(nblocks_per_device):
            shift = ndev * T_A * block + dev * T_A
            for i in range(T_A):
                idx = shift + i
                interleaved.append(idx)
    return interleaved


import jax.numpy as jnp


def manual_block_cyclic_layout(A, T_A, ndev):
    N, M = A.shape  # input is (N, N // ndev), columns
    shard_size = M // ndev
    if shard_size < ndev:
        raise ValueError(
            f"We require shard_size >= ndev, but received shard_size = {shard_size} with {ndev} devices."
        )
    # If the tile size is larger than the shard the matrix is already in 1D block cyclic form
    if T_A >= shard_size:
        return A
    else:
        # To ensure unique tile ownership per device we need to pad the matrices
        # with zeros and shift the columns over the devices. The target is therefore
        # A matrix that is a multiple of T_A * ndev
        target = T_A * ndev
        padding = (target - ((shard_size) % target)) % target
        shard_size_padded_necessary = shard_size + (T_A - (shard_size % T_A)) % T_A
        shards = [jnp.zeros((N, shard_size_padded_necessary))] * ndev
        mod_dev = 0
        i = [0] * ndev
        for tile_start in range(0, N, T_A):
            dev = mod_dev % ndev
            tile_end = min(tile_start + T_A, N)
            tile = A[:, tile_start:tile_end]
            shards[dev] = (
                shards[dev]
                .at[:, i[dev] * T_A : i[dev] * T_A + (tile_end - tile_start)]
                .set(tile)
            )
            mod_dev += 1
            i[dev] += 1
        return jnp.concatenate([shards[dev] for dev in range(ndev)], axis=1)


def manual_block_cyclic_layout_old(A, T_A, ndev):
    N = A.shape[0]
    shard_size = N // ndev
    if T_A >= shard_size:
        return A

    shards = []
    for dev in range(ndev):
        cols = []
        for tile_start in range(0, N, T_A):
            tile_end = min(tile_start + T_A, N)
            tile = A[:, tile_start:tile_end]

            # Pad tile if needed
            if tile.shape[1] < T_A:
                pad_width = T_A - tile.shape[1]
                tile = jnp.pad(
                    tile, ((0, 0), (0, pad_width)), mode="constant", constant_values=0
                )

            tile_idx = tile_start // T_A
            if tile_idx % ndev == dev:
                cols.append(tile)

        shard = (
            jnp.concatenate(cols, axis=1)
            if cols
            else jnp.zeros((A.shape[0], 0), dtype=A.dtype)
        )

        remainder = shard.shape[1] % T_A
        if remainder != 0:
            pad_width = T_A - remainder
            shard = jnp.pad(
                shard, ((0, 0), (0, pad_width)), mode="constant", constant_values=0
            )

        shards.append(shard)

    return jnp.concatenate(shards, axis=1)
