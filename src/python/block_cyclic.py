import jax
import jax.numpy as jnp

from functools import partial
from .utils import get_mesh_and_spec_from_array


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
        # print(x_block)
        # We have to cut off cols 1:padding in the last GPU
        cond_last_gpu = dev_nr == ndev - 1
        x_block = jax.lax.cond(
            cond_last_gpu,
            lambda x: concat_chunk_gpu_last(x, x_left_chunk, padding),
            lambda x: concat_chunk_gpu_other(x, x_left_chunk),
            x_block,
        )
        # print(x_block)
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
        shard_size_padded_necessary = shard_size + (T_A - (shard_size % T_A))%T_A
        return x_block[:, :shard_size_padded_necessary]


def get_chunk_gpu_first(x_block, padding, axis_name):
    N = x_block.shape[0]
    # Create zero chunk, use pvary to add sharding axis
    chunk = jnp.zeros((N, padding), dtype=x_block.dtype)
    return jax.lax.pvary(chunk, axis_name=axis_name)


def get_chunk_gpu_other(x_block, padding):
    # Get the first 1:padding columns
    return x_block[:, :padding]


def concat_chunk_gpu_other(x_block, x_left_chunk):
    # Add columns from next gpu to the right
    return jnp.concatenate([x_block, x_left_chunk], axis=1)


def concat_chunk_gpu_last(x_block, x_left_chunk, padding):
    # Chop of first columns and add padding
    return jnp.concatenate([x_block[:, padding:], x_left_chunk, x_left_chunk], axis=1)


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
    return jnp.array(interleaved)


import jax.numpy as jnp

def manual_block_cyclic_layout(A, T_A, ndev):
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
        
        shard = jnp.concatenate(cols, axis=1) if cols else jnp.zeros((A.shape[0], 0), dtype=A.dtype)
        
        remainder = shard.shape[1] % T_A
        if remainder != 0:
            pad_width = T_A - remainder
            shard = jnp.pad(
                shard, ((0, 0), (0, pad_width)), mode="constant", constant_values=0
            )
        
        shards.append(shard)

    return jnp.concatenate(shards, axis=1)