import jax
import jax.numpy as jnp

from functools import partial
from .utils import get_mesh_and_spec_from_array


def _cyclic_1d(x_block, T_A, ndev, axis_name):
    """
    Convert a per-device column-sharded block of shape (N, shard_size) into a
    1D cyclic layout across `ndev` devices on `axis_name` with tileing `T_A`.

    Forward steps (conceptual):
      1) Compute padding so (shard_size + padding) is a multiple of ndev * T_A.
      2) Shift leftmost chunks across devices with ppermute (i -> i-1) to ensure unique tile ownership.
      3) Reshape/transpose into (dev, blocks, tile), then all_to_all across `axis_name`.
      4) Drop the extra padding so width == ceil(shard_size, T_A).

    Args:
      x_block: (N, shard_size) local columns on this device.
      T_A    : tile width.
      ndev   : number of devices along `axis_name`.
      axis_name: mesh axis name.

    Returns:
      x_block_bc: (N, need) local block-cyclic shard, where
                  need = shard_size rounded up to a multiple of T_A.
    """
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
    padding = calculate_padding(shard_size, T_A, ndev)
    # the left-shift we need should not exceed shard_size on the last device.
    validate_padding(padding, ndev, shard_size, T_A)
    # Calculate new padding and padding we actually need
    shard_size_padded = shard_size + padding
    need = shard_size + (T_A - (shard_size % T_A)) % T_A
    # We can shift the matrices to the right by shifting the leftmost columns
    # to the previous GPU, so cols 1:padding get shifted to gpu(i)->gpu(i-1)
    # We handle the last gpu specifically and pad the back with zeros.
    width = ndev * padding
    if width:
        dev_nr = jax.lax.axis_index(axis_name)

        x_left_chunk = jax.lax.cond(
            dev_nr == 0,
            lambda x: get_chunk_gpu_zero(x, width, axis_name),
            lambda x: get_chunk_gpu_left(x, width, axis_name),
            x_block,
        )
        # Circulate: i -> i-1 (mod ndev)
        x_left_chunk = jax.lax.ppermute(
            x=x_left_chunk,
            axis_name=axis_name,
            perm=[(i, (i - 1) % ndev) for i in range(ndev)],
        )
        # On each device, splice the received chunk to the left and trim the tail.
        branches = tuple(
            partial(concat_chunk_gpus_left, padding=padding, dev=i) for i in range(ndev)
        )
        x_block = jax.lax.switch(dev_nr, branches, x_block, x_left_chunk)

    # -------- Tile exchange via all_to_all --------
    # Reshape to (N, blocks, ndev, T_A) → transpose to (N, ndev, blocks, T_A)
    # shard_cols is guaranteed multiple of ndev*T_A by your padding
    blocks = shard_size_padded // (ndev * T_A)
    x = x_block.reshape(N, blocks, ndev, T_A)  # (N, blocks, dev, tile)
    x = jnp.transpose(x, (0, 2, 1, 3))  # (N, dev, blocks, tile)
    x = jnp.reshape(x, (N, ndev, blocks * T_A))  # Collapse (blocks, T_A)

    # Exchange tiles across devices
    x = jax.lax.all_to_all(
        x, axis_name=axis_name, split_axis=1, concat_axis=1, tiled=False
    )  # (N, ndev, blocks*T_A)

    # Back to (N, shard_size_padded)
    x_block = jnp.reshape(x, (N, shard_size_padded))

    return x_block[:, :need]


def _undo_cyclic_1d(x_block_bc, T_A, ndev, axis_name):
    """
    Inverse of _make_block_cyclic:
      x_block_bc: per-device block-cyclic local shard, shape (N, need)
      T_A       : tile size used in forward pass
      ndev      : number of devices on mesh axis `axis_name`
    Returns a per-device column-sharded block of shape (N, shard_size) on each device.
    """

    N, need = x_block_bc.shape
    if T_A >= need:
        return x_block_bc

    target_shard_size = N // ndev
    padding = calculate_padding(target_shard_size, T_A, ndev)
    extra = calculate_padding(need, T_A, ndev)

    if extra:
        x_block_bc = jnp.pad(x_block_bc, ((0, 0), (0, extra)))
        need += extra

    # -------- Inverse of tile exchange --------
    blocks = need // (ndev * T_A)
    x = x_block_bc.reshape(N, ndev, blocks * T_A)
    x = jax.lax.all_to_all(x, axis_name=axis_name, split_axis=1, concat_axis=1)
    x = x.reshape(N, ndev, blocks, T_A)
    x = jnp.transpose(x, (0, 2, 1, 3))  # (N, blocks, ndev, T_A)
    x = x.reshape(N, blocks * ndev * T_A)

    # -------- Inverse of shift --------
    width = ndev * padding
    if width and ndev > 1:
        dev_nr = jax.lax.axis_index(axis_name)

        # take the rightmost width columns (inverse of leftmost in forward)
        x_right_chunk = jax.lax.cond(
            dev_nr == ndev - 1,
            lambda x: get_chunk_gpu_zero(x, width, axis_name),
            lambda x: get_chunk_gpu_right(x, width, axis_name),
            x,
        )
        # inverse perm: i -> i+1
        x_right_chunk = jax.lax.ppermute(
            x_right_chunk,
            axis_name=axis_name,
            perm=[(i, (i + 1) % ndev) for i in range(ndev)],
        )

        # On each device, splice the concat chunk to the right and trim the tail.
        branches = tuple(
            partial(concat_chunk_gpus_right, padding=padding, dev=i)
            for i in range(ndev)
        )
        x = jax.lax.switch(dev_nr, branches, x, x_right_chunk)

    # finally trim back down to the true shard_size
    return x[:, :target_shard_size]


def validate_padding(padding, ndev, shard_size, T_A):
    if (ndev - 1) * padding > shard_size:
        N = ndev * shard_size
        new_T_A_min, new_T_A_max = calculate_valid_T_A(shard_size, T_A, ndev, T_A_max=shard_size)
        suggested_padding_str_max = (
                f"Smallest {T_A} < T_A <= shard_size that would result in ndev * padding <= shard_size: T_A = {new_T_A_max}."
            )
        if new_T_A_min > 0:
            suggested_padding_str = f"Largest 0 < T_A < {T_A} that would result in ndev * padding <= shard_size: T_A = {new_T_A_min}."
            suggested_padding_str = suggested_padding_str + "\n" + suggested_padding_str_max
        else:
            suggested_padding_str = suggested_padding_str_max

        raise ValueError(
            "Attempting 1d cylic relayout with:\n"
            f"\t- N = {N}\n"
            f"\t- shard_size = {shard_size}\n"
            f"\t- ndev = {ndev}\n"
            f"\t- T_A = {T_A}\n"
            "In order to use an all-to-all call to remap the matrix, we would need to add zero padding of\n"
            f"\t- padding: {padding}\n"
            f"This would require a shift of the last matrix of (ndev - 1) * padding = {(ndev-1) * padding} cols,"
            f"which is larger than the shard_size {shard_size}.\n"
            f"{suggested_padding_str}\n"
            f"Use `calculate_valid_T_A` to calculate a valid T_A for 1d cyclic resharding."
        )


def calculate_padding(shard_size, T_A, ndev):
    target = T_A * ndev
    padding = (-shard_size) % target
    return padding


def calculate_valid_T_A(shard_size, T_A, ndev, T_A_max=256):
    new_T_A_min = T_A
    new_T_A_max = T_A
    while new_T_A_min > 0:
        suggested_padding = calculate_padding(shard_size, new_T_A_min, ndev)
        if (ndev - 1) * suggested_padding <= shard_size:
            break
        new_T_A_min -= 1
    while new_T_A_max < T_A_max:
        suggested_padding = calculate_padding(shard_size, new_T_A_max, ndev)
        if (ndev - 1) * suggested_padding <= shard_size:
            break
        new_T_A_max += 1

    return new_T_A_min, new_T_A_max


def get_chunk_gpu_zero(x_block, padding, axis_name):
    N = x_block.shape[0]
    # Create zero chunk, use pvary to add sharding axis
    chunk = jnp.zeros((N, padding), dtype=x_block.dtype)
    return jax.lax.pvary(chunk, axis_name=axis_name)


def get_chunk_gpu_left(x_block, padding, axis_name):
    # Get the first 1:padding columns
    N, shard_size = x_block.shape
    if padding > shard_size:
        # Create zero chunk, use pvary to add sharding axis
        chunk = jnp.zeros((N, padding - shard_size), dtype=x_block.dtype)
        chunk_p = jax.lax.pvary(chunk, axis_name=axis_name)
        return jnp.concatenate([x_block, chunk_p], axis=1)
    else:
        return x_block[:, :padding]


def get_chunk_gpu_right(x_block, padding, axis_name):
    # Get the first 1:padding columns
    N, shard_size = x_block.shape
    if padding > shard_size:
        # Create zero chunk, use pvary to add sharding axis
        chunk = jnp.zeros((N, padding - shard_size), dtype=x_block.dtype)
        chunk_p = jax.lax.pvary(chunk, axis_name=axis_name)
        return jnp.concatenate([chunk_p, x_block], axis=1)
    else:
        return x_block[:, -padding:]


def concat_chunk_gpus_left(x_block, x_left_chunk, padding: int, dev: int):
    # Add columns from next gpu to the right
    x_block = jnp.concatenate(
        [x_block[:, dev * padding :], x_left_chunk[:, : (dev + 1) * padding]], axis=1
    )
    return x_block[:, : x_block.shape[1] + padding]


def concat_chunk_gpus_right(x_block, x_right_chunk, padding: int, dev: int):
    # Add columns from next gpu to the right
    x_block = jnp.concatenate(
        [
            x_right_chunk[:, x_right_chunk.shape[1] - (dev * padding) :],
            x_block[:, : x_block.shape[1] - (dev * padding)],
        ],
        axis=1,
    )
    return x_block[:, : x_block.shape[1] + padding]


def cyclic_1d_layout(a, T_A):
    """Perform block cyclic relayout"""
    mesh_a, spec_a = get_mesh_and_spec_from_array(a)
    return jax.jit(
        jax.shard_map(
            partial(
                _cyclic_1d,
                T_A=T_A,
                ndev=len(mesh_a.devices),
                axis_name=spec_a._partitions[1],
            ),
            mesh=mesh_a,
            in_specs=spec_a,
            out_specs=spec_a,
        ),
    )(a)


def undo_cyclic_1d_layout(a, T_A):
    """Undo block cyclic relayout"""
    mesh_a, spec_a = get_mesh_and_spec_from_array(a)
    return jax.shard_map(
        partial(
            _undo_cyclic_1d,
            T_A=T_A,
            ndev=len(mesh_a.devices),
            axis_name=spec_a._partitions[1],
        ),
        mesh=mesh_a,
        in_specs=spec_a,
        out_specs=spec_a,
    )(a)


def manual_cyclic_1d_layout(A, T_A, ndev):
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
