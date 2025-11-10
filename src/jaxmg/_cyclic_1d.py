import numpy as np

import jax
import jax.numpy as jnp
from jax import Array
from typing import Tuple

from functools import partial

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

from .utils import get_mesh_and_spec_from_array


def cyclic_1d_no_shardmap(x_block: Array, T_A: int, ndev: int, axis_name: str) -> Array:
    """Convert a per-device column-sharded block into a 1D block-cyclic layout.

    Forward steps (conceptual):
      1) Compute padding so (shard_size + padding) is a multiple of ndev * T_A.
      2) Shift leftmost chunks across devices with ppermute (i -> i-1) to ensure unique tile ownership.
      3) Reshape/transpose into (dev, blocks, tile), then all_to_all across ``axis_name``.
      4) Drop the extra padding so width == ceil(shard_size, T_A).

    Args:
        x_block: Array
            Local columns on this device with shape ``(N, shard_size)``.
        T_A: int
            Tile width.
        ndev: int
            Number of devices along ``axis_name``.
        axis_name: str
            Mesh axis name.

    Returns:
        Array
            Local block-cyclic shard with shape ``(N, need)`` where ``need`` is
            ``shard_size`` rounded up to a multiple of ``T_A``.
    """
    N, shard_size = x_block.shape  # input is (N, N // ndev), columns

    # If the tile size is larger than the shard the matrix is already in 1D block cyclic form
    if T_A >= shard_size:
        return x_block

    if shard_size < ndev:
        raise ValueError(
            f"We require shard_size >= ndev, but received shard_size = {shard_size} with {ndev} devices."
        )
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


def undo_cyclic_1d_no_shardmap(x_block_bc: Array, T_A: int, ndev: int, axis_name: str) -> Array:
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


def validate_padding(padding: int, ndev: int, shard_size: int, T_A: int) -> None:
    if (ndev - 1) * padding > shard_size:
        N = ndev * shard_size
        new_T_A_min, new_T_A_max = calculate_valid_T_A(
            shard_size, T_A, ndev, T_A_max=shard_size
        )
        suggested_padding_str_max = f"Smallest {T_A} < T_A <= shard_size that would result in ndev * padding <= shard_size: T_A = {new_T_A_max}."
        if new_T_A_min > 0:
            suggested_padding_str = f"Largest 0 < T_A < {T_A} that would result in ndev * padding <= shard_size: T_A = {new_T_A_min}."
            suggested_padding_str = (
                suggested_padding_str + "\n" + suggested_padding_str_max
            )
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


def calculate_padding(shard_size: int, T_A: int, ndev: int) -> int:
    """Return number of padding columns required so ``shard_size + padding`` is a multiple of ``T_A * ndev``."""
    target = T_A * ndev
    padding = (-shard_size) % target
    return padding


def calculate_valid_T_A(shard_size: int, T_A: int, ndev: int, T_A_max: int) -> Tuple[int, int]:
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

def calculate_all_valid_T_A(shard_size: int, ndev: int, T_A_max: int) -> list:
    """Return a list of valid ``T_A`` values for the given shard and device count."""
    suggested_T_A = []
    new_T_A_min = 1
    while new_T_A_min < T_A_max:
        suggested_padding = calculate_padding(shard_size, new_T_A_min, ndev)
        if (ndev - 1) * suggested_padding <= shard_size:
            suggested_T_A.append(new_T_A_min)
        new_T_A_min += 1

    return suggested_T_A

def get_chunk_gpu_zero(x_block: Array, padding: int, axis_name: str) -> Array:
    N = x_block.shape[0]
    # Create zero chunk, use pvary to add sharding axis
    chunk = jnp.zeros((N, padding), dtype=x_block.dtype)
    return jax.lax.pvary(chunk, axis_name=axis_name)


def get_chunk_gpu_left(x_block: Array, padding: int, axis_name: str) -> Array:
    # Get the first 1:padding columns
    N, shard_size = x_block.shape
    if padding > shard_size:
        # Create zero chunk, use pvary to add sharding axis
        chunk = jnp.zeros((N, padding - shard_size), dtype=x_block.dtype)
        chunk_p = jax.lax.pvary(chunk, axis_name=axis_name)
        return jnp.concatenate([x_block, chunk_p], axis=1)
    else:
        return x_block[:, :padding]


def get_chunk_gpu_right(x_block: Array, padding: int, axis_name: str) -> Array:
    # Get the first 1:padding columns
    N, shard_size = x_block.shape
    if padding > shard_size:
        # Create zero chunk, use pvary to add sharding axis
        chunk = jnp.zeros((N, padding - shard_size), dtype=x_block.dtype)
        chunk_p = jax.lax.pvary(chunk, axis_name=axis_name)
        return jnp.concatenate([chunk_p, x_block], axis=1)
    else:
        return x_block[:, -padding:]


def concat_chunk_gpus_left(x_block: Array, x_left_chunk: Array, padding: int, dev: int) -> Array:
    # Add columns from next gpu to the right
    x_block = jnp.concatenate(
        [x_block[:, dev * padding :], x_left_chunk[:, : (dev + 1) * padding]], axis=1
    )
    return x_block[:, : x_block.shape[1] + padding]


def concat_chunk_gpus_right(
    x_block: Array, x_right_chunk: Array, padding: int, dev: int
) -> Array:
    # Add columns from next gpu to the right
    x_block = jnp.concatenate(
        [
            x_right_chunk[:, x_right_chunk.shape[1] - (dev * padding) :],
            x_block[:, : x_block.shape[1] - (dev * padding)],
        ],
        axis=1,
    )
    return x_block[:, : x_block.shape[1] + padding]


def cyclic_1d_layout(a: Array, T_A: int) -> Array:
    """Perform block cyclic relayout"""
    mesh_a, spec_a = get_mesh_and_spec_from_array(a)
    return jax.shard_map(
        partial(
            cyclic_1d_no_shardmap,
            T_A=T_A,
            ndev=len(mesh_a.devices),
            axis_name=spec_a._partitions[1],
        ),
        mesh=mesh_a,
        in_specs=spec_a,
        out_specs=spec_a,
    )(a)


def undo_cyclic_1d_layout(a: Array, T_A: int) -> Array:
    """Undo block cyclic relayout"""
    mesh_a, spec_a = get_mesh_and_spec_from_array(a)
    return jax.shard_map(
        partial(
            undo_cyclic_1d_no_shardmap,
            T_A=T_A,
            ndev=len(mesh_a.devices),
            axis_name=spec_a._partitions[1],
        ),
        mesh=mesh_a,
        in_specs=spec_a,
        out_specs=spec_a,
    )(a)


def manual_cyclic_1d_layout(a: Array, T_A: int, ndev: int) -> Array:
    N, M = a.shape  # input is (N, N // ndev), columns
    shard_size = M // ndev
    if shard_size < ndev:
        raise ValueError(
            f"We require shard_size >= ndev, but received shard_size = {shard_size} with {ndev} devices."
        )
    # If the tile size is larger than the shard the matrix is already in 1D block cyclic form
    if T_A >= shard_size:
        return a
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
            tile = a[:, tile_start:tile_end]
            shards[dev] = (
                shards[dev]
                .at[:, i[dev] * T_A : i[dev] * T_A + (tile_end - tile_start)]
                .set(tile)
            )
            mod_dev += 1
            i[dev] += 1
        return jnp.concatenate([shards[dev] for dev in range(ndev)], axis=1)


def plot_block_to_cyclic(N: int, T_A: int, ndev: int, N_rows: int = 8) -> Tuple[plt.Figure, np.ndarray]:
    """Visualize global column ownership (by device id) before and after converting
    column-block sharding to 1D block-cyclic with tile size ``T_A`` across ``ndev`` devices.

    - Before (axs[0]): contiguous blocks of size ``shard_size`` per device.
    - After  (axs[1]): tiles of width ``T_A`` assigned round-robin to devices.
      Any right-side padding added to make the total width a multiple of ``T_A``
      is shown in light gray.

    Args:
        N: int
            Global matrix dimension (number of rows/columns in the square matrix).
        T_A: int
            Tile width for the cyclic layout.
        ndev: int
            Number of devices.
        N_rows: int, optional
            Number of matrix rows to draw (purely visual; content is repetitive). Defaults to 8.

    Returns:
        fig: plt.Figure
            The matplotlib Figure containing the two subplots.
        axs: np.ndarray
            Array of Axes objects where ``axs[0]`` is the "before" plot and ``axs[1]`` the "after" plot.
    """
    shard_size = N // ndev
    if shard_size < 1 or T_A < 1 or ndev < 1:
        raise ValueError("shard_size, T_A, and ndev must be positive integers.")

    total_cols = ndev * shard_size
    # Before: device d owns columns [d*shard_size:(d+1)*shard_size)

    # After: 1D block-cyclic by tiles; pad to multiple of T_A

    pad = calculate_padding(shard_size, T_A, ndev)
    validate_padding(pad, ndev, shard_size, T_A)
    total_cols_padded = total_cols + pad * ndev
    before = np.ones((N_rows, total_cols_padded), dtype=int) * ndev
    # before = np.ones((N_rows, total_cols), dtype=int)

    for d in range(ndev):
        before[:, d * (shard_size + pad) : d * (shard_size + pad) + shard_size] = d

    after = np.full(
        (N_rows, total_cols_padded), fill_value=ndev, dtype=int
    )  # 'ndev' = padding label

    # Assign device per T_A tile: tile k → device (k % ndev)
    n_tiles = total_cols_padded // T_A
    for k in range(n_tiles):
        dev = k % ndev
        start = k * T_A
        end = start + T_A
        after[:, start:end] = dev

    # But mark actual padding columns as padding label
    if pad:
        after[:, total_cols:] = ndev  # padding

    # ---- Plotting ----
    # Build a colormap: ndev distinct device colors + 1 gray for padding
    device_colors = plt.cm.tab20.colors  # plenty of distinct colors
    colors = list(device_colors[:ndev]) + [
        (0.85, 0.85, 0.85, 1.0)
    ]  # last entry for padding
    cmap = ListedColormap(colors)
    bounds = list(range(ndev + 2))  # 0..ndev-1 device ids, ndev = padding
    norm = BoundaryNorm(bounds, cmap.N)

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)

    im0 = axs[0].imshow(
        before, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm
    )
    axs[0].set_title(
        f"Column-block sharded (shard_size={shard_size}, required padding={pad})"
    )
    axs[0].set_xlabel("Columns")
    axs[0].set_ylabel("Rows")

    # Grid lines for device boundaries in the block layout
    for d in range(1, ndev):
        axs[0].axvline(
            d * (shard_size + pad) - 0.5, lw=1.2, ls="--", color="k", alpha=0.8
        )

    # ---- Shared x-axis ticks every T_A ----
    max_cols = max(total_cols, total_cols_padded)
    xticks = np.arange(0, max_cols + 1, T_A)
    nticks = max_cols + 1
    renderer = fig.canvas.get_renderer()
    bbox = axs[0].get_window_extent(renderer=renderer)
    x_axis_len = bbox.width
    nticks = int(np.ceil(max_cols / T_A))
    # rough available space per tick
    space_per_tick = x_axis_len / nticks
    # heuristic: scale fontsize to ~70% of available space
    size = max(5, min(int(space_per_tick / 2), 12))
    for ax in axs:
        ax.set_xticks(xticks - 0.5)
        ax.set_xticklabels(xticks, fontsize=size)
        ax.set_yticks([])

    im1 = axs[1].imshow(
        after, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm
    )
    axs[1].set_title(f"1D block-cyclic (tile size = {T_A})")
    axs[1].set_xlabel(r"Columns")
    axs[1].set_ylabel(r"Rows")

    # Grid lines for tile boundaries
    for t in range(1, n_tiles):
        axs[1].axvline(t * T_A - 0.5, lw=0.8, ls=":", alpha=0.5)
    # Vertical line separating real data from padding (if any)
    if pad:
        axs[1].axvline(total_cols - 0.5, lw=1.2, ls="--", color="k", alpha=0.8)

    # Legend: one entry per device + padding
    legend_handles = [
        Patch(facecolor=colors[d], edgecolor="k", label=f"GPU {d}") for d in range(ndev)
    ]
    legend_handles.append(Patch(facecolor=colors[-1], edgecolor="k", label="padding"))
    axs[1].legend(handles=legend_handles, loc="upper right", frameon=True)
    fig.suptitle(f"N={N}, ndev={ndev}\n")

    return fig, axs
