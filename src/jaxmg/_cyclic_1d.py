import numpy as np
import os
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax import Array
from typing import Tuple, List

from functools import partial

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

from .utils import get_mesh_and_spec_from_array

import ctypes

import importlib
import pathlib

SHARED_LIBRARY_CYCLIC = os.path.join(os.path.dirname(__file__), "bin/libcyclic.so")
library_cyclic = ctypes.cdll.LoadLibrary(SHARED_LIBRARY_CYCLIC)

# Register FFI targets
jax.ffi.register_ffi_target(
    "cyclic_mg", jax.ffi.pycapsule(library_cyclic.CyclicMgFFI), platform="CUDA"
)


def cyclic_1d(a: Array, T_A: int, mesh: Mesh, in_specs: Tuple[P] | List[P], pad=True):

    ndev = jax.local_device_count()
    # Normalize in_specs so it's a single PartitionSpec instance (not an iterable)
    if isinstance(in_specs, (list, tuple)):
        if len(in_specs) != 1:
            raise ValueError(
                "in_specs must be a single PartitionSpec or a 1-element list/tuple."
            )
        in_specs = in_specs[0]
    if not isinstance(in_specs, P):
        raise TypeError(
            "in_specs must be a PartitionSpec or a 1-element list/tuple containing one."
        )
    if (in_specs._partitions[1] != None) or (in_specs._partitions[0] == None):
        raise ValueError(
            "A must be sharded along the rows with PartitionSpec P(str, None)."
        )
    assert a.ndim == 2, "a must be a 2D array."
    N_rows, N = a.shape
    ndev = jax.device_count()
    shard_size = N_rows // ndev

    input_layouts = ((0, 1),)
    output_layouts = ((0, 1),)

    # Calculate padding
    padding = calculate_padding(shard_size, T_A)
    out_type = (jax.ShapeDtypeStruct((shard_size + padding, N), a.dtype),)

    # Prepare ffi call
    ffi_fn = partial(
        jax.ffi.ffi_call(
            "cyclic_mg",
            out_type,
            input_layouts=input_layouts,
            output_layouts=output_layouts,
            input_output_aliases={0: 0},  # Crucial for buffer sharing
        ),
        T_A=T_A,
    )

    # Jit with donate_argnums=0 is crucial for buffer sharing
    @partial(jax.jit, donate_argnums=0)
    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=P("x", None),
        out_specs=P("x", None),
        check_vma=True,
    )
    def impl(_a):
        (_a,) = ffi_fn(_a)
        return _a

    if not pad or padding == 0:
        assert (
            N_rows != N + ndev * padding
        ), f"pad=False, but with T_A={T_A}, we need padding of {padding} rows per device."
        f"Expected {N + ndev * padding} rows, but received {N_rows}"

        def fn(_a):
            return impl(_a)

    else:
        # Make padding fns
        pad_fn = jax.shard_map(
            partial(pad_rows, padding=padding),
            mesh=mesh,
            in_specs=P("x", None),
            out_specs=P("x", None),
            check_vma=True,
        )
        unpad_fn = jax.shard_map(
            partial(unpad_rows, padding=padding),
            mesh=mesh,
            in_specs=P("x", None),
            out_specs=P("x", None),
            check_vma=True,
        )

        def fn(_a):
            _a = pad_fn(_a)
            _a = impl(_a)
            return unpad_fn(_a)

    return fn(a)


def pad_rows(_a: Array, padding: int):
    _a = jax.lax.pad(_a, 0.0, ((0, padding, 0), (0, 0, 0)))
    return _a


def unpad_rows(_a: Array, padding: int):
    _a = _a[:-padding, :]
    return _a


def calculate_padding(shard_size: int, T_A: int) -> int:
    """Return number of padding columns required so ``shard_size + padding`` is a multiple of ``T_A * ndev``."""
    return (-shard_size) % T_A


def get_cols_cyclic(N, N_batch, T_A, num_devices):
    col_list = []
    shard_size = N // num_devices
    dst_cols = [0] * num_devices
    dst_dev = -1
    offset = N_batch - N // num_devices
    for col in range(N):

        if col % T_A == 0:
            dst_dev = (dst_dev + 1) % num_devices
        num_offsets = col // shard_size

        global_col_src = col + offset * num_offsets
        global_col_dst = dst_cols[dst_dev] + dst_dev * N_batch
        col_list.append((col, global_col_src, global_col_dst))
        # print(
        #     f"src={global_col_src}, dst={global_col_dst}"
        # )
        dst_cols[dst_dev] += 1
    return col_list


def verify_cyclic(A, A_cyclic, T_A):
    ndev = jax.device_count()
    N, N = A.shape[0]
    shard_size = N // ndev
    padding = calculate_padding(shard_size, T_A, ndev=ndev)
    N_batch = shard_size + padding
    col_list = get_cols_cyclic(N, N_batch, T_A, ndev)
    A_array = np.array(A)
    A_cyclic_array = np.array(A_cyclic)
    for col in col_list:
        number, _, dst = col
        assert np.allclose(A_array[number, :], A_cyclic_array[dst, :])


def plot_block_to_cyclic(
    N: int, T_A: int, ndev: int, N_rows: int = 8
) -> Tuple[plt.Figure, np.ndarray]:
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
