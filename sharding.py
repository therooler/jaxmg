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
from jaxmg.cyclic_1d import cyclic_1d_layout, manual_cyclic_1d_layout
from src.jaxmg.cyclic_1d import calculate_padding, calculate_valid_T_A, undo_cyclic_1d_layout, validate_padding

from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

def plot_block_to_cyclic(N: int, T_A: int, ndev: int, N_rows: int = 8):
    """
    Visualize global column ownership (by device id) before and after converting
    column-block sharding to 1D block-cyclic with tile size T_A across ndev devices.

    - Before (axs[0]): contiguous blocks of size `shard_size` per device.
    - After  (axs[1]): tiles of width `T_A` assigned round-robin to devices.
      Any right-side padding added to make the total width a multiple of `T_A`
      is shown in light gray.

    Args:
      shard_size: number of columns per device in the block-sharded layout.
      T_A:        tile width for the cyclic layout.
      ndev:       number of devices.
      N_rows:     number of matrix rows to draw (purely visual; content is repetitive).

    Returns:
      fig, axs: matplotlib Figure and Axes array (axs[0] = before, axs[1] = after).
    """
    shard_size = N // ndev
    if shard_size < 1 or T_A < 1 or ndev < 1:
        raise ValueError("shard_size, T_A, and ndev must be positive integers.")
    
    total_cols = ndev * shard_size
    # Before: device d owns columns [d*shard_size:(d+1)*shard_size)
    before = np.zeros((N_rows, total_cols), dtype=int)
    for d in range(ndev):
        before[:, d * shard_size : (d + 1) * shard_size] = d

    # After: 1D block-cyclic by tiles; pad to multiple of T_A
    
    pad = calculate_padding(shard_size, T_A, ndev)
    validate_padding(pad, ndev, shard_size, T_A)
    total_cols_padded = total_cols + pad * ndev
    after = np.full((N_rows, total_cols_padded), fill_value=ndev, dtype=int)  # 'ndev' = padding label

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
    colors = list(device_colors[:ndev]) + [(0.85, 0.85, 0.85, 1.0)]  # last entry for padding
    cmap = ListedColormap(colors)
    bounds = list(range(ndev + 2))  # 0..ndev-1 device ids, ndev = padding
    norm = BoundaryNorm(bounds, cmap.N)

    fig, axs = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)

    im0 = axs[0].imshow(before, aspect='auto', interpolation='nearest', cmap=cmap, norm=norm)
    axs[0].set_title(f"Before: Column-block sharded\n(ndev={ndev}, shard_size={shard_size}, "
                     f"expanded size={total_cols_padded}={total_cols_padded//(ndev*T_A)}*ndev*T_A)")
    axs[0].set_xlabel("Columns")
    axs[0].set_ylabel("Rows")

    # Grid lines for device boundaries in the block layout
    for d in range(1, ndev):
        axs[0].axvline(d * shard_size - 0.5, lw=1, ls='--', alpha=0.6)

    # ---- Shared x-axis ticks every T_A ----
    max_cols = max(total_cols, total_cols_padded)
    xticks = np.arange(0, max_cols + 1, T_A)
    for ax in axs:
        ax.set_xticks(xticks -0.5)
        ax.set_xticklabels(xticks, fontsize=8)
        ax.set_yticks([])

    im1 = axs[1].imshow(after, aspect='auto', interpolation='nearest', cmap=cmap, norm=norm)
    axs[1].set_title(f"After: 1D block-cyclic (tile={T_A})\n(pad per dev ={pad})")
    axs[1].set_xlabel("Columns")

    # Grid lines for tile boundaries
    for t in range(1, n_tiles):
        axs[1].axvline(t * T_A - 0.5, lw=0.8, ls=':', alpha=0.5)
    # Vertical line separating real data from padding (if any)
    if pad:
        axs[1].axvline(total_cols - 0.5, lw=1.2, ls='--', color='k', alpha=0.8)

    # Legend: one entry per device + padding
    legend_handles = [Patch(facecolor=colors[d], edgecolor='k', label=f"dev {d}") for d in range(ndev)]
    legend_handles.append(Patch(facecolor=colors[-1], edgecolor='k', label="padding"))
    axs[1].legend(handles=legend_handles, loc='upper right', frameon=True)


    return fig, axs

# Example usage:
# fig, axs = plot_block_to_cyclic(shard_size=12, T_A=4, ndev=3, N_rows=6)
# plt.show()



if __name__ == "__main__":
    N = 24 # - 2**12
    print(N)
    NRHS = 1
    T_A = 6
    fig, axs = plot_block_to_cyclic(N=N, T_A=T_A, ndev=4, N_rows=6)
    fig.savefig("mat.pdf")
    plt.show()
    exit()
    dtype = jnp.float64

    chunk_size = N // ndev
    mesh = jax.make_mesh((ndev,), ("x",), )

    _A = jnp.diag(jnp.arange(1, N+1, dtype=dtype))
    print(_A)
    # _A = manual_cyclic_1d_layout(_A, T_A=T_A, ndev=ndev)
    _A_bc = jax.device_put(_A, NamedSharding(mesh, P(None, "x")))
    _A_bc = cyclic_1d_layout(_A_bc, T_A)
    print(_A_bc)
    _A_bc = undo_cyclic_1d_layout(_A_bc, T_A=T_A)
    print(_A_bc)
    for shard in jax.device_put(_A_bc, NamedSharding(mesh, P(None, "x"))).addressable_shards:
        print(shard)
    
    assert jnp.allclose(_A_bc, _A)