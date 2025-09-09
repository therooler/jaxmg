import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import subprocess
from pathlib import Path

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".80"

import time
import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from functools import partial
from jax.sharding import PartitionSpec as P, NamedSharding

from jaxmg.potrs import potrs
from jaxmg.cyclic_1d import calculate_valid_T_A, validate_padding, calculate_padding
import re
from pathlib import Path
import matplotlib.pyplot as plt

dtype = jnp.float32
devices = jax.devices("gpu")
ndev = len(devices)

def main(N, T_A):
    
    print(f"Available devices: {ndev}")
    gpu_name = ""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            encoding="utf-8",
        )
        gpu_name = output.strip().split("\n")[0]
    except Exception as e:
        raise ValueError(f"Error querying GPU name: {e}")
    gpu_name = "_".join(gpu_name.split(" "))
    # PARAMETERS
    NRHS = 1
    n_runs = 10
    save_path = f"{Path(__file__).parent}/data_potrs/{gpu_name}/{jnp.dtype(dtype).name}/ndev_{ndev}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = f"N_{N}_T_A_{T_A}"
    if os.path.exists(f"{save_path}{file_name}.npy"):
        print(f"File: {save_path}{file_name}.npy already found, skipping...")
        return np.load(f"{save_path}{file_name}.npy")
    # INFO
    print(f"GPU name: {gpu_name}")
    print(f"N={N}, T_A={T_A}, dtype={dtype}")
    print(f"Memory allocated: {N*N*jnp.dtype(dtype).itemsize/1e9} GB")

    # MESH
    shard_size = N // ndev
    mesh = jax.make_mesh((ndev,), ("x",))
    if ndev > 1:

        @jax.jit
        @partial(jax.shard_map, mesh=mesh, in_specs=(), out_specs=P(None, "x"))
        def make_diag():
            idx = jax.lax.axis_index("x")  # device index
            col_start = idx * shard_size  # global column offset
            # Allocate zeros of shape (N, chunk_size)
            local = jnp.zeros((N, shard_size), dtype=dtype)
            # Global column indices handled by this shard
            cols = jax.lax.iota(jnp.int32, shard_size) + col_start
            # Rows = same as global cols (diagonal)
            rows = cols
            # Values for the diagonal
            vals = cols + 1  # because your diag entries are 1..N
            # Scatter into local slice (adjust columns relative to col_start)
            local = local.at[(rows, cols - col_start)].set(vals)
            return local

    else:
        make_diag = lambda: jax.device_put(
            jnp.diag(np.arange(N, dtype=dtype) + 1), NamedSharding(mesh, P(None, "x"))
        )
        
    myfn = partial(potrs, mesh=mesh, in_specs=(P(None, "x"), P(None, None)))

    @jax.jit
    def run_once():
        A = make_diag()
        b = jax.device_put(
            jnp.ones((N, NRHS), dtype=dtype), NamedSharding(mesh, P(None, None))
        )
        b = myfn(A, b, T_A)
        return b

    times = []
    for run in range(n_runs + 1):
        print("Data allocated")
        start = time.time()
        out = run_once()
        out.block_until_ready()
        end = time.time()
        if run > 0:  # skip jitted run
            times.append(end - start)
            print(f"Elapsed time {times[-1]} [s]")

    np.save(f"{save_path}/{file_name}", np.array(times))
    return np.array(times)


def collect_results(root: Path):
    """
    Walks through root/data_potrs folders and returns a dict:
    results[(gpu, dtype, ndev)][T_A][N] = median_time
    """
    results = {}

    for path in root.rglob("N_*_T_A_*.npy"):
        # Expect directory: data/<gpu>/<dtype>/ndev_<k>/N_<N>_T_A_<TA>.npy
        try:
            gpu = path.parents[2].name
            dtype = path.parents[1].name
            ndev = int(path.parents[0].name.split("_")[-1])
        except Exception:
            continue

        m = re.match(r"N_(\d+)_T_A_(\d+)\.npy$", path.name)
        if not m:
            continue
        N = int(m.group(1))
        T_A = int(m.group(2))

        times = np.load(path)
        if times.size == 0:
            continue

        med = float(np.median(times))
        key = (gpu, dtype, ndev)
        results.setdefault(key, {}).setdefault(T_A, {})[N] = med

    return results


def plot_group(group, title, figpath):
    fig, ax = plt.subplots(figsize=(7, 5))

    for T_A, runs in sorted(group.items()):
        Ns = sorted(runs.keys())
        meds = [runs[N] for N in Ns]
        ax.plot(Ns, meds, marker="o", label=f"T_A={T_A}")

    try:
        ax.set_xscale("log", base=2)
    except TypeError:
        ax.set_xscale("log", basex=2)
    ax.set_yscale("log")

    ax.set_xlabel("N")
    ax.set_ylabel("Median time [s]")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figpath / f"potrf_{title}.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    if 1:
        for T_A in [128, 256, 512, 1024, 2048]:
            # for N in [2**i for i in range(4, 18)]:
            #     print(f"N={N}, T_A={T_A}")
            #     shard_size = N//ndev
            #     try:
            #         validate_padding(calculate_padding(shard_size, T_A, ndev), ndev, shard_size, T_A)
            #     except ValueError:
            #         print(f"Tiling error: N={N}, T_A={T_A}")
            #         # T_A_min, T_A_max = calculate_valid_T_A(shard_size, T_A, ndev, T_A_max=shard_size)
            #         # print(f"New T_A {T_A_max}")
            #         continue
            #     data = main(N, T_A=T_A)
            if ndev==8:
                data = main(2**17+2**16, T_A=512)


    root = Path(__file__).parent / "data_potrs"
    figpath = Path(__file__).parent / "figures"
    results = collect_results(root)
    if not results:
        raise SystemExit("No .npy results found.")

    for (gpu, dtype, ndev), group in results.items():
        title = f"{gpu}_{dtype}_ndev={ndev}"
        plot_group(group, title, figpath)
