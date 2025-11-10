import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import subprocess
from pathlib import Path

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
import time
import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from functools import partial
from jax.sharding import PartitionSpec as P, NamedSharding

from jaxmg._potrs import potrs
from jaxmg._cyclic_1d import (
    calculate_valid_T_A,
    validate_padding,
    calculate_padding,
    cyclic_1d_no_shardmap,
)
import re
from pathlib import Path
import matplotlib.pyplot as plt

dtype = jnp.float64
devices = jax.devices("gpu")
ndev = len(devices)


def main_potrs(N, T_A):

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
    save_path = f"{Path(__file__).parent}/data_potrs_fix/{gpu_name}/{jnp.dtype(dtype).name}/ndev_{ndev}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = f"N_{N}_T_A_{T_A}"
    if os.path.exists(f"{save_path}{file_name}.npy"):
        print(f"File: {save_path}{file_name}.npy already found, skipping...")
        return np.load(f"{save_path}{file_name}.npy")
    # INFO
    print(f"GPU name: {gpu_name}")
    print(f"N={N}, T_A={T_A}, dtype={dtype}")
    print(jnp.dtype(dtype).itemsize)
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
        make_diag = lambda: jax.lax.with_sharding_constraint(
            jnp.diag(jnp.arange(N, dtype=dtype) + 1), NamedSharding(mesh, P(None, "x"))
        )

    myfn = jax.jit(partial(
            potrs, mesh=mesh, in_specs=(P(None, "x"), P(None, None)), cyclic_1d=False
        ), static_argnums=2)

    @jax.jit
    def run_once():
        A = make_diag()
        b = jax.lax.with_sharding_constraint(
            jnp.ones((N, NRHS), dtype=dtype), NamedSharding(mesh, P(None, None))
        )
        out = myfn(A, b, T_A)
        return out

    cyclic_fn = jax.jit(
        jax.shard_map(
            partial(
                cyclic_1d_no_shardmap,
                T_A=T_A,
                ndev=ndev,
                axis_name="x",
            ),
            mesh=mesh,
            in_specs=P(None, "x"),
            out_specs=P(None, "x"),
        ),
        donate_argnums=0,
    )
    times = []
    for run in range(n_runs + 1):
        print("Data allocated")
        # if ndev > 1:
        #     A = cyclic_fn(A)
        # A.block_until_ready()
        start = time.time()
        out = run_once()
        out.block_until_ready()
        # time.sleep(1)
        end = time.time()
        if run > 0:  # skip jitted run
            times.append(end - start)
            print(f"Elapsed time {times[-1]} [s]")

    np.save(f"{save_path}/{file_name}", np.array(times))
    return np.array(times)


def main_cho_solve(N):

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
    save_path = f"{Path(__file__).parent}/data_potrs_cyclic/{gpu_name}/{jnp.dtype(dtype).name}/ndev_{ndev}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = f"N_{N}_chosolve"
    if os.path.exists(f"{save_path}{file_name}.npy"):
        print(f"File: {save_path}{file_name}.npy already found, skipping...")
        return np.load(f"{save_path}{file_name}.npy")
    # INFO
    print(f"GPU name: {gpu_name}")
    print(f"N={N}, dtype={dtype}")
    print(jnp.dtype(dtype).itemsize)
    print(f"Memory allocated: {N*N*jnp.dtype(dtype).itemsize/1e9} GB")

    # MESH
    make_diag = jax.jit(lambda: jnp.diag(jnp.arange(N, dtype=dtype) + 1))

    @jax.jit
    def chosolve(A, b):
        cfac = jax.scipy.linalg.cho_factor(A)
        return jax.scipy.linalg.cho_solve(cfac, b)

    @jax.jit
    def run_once():
        A = make_diag()
        b = jnp.ones((N, NRHS), dtype=dtype)
        out = chosolve(A, b)
        return out

    times = []
    for run in range(n_runs + 1):
        print("Data allocated")
        # if ndev > 1:
        #     A = cyclic_fn(A)
        # A.block_until_ready()
        start = time.time()
        out = run_once()
        out.block_until_ready()
        # time.sleep(1)
        end = time.time()
        if run > 0:  # skip jitted run
            times.append(end - start)
            print(f"Elapsed time {times[-1]} [s]")

    np.save(f"{save_path}/{file_name}", np.array(times))
    return np.array(times)


def collect_results(root: Path):
    """
    Walks through root/data_potrs_cyclic folders and returns a dict:
    results[(gpu, dtype, ndev)][T_A][N] = median_time
    """
    results = {}

    for path in root.rglob("N_*.npy"):
        # Expect directory: data/<gpu>/<dtype>/ndev_<k>/N_<N>_T_A_<TA>.npy
        try:
            gpu = path.parents[2].name
            dtype = path.parents[1].name
            ndev = int(path.parents[0].name.split("_")[-1])
        except Exception:
            continue

        m = re.match(r"N_(\d+)_T_A_(\d+)\.npy$", path.name)
        if not m:
            m = re.match(r"N_(\d+)_chosolve\.npy$", path.name)
            if not m:
                continue
            N = int(m.group(1))

            times = np.load(path)
            if times.size == 0:
                continue

            med = float(np.median(times))
            key = (gpu, dtype, ndev)
            results.setdefault(key, {}).setdefault("chosolve", {})[N] = med
        else:
            N = int(m.group(1))
            T_A = int(m.group(2))

            times = np.load(path)
            if times.size == 0:
                continue

            med = float(np.median(times))
            key = (gpu, dtype, ndev)
            results.setdefault(key, {}).setdefault(T_A, {})[N] = med

    return results


def plot_group_T_A(results, figpath):
    for (gpu, dtype, ndev), group in results.items():
        title = f"{gpu}_{dtype}_ndev={ndev}"
        fig, ax = plt.subplots(figsize=(7, 5))
        group_filtered = {
            key: group[key] for key in group.keys() if isinstance(key, int)
        }

        for T_A, runs in sorted(group_filtered.items()):
            Ns = sorted(runs.keys())
            meds = [runs[N] for N in Ns]
            ax.plot(Ns, meds, marker="o", label=f"T_A={T_A}")
        try:
            runs = results[(gpu, dtype, 1)]["chosolve"]
            Ns = sorted(runs.keys())
            meds = [runs[N] for N in Ns]
            ax.plot(Ns, meds, marker="o", label=f"chosolve", color="black")
        except KeyError:
            pass
        try:
            ax.set_xscale("log", base=2)
        except TypeError:
            ax.set_xscale("log", basex=2)
        ax.set_yscale("log")

        ax.set_xlabel("N")
        ax.set_ylabel("Median time [s]")
        ax.set_ylim([1e-4, 1e2])
        ax.set_ylabel("Median time [s]")
        ax.set_title(title)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figpath / f"T_A_potrf_{title}.pdf", bbox_inches="tight")
        plt.show()


def plot_group_ndev(group, figpath):
    data_plot = {}
    for (gpu, dtype, ndev), group in results.items():
        for T_A, runs in group.items():
            sorted_runs = sorted(runs.items(), key=lambda x: x[0])
            Ns = [k for k, v in sorted_runs]
            meds = [v for k, v in sorted_runs]
            data_plot.setdefault((gpu, dtype, T_A), {}).setdefault(ndev, (Ns, meds))
    for gpu, dtype, T_A in data_plot.keys():
        fig, ax = plt.subplots(figsize=(7, 5))
        title = f"{gpu}_{dtype}_T_A={T_A}"
        for ndev in data_plot[(gpu, dtype, T_A)].keys():
            x, y = data_plot[(gpu, dtype, T_A)][ndev]
            ax.plot(x, y, marker="o", label=f"ndev={ndev}")

            try:
                x, y = data_plot[(gpu, dtype, "chosolve")][1]
                ax.plot(x, y, marker="o", label=f"chosolve", color="black")
            except KeyError:
                pass
        try:
            ax.set_xscale("log", base=2)
        except TypeError:
            ax.set_xscale("log", basex=2)
        ax.set_yscale("log")

        ax.set_xlabel("N")
        ax.set_ylabel("Median time [s]")
        ax.set_ylim([1e-4, 1e2])
        ax.set_title(title)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figpath / f"ndev_potrf_{title}.pdf", bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    if ndev == 8:
        extra_N = [
            140000,
        ]
    else:
        extra_N = []
    # for N in [2**i for i in range(4, 16)] + extra_N:
    for N in [2**16,]:
        for T_A in [
            512 *4,
        ]:
            print(f"N={N}, T_A={T_A}")
            shard_size = N // ndev
            try:
                validate_padding(
                    calculate_padding(shard_size, T_A, ndev), ndev, shard_size, T_A
                )
            except ValueError:
                print(f"Tiling error: N={N}, T_A={T_A}")
                print(calculate_valid_T_A(shard_size, T_A, ndev, shard_size))
                continue
            main_potrs(N, T_A=T_A)
        if ndev == 1:
            main_cho_solve(N)

    root = Path(__file__).parent / "data_potrs_fix"
    figpath = Path(__file__).parent / "figures_fix"
    results = collect_results(root)
    if not results:
        raise SystemExit("No .npy results found.")

    plot_group_T_A(results, figpath)
    plot_group_ndev(results, figpath)
