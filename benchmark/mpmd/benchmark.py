# In file gpu_example.py...
import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"
import sys

# Get the coordinator_address, process_id, and num_processes from the command line.

if len(sys.argv) > 1:
    coord_addr = sys.argv[1]
    proc_id = int(sys.argv[2])
    num_procs = int(sys.argv[3])
    import jax

    # Initialize the GPU machines.
    jax.distributed.initialize(
        coordinator_address=coord_addr,
        num_processes=num_procs,
        process_id=proc_id,
        local_device_ids=proc_id,
    )
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import jax
print("process id =", jax.process_index())
print("global devices =", jax.devices())
print("local devices =", jax.local_devices())
print("visible devices", os.environ["CUDA_VISIBLE_DEVICES"])


import numpy as np
from functools import partial
import time


def random_psd(n, dtype, seed):
    """
    Generate a random n x n positive semidefinite matrix.
    """
    key = jax.random.key(seed)
    A = jax.random.normal(key, (n, n), dtype=dtype) / jnp.sqrt(n)
    return A @ A.T.conj() + jnp.eye(n, dtype=dtype) * 1e-5  # symmetric PSD


import jax.numpy as jnp
from jax.experimental.multihost_utils import process_allgather
from jax.sharding import NamedSharding, PartitionSpec as P

from jaxmg import potrs, calculate_padding, cyclic_1d_no_shardmap

dtype = jnp.float64
ndev = int(os.environ["JAXMG_NUMBER_OF_DEVICES"])
mesh = jax.make_mesh((ndev,), ("x",))

import re
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt


@jax.jit
def run_once(seed, N, NRHS):
    _A = jax.lax.with_sharding_constraint(
        random_psd(N, dtype=dtype, seed=seed), NamedSharding(mesh, P(None, "x"))
    )
    _b = jax.lax.with_sharding_constraint(
        jnp.ones((N, NRHS), dtype=dtype), NamedSharding(mesh, P(None, None))
    )
    return _A, _b


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


def plot_group_ndev(results, figpath):
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


def main_potrs(N, T_A):
    pid = jax.process_index()
    print(f"Available devices: {ndev}")
    gpu_name = ""
    if pid == 0:
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
    n_runs = 5
    save_path = f"{Path(__file__).parent}/data_potrs/{gpu_name}/{jnp.dtype(dtype).name}/ndev_{ndev}/"
    if not os.path.exists(save_path) and pid == 0:
        os.makedirs(save_path)
    file_name = f"N_{N}_T_A_{T_A}"
    returning = False
    if os.path.exists(f"{save_path}{file_name}.npy"):
        print(f"File: {save_path}{file_name}.npy already found, skipping...")
        returning = True

    gathered_return = process_allgather(jnp.array(returning, dtype=jnp.bool))
    print(f"Are we returning? {gathered_return}")
    if bool(jnp.any(gathered_return)):
        print(f"pid {pid} is returning")
        return
    # INFO
    print(f"GPU name: {gpu_name}")
    print(f"N={N}, T_A={T_A}, dtype={dtype}")
    print(jnp.dtype(dtype).itemsize)
    print(f"Memory allocated: {N*N*jnp.dtype(dtype).itemsize/1e9} GB")

    myfn = jax.jit(
        partial(potrs, mesh=mesh, in_specs=(P(None, "x"), P(None, None))), 
        static_argnums=2, 
    )

    @jax.jit
    def run_once():
        _A = jax.lax.with_sharding_constraint(
            jnp.diag(jnp.arange(N, dtype=dtype) + 1), NamedSharding(mesh, P(None, "x"))
        )
        _b = jax.lax.with_sharding_constraint(
            jnp.ones((N, NRHS), dtype=dtype), NamedSharding(mesh, P(None, None))
        )
        return myfn(_A, _b, T_A)

    times = []
    for run in range(n_runs + 1):
        print("Allocating A, b")
        start = time.time()
        out = run_once()
        out.block_until_ready()
        end = time.time()
        if run > 0:  # skip jitted run
            times.append(end - start)
            print(f"Elapsed time {times[-1]} [s]")
    if pid == 0:
        np.save(f"{save_path}/{file_name}", np.array(times))
    return np.array(times)


def main_cho_solve(N):

    print(f"Available devices: {ndev}")
    gpu_name = ""
    pid = jax.process_index()
    if pid == 0:
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
    n_runs = 5
    save_path = f"{Path(__file__).parent}/data_potrs/{gpu_name}/{jnp.dtype(dtype).name}/ndev_{ndev}/"
    if not os.path.exists(save_path) and pid == 0:
        os.makedirs(save_path)
    file_name = f"N_{N}_chosolve"
    if os.path.exists(f"{save_path}{file_name}.npy"):
        print(f"File: {save_path}{file_name}.npy already found, skipping...")
        return

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
        start = time.time()
        out = run_once()
        out.block_until_ready()
        end = time.time()
        if run > 0:  # skip jitted run
            times.append(end - start)
            print(f"Elapsed time {times[-1]} [s]")
    np.save(f"{save_path}/{file_name}", np.array(times))
    return np.array(times)


def main():
    for N in [72000]:
        for T_A in [1500]:
            print(f"N={N}, T_A={T_A}")
            main_potrs(N, T_A=T_A)
        if ndev == 1:
            main_cho_solve(N)


main()
