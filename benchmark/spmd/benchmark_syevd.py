import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import subprocess
from pathlib import Path

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
import time
import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from functools import partial
from jax.sharding import PartitionSpec as P, NamedSharding

from jaxmg import syevd

from pathlib import Path

dtype = jnp.float64
devices = jax.devices("gpu")
ndev = len(devices)

n_runs = 5

def main_syevd(N, T_A):

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
    save_path = f"{Path(__file__).parent}/data_syevd/{gpu_name}/{jnp.dtype(dtype).name}/ndev_{ndev}/"
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
        @partial(jax.shard_map, mesh=mesh, in_specs=(), out_specs=P("x", None))
        def make_diag():
            idx = jax.lax.axis_index("x")  # device index
            col_start = idx * shard_size  # global column offset
            # Allocate zeros of shape (N, chunk_size). Allocate the padded
            # first-dimension so we can avoid a separate `pad` op later.
            # This makes the shard have shape (shard_size + padding, N).
            local = jnp.zeros((shard_size, N), dtype=dtype)
            # Global column indices handled by this shard
            cols = jax.lax.iota(jnp.int32, shard_size) + col_start
            # Rows = same as global cols (diagonal)
            rows = cols
            # Values for the diagonal
            vals = cols + 1  # because your diag entries are 1..N
            # Scatter into local slice (adjust columns relative to col_start)
            local = local.at[(rows - col_start, cols)].set(vals)
            return local

    else:
        make_diag = lambda: jax.lax.with_sharding_constraint(
            jnp.diag(jnp.arange(N, dtype=dtype) + 1), NamedSharding(mesh, P("x", None,))
        )

    myfn = jax.jit(partial(syevd, mesh=mesh, in_specs=(P("x", None))), static_argnums=1)

    @jax.jit
    def run_once():
        A = make_diag()
        return A

    times = []
    for run in range(n_runs + 1):
        print("Data allocated")

        start = time.time()
        A = run_once()
        A.block_until_ready()
        ev, V  = myfn(A, T_A)
        ev.block_until_ready()
        V.block_until_ready()
        end = time.time()
        if run > 0:  # skip jitted run
            times.append(end - start)
            print(f"Elapsed time {times[-1]} [s]")

    np.save(f"{save_path}/{file_name}", np.array(times))
    return np.array(times)


def main_eigh(N):

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
    save_path = f"{Path(__file__).parent}/data_syevd/{gpu_name}/{jnp.dtype(dtype).name}/ndev_{ndev}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = f"N_{N}_jax_native"
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

    @partial(jax.jit, donate_argnums=0)
    def eigh(A):
        return jnp.linalg.eigh(A)

    @jax.jit
    def run_once():
        A = make_diag()
        return A

    times = []
    for run in range(n_runs + 1):
        print("Data allocated")

        start = time.time()
        A = run_once()
        A.block_until_ready()
        ev, V = eigh(A)
        ev.block_until_ready()
        V.block_until_ready()
        end = time.time()
        if run > 0:  # skip jitted run
            times.append(end - start)
            print(f"Elapsed time {times[-1]} [s]")

    np.save(f"{save_path}/{file_name}", np.array(times))
    return np.array(times)


if __name__ == "__main__":

    for N in [2**i for i in range(4, 18)] + [58000] + [140000]:
        if ndev == 1:
            main_eigh(N)
        else:
            for T_A in [2**i for i in range(8,11)]:
                print(f"N={N}, T_A={T_A}")
                main_syevd(N, T_A=T_A)