import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
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

from jaxmg.potrs import potrs

def main():
    dtype = jnp.float64
    devices = jax.devices("gpu")
    ndev = len(devices)
    N = 2**16
    T_A = 2048
    NRHS = 1
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

    myfn = jax.jit(
        partial(
            potrs, mesh=mesh, in_specs=(P(None, "x"), P(None, None)), cyclic_1d=False
        ),
        donate_argnums=0,
        static_argnums=2,
    )

    @jax.jit
    def run_once():
        A = make_diag()
        b = jax.lax.with_sharding_constraint(
            jnp.ones((N, NRHS), dtype=dtype), NamedSharding(mesh, P(None, None))
        )
        return myfn(A, b, T_A)
        return A,b
    @jax.jit
    def chosolve(A, b):
        cfac = jax.scipy.linalg.cho_factor(A)
        return jax.scipy.linalg.cho_solve(cfac, b)
    run_once()
    out = run_once()
    # out = myfn(A, b, T_A)
    # out = chosolve(A,b)
    # out = run_once()
    # A.block_until_ready()
    # b.block_until_ready()
    jax.profiler.start_trace("./tensorboard")
    out =run_once() #myfn(A, b, T_A)
    # out = chosolve(A,b)
    out.block_until_ready()
    jax.profiler.stop_trace()
        
    print("Done")
    

if __name__=="__main__":
    main()
    # /mnt/sw/nix/store/r3bwp9b2501bv77y6g1nwkb483p0y9z2-cuda-12.3.2/nsight-systems-2023.3.3/target-linux-x64/nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas --gpu-metrics-device=all --capture-range=cudaProfilerApi --output=profiler_output python accuracy.py jax