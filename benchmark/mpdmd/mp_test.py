# In file gpu_example.py...
import os

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"
# os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import jax
import sys
import numpy as np
from functools import partial

def random_psd(n, dtype, seed):
    """
    Generate a random n x n positive semidefinite matrix.
    """
    key = jax.random.key(seed)
    A = jax.random.normal(key, (n, n), dtype=dtype) / jnp.sqrt(n)
    return A @ A.T.conj() + jnp.eye(n, dtype=dtype) * 1e-5  # symmetric PSD

# # 4 GPUs 80gb
# N = 2**16  # 2**16
# print("N=", N)
# NRHS = 1
# T_A = 2120 # From 2120 onwards it gives NaN

#
N = 2**16  # 2**16
print("N=", N)
NRHS = 1
T_A = 1024 # From 2120 onwards it gives NaN



print("shards", (N//2) / T_A)

if len(sys.argv)>1:
    # Get the coordinator_address, process_id, and num_processes from the command line.
    coord_addr = sys.argv[1]
    proc_id = int(sys.argv[2])
    num_procs = int(sys.argv[3])

    # Initialize the GPU machines.
    jax.distributed.initialize(
        coordinator_address=coord_addr,
        num_processes=num_procs,
        process_id=proc_id,
        local_device_ids=proc_id,
    )
    print("process id =", jax.process_index())
    print("global devices =", jax.devices())
    print("local devices =", jax.local_devices())
    print("visible devices", os.environ["CUDA_VISIBLE_DEVICES"])
    import jax.numpy as jnp
    
    from jax.sharding import NamedSharding, PartitionSpec as P
    from jaxmg import potrs_no_shardmap
    from jaxmg import determine_distributed_setup
    from jaxmg import calculate_padding
    print("padding", calculate_padding(N//2, T_A, 2))
    print(determine_distributed_setup())
    dtype = jnp.float64

    ndev = int(os.environ["JAXMG_NUMBER_OF_DEVICES"])
    chunk_size = N // ndev
    mesh = jax.make_mesh((ndev,), ("x",))

    @jax.jit
    def run_once(seed):
        _A = jax.lax.with_sharding_constraint(
            random_psd(N, dtype=dtype, seed=seed), NamedSharding(mesh, P(None, "x"))
        )
        _b = jax.lax.with_sharding_constraint(
            jnp.ones((N, NRHS), dtype=dtype), NamedSharding(mesh, P(None, None))
        )
        return _A, _b
    for i in range(1):
        print(f"I = {i}")
        A, b = run_once(i)

        @partial(jax.jit, static_argnames=("_T_A",))
        def _solve(_a, _b, _T_A):
            return potrs_no_shardmap(_a, _b, _T_A)

        @partial(jax.jit, static_argnames=("_T_A",))
        def jitted_potrs(_a, _b, _T_A):
            out = jax.shard_map(
                partial(_solve, _T_A=_T_A),
                mesh=mesh,
                in_specs=(P(None, "x"), P(None, None)),
                out_specs=(P(None, None), P(None)),
                check_vma=False,
            )(_a, _b)
            return out

        out, status = jitted_potrs(A, b, T_A)
        print(f"Status: {status.block_until_ready()}")
        print(out.block_until_ready())
        assert status==0, f"FAILURE AT N={N} - T_A={T_A}"
        print("passed assertion")
else:

    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from jaxmg import potrs_no_shardmap
    from jaxmg import determine_distributed_setup

    print(determine_distributed_setup())

    print(T_A)
    dtype = jnp.float64

    ndev = int(os.environ["JAXMG_NUMBER_OF_DEVICES"])
    chunk_size = N // ndev
    mesh = jax.make_mesh((ndev,), ("x",))

    @jax.jit
    def run_once(seed):
        _A = jax.lax.with_sharding_constraint(
            random_psd(N, dtype=dtype, seed=seed), NamedSharding(mesh, P(None, "x"))
        )
        _b = jax.lax.with_sharding_constraint(
            jnp.ones((N, NRHS), dtype=dtype), NamedSharding(mesh, P(None, None))
        )
        return _A, _b

    for i in range(3):
        print(f"I = {i}")
        A, b = run_once(100*i)

        @partial(jax.jit, static_argnames=("_T_A",))
        def _solve(_a, _b, _T_A):
            return potrs_no_shardmap(_a, _b, _T_A)

        @partial(jax.jit, static_argnames=("_T_A",))
        def jitted_potrs(_a, _b, _T_A):
            out = jax.shard_map(
                partial(_solve, _T_A=_T_A),
                mesh=mesh,
                in_specs=(P(None, "x"), P(None, None)),
                out_specs=(P(None, None), P(None)),
                check_vma=False,
            )(_a, _b)
            return out

        out, status = jitted_potrs(A, b, T_A)
        print(f"Status: {status.block_until_ready()}")
        print(out.block_until_ready())
