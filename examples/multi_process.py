# Call ./examples/multi_process.sh to launch this code!
import os
import sys

proc_id = int(sys.argv[1])
num_procs = int(sys.argv[2])

# initialize the distributed system
import jax

jax.config.update("jax_platform_name", "cpu")
jax.distributed.initialize("localhost:6000", num_procs, proc_id)

from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import jax.numpy as jnp
import numpy as np


def get_device_grid():
    by_proc = {}
    for d in jax.devices():
        by_proc.setdefault(d.process_index, []).append(d)
    hosts = sorted(by_proc)  # stable host order
    # dev_grid[y, x] = device with local index x on host y
    return np.array(
        [[by_proc[h][x] for x in range(jax.local_device_count())] for h in hosts]
    )


def create_2d_mesh():
    dev_grid = get_device_grid()
    return Mesh(dev_grid, ("x", "y"))


def create_1d_mesh():
    dev_grid = get_device_grid()
    return Mesh(dev_grid.flatten(), ("y",))


print(f"Rank {proc_id}")
print(f"Local devices {jax.local_device_count()}")
print(f"Global devices {jax.device_count()}")
print(f"World size {num_procs}")
print(f"Device grid\n {get_device_grid()}")


mesh2d = create_2d_mesh()

A = jax.device_put(
    jnp.diag(jnp.arange(1, jax.device_count() + 1, dtype=jnp.float32)),
    NamedSharding(mesh2d, P(None, ("x", "y"))),
)

for shard in A.addressable_shards:
    print(f"shard\n {shard.data}")

# Gather over the number of hosts
A = jax.lax.with_sharding_constraint(A, NamedSharding(mesh2d, P(None, "y")))

for shard in A.addressable_shards:
    print(f"shard\n {shard.data}")

from jaxmg import potrs
out= potrs(
    A,
    jnp.ones((jax.device_count(), 1), dtype=jnp.float32),
    T_A=256,
    mesh=mesh2d,
    in_specs=(P(None, "T"), P(None, None)),
)
