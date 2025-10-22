# In file gpu_example.py...

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
import sys

# Get the coordinator_address, process_id, and num_processes from the command line.
coord_addr = sys.argv[1]
proc_id = int(sys.argv[2])
num_procs = int(sys.argv[3])

# Initialize the GPU machines.
jax.distributed.initialize(coordinator_address=coord_addr,
                           num_processes=num_procs,
                           process_id=proc_id)
print("process id =", jax.process_index())
print("global devices =", jax.devices())
print("local devices =", jax.local_devices())

mesh = jax.make_mesh(
        (jax.device_count(),),
        ("x",),
    )

_A = jnp.zeros(jax.device_count())
A = jax.device_put(_A, NamedSharding(mesh, P("x")))