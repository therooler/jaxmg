# In file gpu_example.py...
import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import sys

# Get the coordinator_address, process_id, and num_processes from the command line.
coord_addr = sys.argv[1]
proc_id = int(sys.argv[2])
num_procs = int(sys.argv[3])

# Initialize the GPU machines.
jax.distributed.initialize(coordinator_address=coord_addr,
                           num_processes=num_procs,
                           process_id=proc_id,
                           local_device_ids=proc_id)
print("process id =", jax.process_index())
print("global devices =", jax.devices())
print("local devices =", jax.local_devices())
print("visible devices", os.environ["CUDA_VISIBLE_DEVICES"])
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from jaxmg import potrs
from jaxmg import determine_distributed_setup

print(determine_distributed_setup())
N = 8  # - 2**12
NRHS = 1
T_A = 2
dtype = jnp.float32

ndev = jax.device_count()
chunk_size = N // ndev
mesh = jax.make_mesh((ndev,), ("x",))

_A = jnp.diag(jnp.arange(N, dtype=dtype) + 1)
_b = jnp.ones((N, 1), dtype=dtype)
A = jax.device_put(_A, NamedSharding(mesh, P(None, "x")))
b = jax.device_put(_b, NamedSharding(mesh, P(None, None)))
print("Calling solver")
out, status = potrs(A,b,T_A,mesh=mesh, in_specs =((P(None,"x"), P(None,None))), return_status=True)
out.block_until_ready()
print(out)
print(status)