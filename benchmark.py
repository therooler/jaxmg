# Copyright 2024 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""An end-to-end example demonstrating the use of the JAX FFI with CUDA.

The specifics of the kernels are not very important, but the general structure,
and packaging of the extension are useful for testing.
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
import ctypes
import time
import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import ffi
import os
from functools import partial
from jax.sharding import PartitionSpec as P, NamedSharding
from src.jaxmg.potrf import potrf

devices = jax.devices("gpu")


def main():
    # print(f"Getting FFI function from: {SHARED_LIBRARY}")
    N = 2**4  # - 2**12
    print(N)
    NRHS = 1
    T_A = 2
    dtype = jnp.float32
    print(f"Memory alloc: {N*N*jnp.dtype(dtype).itemsize/1e9} GB")

    ndev = len(devices)
    chunk_size = N // ndev
    mesh = jax.make_mesh((ndev,), ("x",))
    if ndev > 1:
        @jax.jit
        @partial(jax.shard_map, mesh=mesh, in_specs=(), out_specs=P(None, "x"))
        def make_diag():
            idx = jax.lax.axis_index("x")  # device index
            col_start = idx * chunk_size  # global column offset
            # Allocate zeros of shape (N, chunk_size)
            local = jnp.zeros((N, chunk_size), dtype=dtype)
            # Global column indices handled by this shard
            cols = jax.lax.iota(jnp.int32, chunk_size) + col_start
            # Rows = same as global cols (diagonal)
            rows = cols
            # Values for the diagonal
            vals = cols + 1  # because your diag entries are 1..N
            # Scatter into local slice (adjust columns relative to col_start)
            local = local.at[(rows, cols - col_start)].set(vals)
            return local

        A = make_diag()
    else:
        _A = jnp.diag(np.arange(N, dtype=dtype)+1)
        A = jax.device_put(_A, NamedSharding(mesh, P(None, "x")))

    # _b = jnp.ones((N, NRHS), dtype=dtype)
    # _b = jnp.concat([jnp.ones((N//2, NRHS), dtype=dtype), jnp.zeros((N//2, NRHS), dtype=dtype)], axis=0)
    b = jax.device_put(_b, NamedSharding(mesh, P(None, None)))

    print("Mat put on device")
    # time.sleep(5)
    # for i, shard in enumerate(A.addressable_shards):
    #     print(f"Shard A {i} on device {shard.device}:")
    #     print(shard.data)
    # for i, shard in enumerate(b.addressable_shards):
    #     print(f"Shard b {i} on device {shard.device}:")
    #     print(shard.data)
    # Reconstruct from getrf
    start = time.time()
    b_before = b.copy()
    out = potrf(A, b, T_A=T_A)
    out.block_until_ready()
    print(out)
    print(f"Done, elapsed time { time.time() - start} [s]")
    assert jnp.allclose(b_before.flatten(), (A@out).flatten())


if __name__ == "__main__":
    main()
