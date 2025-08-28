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
import ctypes

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import ffi
import os
from jax.sharding import PartitionSpec as P, NamedSharding
from src.python.potrf import potrf
devices = jax.devices("gpu")
print(devices)


def main():
    # print(f"Getting FFI function from: {SHARED_LIBRARY}")
    N = 8
    T_A = 2
    A = jnp.diag(jnp.arange(N, dtype=jnp.float64)+1)
    print(jnp.linalg.eigvalsh(A))
    print(A)
    b = jnp.ones((N, 1), dtype=jnp.float64)
    ndev = len(devices)
    # Make mesh and place data
    mesh = jax.make_mesh((ndev,), ('x', ))
    A = jax.device_put(A, NamedSharding(mesh, P(None, 'x')))
    b = jax.device_put(b, NamedSharding(mesh, P(None, None)))

    for i, shard in enumerate(A.addressable_shards):
        print(f"Shard A {i} on device {shard.device}:")
        print(shard.data)
    for i, shard in enumerate(b.addressable_shards):
        print(f"Shard b {i} on device {shard.device}:")
        print(shard.data)
    # Reconstruct from getrf
    out = potrf(A, b, T_A=T_A)
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
   
