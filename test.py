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
from src.jaxmg import potri, potrf, undo_cyclic_1d_layout

devices = jax.devices("gpu")


def random_psd(n, dtype, seed):
    """
    Generate a random n x n positive semidefinite matrix.
    """
    key = jax.random.key(seed)
    A = jax.random.normal(key, (n, n), dtype=dtype) / jnp.sqrt(n)
    return A @ A.T  + jnp.eye(n, dtype=dtype)*1e-3# symmetric PSD


def main():
    # print(f"Getting FFI function from: {SHARED_LIBRARY}")
    N = 2**3  # - 2**12
    print(N)
    NRHS = 1
    T_A = 2
    dtype = jnp.float64
    print(f"Memory alloc: {N*N*jnp.dtype(dtype).itemsize/1e9} GB")

    ndev = len(devices)
    chunk_size = N // ndev
    mesh = jax.make_mesh(
        (ndev,),
        ("x",),
    )

    _A = random_psd(N, dtype, seed=0)
    # _A = jnp.diag(jnp.arange(1, N+1, dtype=dtype))
    A = jax.device_put(_A, NamedSharding(mesh, P(None, "x")))

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
    print(A)
    print(jnp.linalg.inv(A))
    with jnp.printoptions(linewidth=500):
        print(A.shape)
        out, status = potri(A, T_A=T_A, return_status=True)
        print(out.shape)
        print(status)
        # out = undo_cyclic_1d_layout(out, T_A)
        out.block_until_ready()
        print("OUT")
        print(out)
        print(A)
        print(A @ out)
    assert jnp.allclose(A @ out, jnp.eye(N, dtype=dtype))


def main2():
    # print(f"Getting FFI function from: {SHARED_LIBRARY}")
    N = 2**10  # - 2**12
    print(N)
    NRHS = 1
    T_A = 256
    dtype = jnp.float32
    print(f"Memory alloc: {N*N*jnp.dtype(dtype).itemsize/1e9} GB")

    ndev = len(devices)
    chunk_size = N // ndev
    mesh = jax.make_mesh((ndev,), ("x",))

    _A = random_psd(N, dtype, seed=0)
    print("eigenvalues", jnp.linalg.eigvalsh(_A))
    _b = jnp.ones((N, NRHS), dtype=dtype)

    cfac = jax.scipy.linalg.cho_factor(_A)
    expected_out = jax.scipy.linalg.cho_solve(cfac, _b)
    A = jax.device_put(_A, NamedSharding(mesh, P(None, "x")))

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
    print(expected_out)
    print(jnp.max((A @ out - b_before) / abs(b_before)))
    print(jnp.max((A @ expected_out - b_before) / abs(b_before)))

    print(jnp.max(jnp.abs(out - expected_out) / abs(out)))
    print(f"Done, elapsed time { time.time() - start} [s]")
    assert jnp.allclose(b_before, A @ expected_out, rtol=jnp.finfo(dtype).eps)
    assert jnp.allclose(b_before, A @ out, rtol=jnp.finfo(dtype).eps)


if __name__ == "__main__":
    main2()
