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
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"
from jax.experimental import checkify
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
from src.jaxmg import potrs, syevd, potri, potrs_no_shardmap
from src.jaxmg.utils import random_psd
devices = jax.devices("gpu")


def random_psd(n, dtype, seed):
    """
    Generate a random n x n positive semidefinite matrix.
    """
    key = jax.random.key(seed)
    A = jax.random.normal(key, (n, n), dtype=dtype) / jnp.sqrt(n)
    return A @ A.T.conj() + jnp.eye(n, dtype=dtype) * 1e-6  # symmetric PSD


def main():
    # print(f"Getting FFI function from: {SHARED_LIBRARY}")
    N = 16  # - 2**12
    print(N)
    T_A = 256
    dtype = jnp.float64
    print(f"Memory alloc: {N*N*jnp.dtype(dtype).itemsize/1e9} GB")

    ndev = len(devices)
    chunk_size = N // ndev
    mesh = jax.make_mesh(
        (ndev,),
        ("x",),
    )

    _A = random_psd(N, dtype, seed=0)
    
    # _A = jnp.diag(jnp.arange(1, N + 1, dtype=dtype))
    eigenvalues_expected, V_expected = jnp.linalg.eigh(_A)
    
    # print(V_expected)
    print(eigenvalues_expected)
    A = jax.device_put(_A, NamedSharding(mesh, P(None, "x")))
    A.block_until_ready()
    print(A.sharding)
    with jnp.printoptions(linewidth=500):
        eigenvalues, v, status = syevd(
            A,
            T_A=T_A,
            mesh=mesh,
            in_specs=(P(None, "x"),),
            return_status=True,
            return_eigenvectors=True,
        )
        print(eigenvalues.sharding)
        print(eigenvalues.block_until_ready())
        # print("V")
        # print(V)
        # eigenvalus_VtAV = jnp.diag(V.T @ _A @ V)
        # print(eigenvalus_VtAV)
        # print(eigenvalues_expected)
        assert jnp.allclose(eigenvalues, eigenvalues_expected, atol=1e-5)


def main2():
    # print(f"Getting FFI function from: {SHARED_LIBRARY}")
    N = 4  # - 2**12
    print(N)
    NRHS = 1
    T_A = 1
    dtype = jnp.float32
    print(f"Memory alloc: {N*N*jnp.dtype(dtype).itemsize/1e9} GB")

    ndev = len(devices)
    chunk_size = N // ndev
    mesh = jax.make_mesh((ndev,), ("x",))
    # _A = jnp.load("updates.npy")
    _A = random_psd(N, dtype=dtype, seed=1230) - jnp.eye(N)*10
    _A = jnp.diag(jnp.arange(N, dtype=dtype) + 1)
    _A = _A.at[1,0].set(2.0)
    _b = jnp.ones((N, 1), dtype=dtype)
    print(jnp.linalg.cond(_A))
    # _A = jnp.diag(jnp.arange(1, N + 1, dtype=dtype))
    # print("eigenvalues", jnp.linalg.eigvalsh(_A))
    # _b = jnp.ones((N, NRHS), dtype=dtype)
    # _b = jax.random.normal(shape=(N, NRHS), key=jax.random.key(1234), dtype=dtype)

    # cfac = jax.scipy.linalg.cho_factor(_A)
    # expected_out = jax.scipy.linalg.cho_solve(cfac, _b)
    A = jax.device_put(_A, NamedSharding(mesh, P(None, "x")))

    # _b = jnp.concat([jnp.ones((N//2, NRHS), dtype=dtype), jnp.zeros((N//2, NRHS), dtype=dtype)], axis=0)
    b = jax.device_put(_b, NamedSharding(mesh, P(None, None)))

    print("Mat put on device")
    A.block_until_ready()
    b.block_until_ready()
    out, status = potrs(A,b,T_A,mesh=mesh, in_specs =((P(None,"x"), P(None,None))), return_status=True)
    out.block_until_ready()
    print(status)
    print(out)
    # def f(mes):
    #     jax.debug.print("mes={}", mes==7)
    #     jax.lax.cond(mes==7,
    #                  lambda :jnp.save("test", mes),
    #                  lambda : None
    #                  )
    #     # checkify.check(mes!=7, "CuSolverMg failed with status 7")
    #     return mes
    # checkedf = checkify.checkify(f)
    # err, status = jax.jit(checkedf)(status)
    # try:
    #     err.throw()
    # except ValueError as e:
    #     print(e)

    # print("status", status)
    # time.sleep(5)
    # for i, shard in enumerate(A.addressable_shards):
    #     print(f"Shard A {i} on device {shard.device}:")
    #     print(shard.data)
    # for i, shard in enumerate(b.addressable_shards):
    #     print(f"Shard b {i} on device {shard.device}:")
    #     print(shard.data)
    # Reconstruct from getrf
    # start = time.time()
    # b_before = b.copy()
    # @partial(jax.shard_map, mesh=mesh, out_specs = P(None, None), in_specs=(P(None, "x"), P(None, None)), check_vma=False)
    # def fn(_A, _b):
    #     return potrs_no_shardmap(_A, _b, T_A=T_A)
    # out=fn(A,b)
    # print(out.shape)
    # out.block_until_ready()
    # print(out)
    # print(expected_out)
    # print(jnp.max((A @ out - b_before) / abs(b_before)))
    # print(jnp.max((A @ expected_out - b_before) / abs(b_before)))

    # print(jnp.max(jnp.abs(out - expected_out) / abs(out)))
    # print(f"Done, elapsed time { time.time() - start} [s]")
    # assert jnp.allclose(b_before, A @ expected_out, rtol=jnp.finfo(dtype).eps)
    # assert jnp.allclose(b_before, A @ out, rtol=jnp.finfo(dtype).eps)


def main3():
    ndev = len(devices)
    print(f"Available devices: {ndev}")

    # PARAMETERS
    N = 7
    T_A = 256
    NRHS = 1
    dtype = jnp.complex64
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
        # make_diag = lambda: jax.device_put(
        #     jnp.diag(np.arange(N, dtype=dtype) + 1), NamedSharding(mesh, P(None, "x"))
        # )
        make_diag = lambda: jax.device_put(
            random_psd(N, dtype=dtype, seed=1234), NamedSharding(mesh, P(None, "x"))
        )

    # A = make_diag()
    _a = jnp.diag(jnp.arange(1 ,N+1,dtype=dtype))
    _a = _a.at[1,0].set(2.)
    _a =  jax.device_put(
           _a, NamedSharding(mesh, P(None, "x"))
        )
    myfn = partial(potri, mesh=mesh, in_specs=(P(None, "x")), return_status=True)
    Ainv, status = myfn(_a, T_A)

    print(status)
    print(Ainv)
    exit()
    # myfn = partial(potrs, mesh=mesh, in_specs=(P(None, "x"), P(None, None)))

    @jax.jit
    def get_data():
        A = make_diag()
        # b = jax.device_put(
        #     jnp.ones((N, NRHS), dtype=dtype), NamedSharding(mesh, P(None, None))
        # )
        Ainv, status = myfn(A, T_A)
        return A, Ainv, status

    times = []
    for i in range(3):
        A, Ainv, status= get_data()
        print(status)
        print(jnp.linalg.eigvals(A))
        print(A @ Ainv)
        Ainvlax = jnp.linalg.inv(A)
        residual = A @ Ainv
        residual_lax = A @ Ainvlax
        error = jnp.linalg.norm(residual - jnp.eye(A.shape[0], dtype=A.dtype), ord="fro")
        errorlax = jnp.linalg.norm(residual_lax - jnp.eye(A.shape[0], dtype=A.dtype), ord="fro")
        print(error)
        print(errorlax)
        print("Done")

def main4():
    N = 8
    T_A=2
    dtype=jnp.float64
    mesh = jax.make_mesh((len(jax.devices()),), ("x",))

    A = random_psd(N, dtype=dtype, seed=1234)
    # Make mesh and place data
    A = jax.device_put(A, NamedSharding(mesh, P(None, "x")))
    out, V = jax.jit(
        partial(syevd, mesh=mesh, in_specs=(P(None, "x"),), return_eigenvectors=True),
        static_argnums=1,
    )(A, T_A=T_A)
    for shard in out.addressable_shards:
        print(shard.data)


if __name__ == "__main__":
    main4()
