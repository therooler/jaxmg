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
from functools import partial
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import Array
from jax.sharding import PartitionSpec as P

from .utils import get_mesh_and_spec_from_array, check_matrix_validity
from .cyclic_1d import cyclic_1d_layout

# Load the shared library with the FFI target definitions
SHARED_LIBRARY = os.path.join(os.path.dirname(__file__), "bin/libpotri.so")
library = ctypes.cdll.LoadLibrary(SHARED_LIBRARY)

jax.ffi.register_ffi_target(
    "potri_mg", jax.ffi.pycapsule(library.PotriMgFFI), platform="CUDA"
)


def potri(
    a: Array,
    T_A: int,
    cyclic_1d: bool = False,
    return_status: bool = False,
):
    """
    Compute the inverse of a symmetric matrix `a`.
    This function uses the JAX FFI to call a CusolverMg CUDA kernel for the computation.

    If `a` is not postive-definite, CusolverMg will fail and raise an error.
    If `a` is not symmetric, CusolverMg will fail and raise an error.

    If `cyclic_1d` is set to True but the input arrays are not sharded in a cyclic 1d manner,
    the data layout will be wrong and the kernel will fail since `a` will likely not be positve definite with the
    given data layout.

    If the provided matrix is not positive definite, or correctly sharded (even though `cyclic_1d` is True),
    the returned result will be NaN.

    Args:
        a: A 2D array representing the matrix to be decomposed.
        T_A: Tile size used for cyclic 1d layout. Only used if `cyclic_1d` is True.
        cyclic_1d: If True, guarantees that the input arrays are sharded in a cyclic 1d manner.
                      If False, the arrays are expected to be sharded along the columns of `a` and replicated for `b`.
        return_status: If True, returns a tuple (x, status) where `status` is an integer indicating the success or failure of the computation.
            If status>0, the integer corresponds to the cusolverStatus_t returned by cusolverMgPotrf.
            If status<0, the integer corresponds to the cusolverStatus_t returned by cusolverMgPotrs.
            For CUDA Toolkit 12.8.0, the possible values can be found in `cusolver_common.h`.

    Returns:
        The solution `A^{-1}` as a 2D array, sharded columnwise across all devices.
        If `return_status` is True, also returns the `status` integer.
    Raises:
        ValueError: If the input arrays do not have the correct shapes or sharding.
    """

    assert a.ndim == 2, "a must be a 2D array."

    mesh_a, spec_a = get_mesh_and_spec_from_array(a)
    ndev = len(mesh_a.devices)
    if (spec_a._partitions[0] != None) or (spec_a._partitions[1] == None):
        raise ValueError(
            "A must be sharded along the columns with PartitionSpec P(None, str)."
        )

    def impl(target_name):
        out_type = (
            jax.ShapeDtypeStruct((a.shape[0], a.shape[1]//ndev), a.dtype),
            jax.ShapeDtypeStruct((1,), jnp.int32),
        )
        def fn(_a):
            out, status = jax.ffi.ffi_call(
                target_name,
                out_type,
                input_layouts=((1, 0),),
                output_layouts=((1, 0), (0,)),
            )(_a, T_A=int(T_A))
            return out, status
        return lambda _a: jax.shard_map(
                fn,
                mesh=mesh_a,
                in_specs=spec_a,
                out_specs=(spec_a, P(spec_a._partitions[1])),
                check_vma=False,
            )(_a)

    if not cyclic_1d and len(mesh_a.devices) > 1:
        check_matrix_validity(a.shape[0], len(mesh_a.devices))
        print("Starting cyclic 1d")
        a = cyclic_1d_layout(a, T_A=T_A)
        print("Done with cyclic 1d")

    # @partial(jax.jit, in_shardings=a.sharding, out_shardings=a.sharding)
    # def symmetrize(L):
    #     L = jnp.tril(L)
    #     return L + L.T - jnp.diag(jnp.diag(L))
    print(a.sharding)
    print(a.shape)
    
    out, status = jax.lax.platform_dependent(a, cuda=impl("potri_mg"))
    print(out.sharding)
    print(out.shape)
    # @partial(jax.jit, in_shardings=out.sharding, out_shardings=out.sharding)
    def symmetrize(L):
        L = jnp.tril(L)
        return L + L.T - jnp.diag(jnp.diag(L))
    # out = symmetrize(out)
    # print(out.sharding)
    print("out")
    print(out)
    # print(jnp.tril(out))
    # for i, shard in enumerate(jnp.tril(out).addressable_shards):
    #     print(f"Shard A {i} on device {shard.device}:")
    #     print(shard.data)
    # print("out.T")
    # print(out.T)
    # out = symmetrize(out)
    if return_status:
        return out, status[0]
    else:
        return out
