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

import jax.numpy as jnp
from jax import Array
from jax.sharding import PartitionSpec as P

from .utils import get_mesh_and_spec_from_array
from .cyclic_1d import cyclic_1d_layout, undo_cyclic_1d_layout

# Load the shared library with the FFI target definitions
SHARED_LIBRARY = os.path.join(os.path.dirname(__file__), "bin/libsyevd.so")
library = ctypes.cdll.LoadLibrary(SHARED_LIBRARY)
SHARED_LIBRARY_NO_V = os.path.join(os.path.dirname(__file__), "bin/libsyevd_no_V.so")
library_no_V = ctypes.cdll.LoadLibrary(SHARED_LIBRARY_NO_V)

jax.ffi.register_ffi_target(
    "syevd_mg", jax.ffi.pycapsule(library.SyevdMgFFI), platform="CUDA"
)
jax.ffi.register_ffi_target(
    "syevd_no_V_mg", jax.ffi.pycapsule(library_no_V.SyevdMgFFI), platform="CUDA"
)


def syevd(
    a: Array,
    T_A: int,
    return_eigenvectors=True,
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
    if T_A > 1024:
        raise ValueError(
            "T_A has a maximum value of 1024 for SyevdMg, received T_A={T_A}"
        )

    def impl(target_name):
        if target_name == "syevd_mg":
            out_type = (
                jax.ShapeDtypeStruct((a.shape[0], a.shape[1] // ndev), a.dtype),
                jax.ShapeDtypeStruct((a.shape[0],), a.dtype),
                jax.ShapeDtypeStruct((1,), jnp.int32),
            )
            out_specs = (spec_a, P(spec_a._partitions[1]), P(spec_a._partitions[1]))
            output_layouts= ((1, 0), (0,), (0,))
        elif target_name == "syevd_no_V_mg":
            out_type = (
                jax.ShapeDtypeStruct((a.shape[0],), a.dtype),
                jax.ShapeDtypeStruct((1,), jnp.int32),
            )
            out_specs = (P(spec_a._partitions[1]), P(spec_a._partitions[1]))
            output_layouts= ((0,), (0,))
        else:
            raise NotImplementedError()

        def fn(_a):
            out = jax.ffi.ffi_call(
                target_name,
                out_type,
                input_layouts=((1, 0),),
                output_layouts=output_layouts,
            )(_a, T_A=int(T_A))
            return out

        return jax.jit(
            lambda _a: jax.shard_map(
                fn,
                mesh=mesh_a,
                in_specs=spec_a,
                out_specs=out_specs,
                check_vma=False,
            )(_a)
        )

    if not cyclic_1d and len(mesh_a.devices) > 1:
        a = cyclic_1d_layout(a, T_A=T_A)
    if return_eigenvectors:
        eigenvalues, V, status = jax.lax.platform_dependent(a, cuda=impl("syevd_mg"))
        if not cyclic_1d and len(mesh_a.devices) > 1:
            V = undo_cyclic_1d_layout(V, T_A)
        out = (eigenvalues, V)
    else:
        eigenvalues, status = jax.lax.platform_dependent(a, cuda=impl("syevd_no_V_mg"))
        out = (eigenvalues, )
    

    if return_status:
        return *out, status[0]
    else:
        return out
