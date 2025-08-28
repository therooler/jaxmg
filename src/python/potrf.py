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
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from .utils import get_mesh_and_spec_from_array, check_matrix_validity
from .block_cyclic import block_cyclic_relayout

# Load the shared library with the FFI target definitions
SHARED_LIBRARY = os.path.join(os.path.dirname(__file__), "../../build/libpotrf.so")
library = ctypes.cdll.LoadLibrary(SHARED_LIBRARY)

jax.ffi.register_ffi_target(
    "potrf_mg", jax.ffi.pycapsule(library.MgFfi), platform="CUDA"
)


def potrf(a, b, T_A, block_cyclic: bool = False):
    """
    Compute the Cholesky decomposition of a symmetric matrix `a` and solve the linear system `a * x = b`.
    This function uses the JAX FFI to call a CusolverMg CUDA kernel for the computation.

    If `a` is not postive-definite, CusolverMg will fail and raise an error.
    If `a` is not symmetric, CusolverMg will fail and raise an error.

    If `block_cyclic` is set to True but the input arrays are not sharded in a block-cyclic manner,
    the data layout will be wrong and the kernel will fail since `a` will likely not be positve definite with the
    given data layout.

    Args:
        a: A 2D array representing the matrix to be decomposed.
        b: A 2D array representing the right-hand side of the linear system.
        block_cyclic: If True, guarantees that the input arrays are sharded in a block-cyclic manner.
                      If False, the arrays are expected to be sharded along the columns of `a` and replicated for `b`.
    Returns:
        The solution `x` as a 2D array, replicated across all devices.
    Raises:
        ValueError: If the input arrays do not have the correct shapes or sharding.
    """

    assert a.shape[0] == b.shape[0], "A and b must have the same number of rows."
    assert a.ndim == 2, "a must be a 2D array."
    assert b.ndim == 2, "b must be a 2D array."

    mesh_a, spec_a = get_mesh_and_spec_from_array(a)
    mesh_b, spec_b = get_mesh_and_spec_from_array(b)
    if (spec_a._partitions[0] != None) or (spec_a._partitions[1] == None):
        raise ValueError(
            "A must be sharded along the columns with PartitionSpec P(None, str)."
        )
    if spec_b != P(None, None):
        raise ValueError(
            "b must be replicated along all shards with PartitionSpec P(None, None)."
        )
    if mesh_a != mesh_b:
        raise ValueError("A and b must be on the same mesh.")

    def impl(target_name):
        out_type = jax.ShapeDtypeStruct(b.shape, jnp.float64)
        fn = lambda _a, _b: jax.ffi.ffi_call(
            target_name,
            (out_type,),
            input_layouts=(
                (1, 0),
                (1, 0),
            ),
            output_layouts=((1, 0),),
        )(_a, _b, T_A=int(T_A))
        return lambda _a, _b: jax.shard_map(
            fn,
            mesh=mesh_a,
            in_specs=(spec_a, spec_b),
            out_specs=spec_b,
            check_vma=False,
        )(_a, _b)

    if not block_cyclic and len(mesh_a.devices)>1:
        check_matrix_validity(a.shape[0], len(mesh_a.devices))
        jax.debug.print("before block-cyclic reshaping:\n{}", a)
        a = block_cyclic_relayout(a, T_A=T_A)
        jax.debug.print("after block-cyclic reshaping:\n{}", a)

    return jax.lax.platform_dependent(a, b, cuda=impl("potrf_mg"))
