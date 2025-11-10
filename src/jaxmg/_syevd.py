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
import jax.numpy as jnp
from jax import Array
from jax.sharding import PartitionSpec as P, Mesh

from functools import partial
from typing import Tuple, Union

from ._cyclic_1d import (
    cyclic_1d_no_shardmap,
    undo_cyclic_1d_no_shardmap,
)
from .utils import maybe_real_dtype_from_complex


def syevd(
    a: Array,
    T_A: int,
    mesh: Mesh,
    in_specs: Tuple[P],
    return_eigenvectors: bool = True,
    cyclic_1d: bool = False,
    return_status: bool = False,
 ) -> Union[Array, Tuple[Array, Array], Tuple[Array, Array, int], Tuple[Array, int]]:
    """
    Computes the eigenvalue decomposition of a symmetric matrix `a` using a distributed multi-GPU CUDA kernel via JAX FFI.

    This function calls a CusolverMg CUDA kernel to compute the eigenvalues (and optionally eigenvectors) of `a`.
    The input matrix `a` must be 2D and symmetric. The matrix must be sharded along its columns using PartitionSpec P(None, str).
    If `cyclic_1d` is True, the input is assumed to be sharded in a cyclic 1D layout.

    If `a` is not postive-definite, CusolverMg will fail and return an error status.
    If `a` is not symmetric, CusolverMg will fail and return an error status.
    In both cases, the returned result will be NaN.

    Args:
        a: 2D JAX array representing the matrix to decompose. Must be symmetric.
        T_A: Tile size for cyclic 1D layout. Only used if `cyclic_1d` is True.
        mesh: JAX Mesh object describing the device mesh.
        in_specs: PartitionSpec or tuple of PartitionSpec describing the sharding of `a`.
        return_eigenvectors: If True, also computes and returns the eigenvectors.
        cyclic_1d: If True, input arrays are assumed to be sharded in a cyclic 1D manner. If False, arrays are sharded along columns.
        return_status: If True, returns a tuple (..., status), where `status` is an integer indicating the success or failure of the computation.
            For CUDA Toolkit 12.8.0, status codes are defined in `cusolver_common.h`.

    Returns:
        out = (eigenvalues, eigenvectors) if `return_eigenvectors` is True.
        out = eigenvalues if `return_eigenvectors` is False.
        Returns additional `status` if `return_status` is True.

    Raises:
        AssertionError: If `a` is not 2D or if `in_specs` is not of the correct length/type.
        ValueError: If the input array does not have the correct sharding or if T_A > 1024.
    """

    assert a.ndim == 2, "a must be a 2D array."
    if isinstance(in_specs, (tuple, list)):
        assert len(in_specs) == 1, f"expected only one `in_specs`, received {in_specs}"

        (spec_a,) = in_specs
    else:
        spec_a = in_specs

    ndev = jax.local_device_count()

    axis_name = spec_a._partitions[1]

    if (spec_a._partitions[0] != None) or (axis_name == None):
        raise ValueError(
            "A must be sharded along the columns with PartitionSpec P(None, str)."
        )
    if T_A > 1024:
        raise ValueError(
            "T_A has a maximum value of 1024 for SyevdMg, received T_A={T_A}"
        )
    shard_size = a.shape[0] // ndev
    shard_size_needed = shard_size + (T_A - (shard_size % T_A)) % T_A

    if return_eigenvectors:
        target_name = "syevd_mg"
        out_type = (
            jax.ShapeDtypeStruct((a.shape[0],), maybe_real_dtype_from_complex(a.dtype)),
            jax.ShapeDtypeStruct((a.shape[0], shard_size_needed), a.dtype),
            jax.ShapeDtypeStruct((1,), jnp.int32),
        )
        out_specs = (P(), spec_a, P(axis_name))
        output_layouts = ((0,), (1, 0), (0,))

        @partial(
            jax.shard_map,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=False,
        )
        def impl(_a):
            if not cyclic_1d and ndev > 1:
                _a = cyclic_1d_no_shardmap(_a, T_A=T_A, ndev=ndev, axis_name=axis_name)

            _ev, _a, status = jax.ffi.ffi_call(
                target_name,
                out_type,
                input_layouts=((1, 0),),
                output_layouts=output_layouts,
            )(_a, T_A=T_A)

            if not cyclic_1d and ndev > 1:
                _a = undo_cyclic_1d_no_shardmap(_a, T_A=T_A, ndev=ndev, axis_name=axis_name)
            return _ev, _a, status

        eigenvalues, V, status = impl(a)
        if return_status:
            out = (eigenvalues, V[:, : a.shape[0]], status)
        else:
            out = (eigenvalues, V[:, : a.shape[0]])
    else:
        target_name = "syevd_no_V_mg"
        out_type = (
            jax.ShapeDtypeStruct((a.shape[0],), maybe_real_dtype_from_complex(a.dtype)),
            jax.ShapeDtypeStruct((1,), jnp.int32),
        )
        out_specs = (P(), P(axis_name))
        output_layouts = ((0,), (0,))

        @partial(
            jax.shard_map,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=False,
        )
        def impl(_a):
            if not cyclic_1d and ndev > 1:
                _a = cyclic_1d_no_shardmap(_a, T_A=T_A, ndev=ndev, axis_name=axis_name)

            _ev, status = jax.ffi.ffi_call(
                target_name,
                out_type,
                input_layouts=((1, 0),),
                output_layouts=output_layouts,
            )(_a, T_A=T_A)
            return _ev, status

        eigenvalues, status = impl(a)
        if return_status:
            out = (eigenvalues, status)
        else:
            out = eigenvalues
    return out
