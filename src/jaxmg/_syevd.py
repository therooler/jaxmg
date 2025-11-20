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

from .utils import maybe_real_dtype_from_complex
from ._cyclic_1d import calculate_padding, pad_rows, unpad_rows


def syevd(
    a: Array,
    T_A: int,
    mesh: Mesh,
    in_specs: Tuple[P],
    return_eigenvectors: bool = True,
    return_status: bool = False,
    pad=True,
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
    ndev =int(os.environ["JAXMG_NUMBER_OF_DEVICES"])
    # Normalize in_specs so it's a single PartitionSpec instance (not an iterable)
    if isinstance(in_specs, (list, tuple)):
        if len(in_specs) != 1:
            raise ValueError(
                "in_specs must be a single PartitionSpec or a 1-element list/tuple."
            )
        in_specs = in_specs[0]
    if not isinstance(in_specs, P):
        raise TypeError(
            "in_specs must be a PartitionSpec or a 1-element list/tuple containing one."
        )
    if (in_specs._partitions[1] != None) or (in_specs._partitions[0] == None):
        raise ValueError(
            "A must be sharded along the rows with PartitionSpec P(str, None)."
        )
    assert a.ndim == 2, "a must be a 2D array."
    if T_A > 1024:
        raise ValueError(
            "T_A has a maximum value of 1024 for SyevdMg, received T_A={T_A}"
        )
    axis_name = in_specs._partitions[0]
    N_rows, N = a.shape

    shard_size = N_rows // ndev

    padding = calculate_padding(shard_size, T_A)
    input_layouts = ((0, 1),)
    if not pad or padding == 0 or T_A >= N // ndev:
        if T_A < N // ndev:
            assert (
                N_rows == N + ndev * padding
            ), f"pad=False, but with T_A={T_A}, we need padding of {padding} rows per device."
            f"Expected {N + ndev * padding} rows, but received {N_rows}"

        # Identity padding
        pad_fn = lambda _a: _a
        unpad_fn = lambda _a: _a
        padding = 0

    else:
        # Make padding fns
        pad_fn = jax.shard_map(
            partial(pad_rows, padding=padding),
            mesh=mesh,
            in_specs=P(axis_name, None),
            out_specs=P(axis_name, None),
            check_vma=True,
        )
        unpad_fn = jax.shard_map(
            partial(unpad_rows, padding=padding),
            mesh=mesh,
            in_specs=P(axis_name, None),
            out_specs=P(axis_name, None),
            check_vma=True,
        )

    if return_eigenvectors:
        target_name = "syevd_mg"
        out_type = (
            jax.ShapeDtypeStruct((N,), maybe_real_dtype_from_complex(a.dtype)),
            jax.ShapeDtypeStruct((shard_size + padding, N), a.dtype),
            jax.ShapeDtypeStruct((1,), jnp.int32),
        )
        output_layouts = ((0,), (0, 1), (0,))
        out_specs = (P(None), P(axis_name, None), P(None))

    else:
        target_name = "syevd_no_V_mg"
        out_type = (
            jax.ShapeDtypeStruct((N,), maybe_real_dtype_from_complex(a.dtype)),
            jax.ShapeDtypeStruct((1,), jnp.int32),
        )
        output_layouts = ((0,), (0,))
        out_specs = (P(None), P(None))

    ffi_fn = partial(
        jax.ffi.ffi_call(
            target_name,
            out_type,
            input_layouts=input_layouts,
            output_layouts=output_layouts,
            input_output_aliases={0: 1} if return_eigenvectors else None,
        ),
        T_A=T_A,
    )

    @partial(jax.jit, donate_argnums=0)
    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )
    def impl(_a):
        return ffi_fn(_a)
    
    def fn(_a):
        _a = pad_fn(_a)
        if target_name == "syevd_mg":
            _ev, _V, _status = impl(_a)
            return _ev, unpad_fn(_V), _status
        else:
            _ev, _status = impl(_a)
            return _ev, _status

    out = fn(a)
    if return_status:
        status = out[-1]
        return *out[:-1], status[0]
    else:
        if len(out) == 3:
            return out[:-1]
        else:
            return out[0]
