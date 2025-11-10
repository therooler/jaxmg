import os
import ctypes
import jax

import jax.numpy as jnp
from jax import Array
from jax.sharding import PartitionSpec as P, Mesh

from typing import Tuple, List, Union
from functools import partial

from ._cyclic_1d import cyclic_1d_no_shardmap


def potrs(
    a: Array,
    b: Array,
    T_A: int,
    mesh: Mesh,
    in_specs: Tuple[P] | List[P],
    cyclic_1d: bool = False,
    return_status: bool = False,
 ) -> Union[Array, Tuple[Array, int]]:
    """
    Solves the linear system `a * x = b` for `x` using the Cholesky decomposition of a symmetric positive-definite matrix `a`
    on multiple GPUs via a distributed CUDA kernel through JAX FFI.

    This function calls a CusolverMg CUDA kernel to compute the solution.
    The input matrix `a` must be 2D, symmetric, and positive-definite,
    and the right-hand side `b` must be a 2D array with the same number of rows as `a`.

    The matrix `a` must be sharded along its columns using PartitionSpec P(None, str),
    and `b` must be replicated across all devices using PartitionSpec P(None, None).
    If `cyclic_1d` is True, the input arrays are assumed to be sharded in a cyclic 1D layout.

    If `a` is not positive-definite or not symmetric, CusolverMg will fail and
    return an error status. In both cases, the returned result will be NaN.

    Args:
        a: 2D JAX array representing the matrix to decompose. Must be symmetric and positive-definite.
        b: 2D JAX array representing the right-hand side of the linear system. Must have the same number of rows as `a`.
        T_A: Tile size for cyclic 1D layout. Only used if `cyclic_1d` is True.
        mesh: JAX Mesh object describing the device mesh.
        in_specs: Tuple of PartitionSpec describing the sharding of `a` and `b`.
        cyclic_1d: If True, input arrays are assumed to be sharded in a cyclic 1D manner.
            If False, arrays are sharded along columns for `a` and replicated for `b`.
        return_status: If True, returns a tuple (x, status), where `status` is an integer
            indicating the success or failure of the computation. For CUDA Toolkit 12.8.0,
            status codes are defined in `cusolver_common.h`.

    Returns:
        If `return_status` is False: The solution `x` as a 2D array, replicated across all devices.
        If `return_status` is True: A tuple (x, status), where `status` is an integer status code from the CUDA kernel.

    Raises:
        AssertionError: If `a` or `b` are not 2D, if their shapes do not match, or if `in_specs` is not of the correct length/type.
        ValueError: If the input arrays do not have the correct sharding.
    """

    assert a.shape[0] == b.shape[0], "A and b must have the same number of rows."
    assert a.ndim == 2, "a must be a 2D array."
    assert b.ndim == 2, "b must be a 2D array."

    assert isinstance(
        in_specs, (tuple, list)
    ), f"expected `in_specs` to be a tuple or list of `PartitionSpec` objects, received {in_specs}"
    assert len(in_specs) == 2, f"expected two `in_specs`, received {in_specs}"

    spec_a, spec_b = in_specs
    ndev = int(os.environ["JAXMG_NUMBER_OF_DEVICES"])
    if (spec_a._partitions[0] != None) or (spec_a._partitions[1] == None):
        raise ValueError(
            "A must be sharded along the columns with PartitionSpec P(None, str)."
        )
    if spec_b != P(None, None):
        raise ValueError(
            "b must be replicated along all shards with PartitionSpec P(None, None)."
        )
    out_type = (
        jax.ShapeDtypeStruct(b.shape, b.dtype),
        jax.ShapeDtypeStruct((1,), jnp.int32),
    )
    input_layouts = (
        (1, 0),
        (1, 0),
    )
    output_layouts = ((1, 0), (0,))
    out_specs = (spec_b, P(spec_a._partitions[1]))

    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )
    def impl(_a, _b):
        if not cyclic_1d and ndev > 1:
            _a = cyclic_1d_no_shardmap(
                _a, T_A=T_A, ndev=ndev, axis_name=spec_a._partitions[1]
            )
        _out, status = jax.ffi.ffi_call(
            "potrs_mp_mg",
            out_type,
            input_layouts=input_layouts,
            output_layouts=output_layouts,
        )(_a, _b, T_A=T_A)
        return _out, status

    out, status = impl(a, b)
    if return_status:
        return out, status[0]
    else:
        return out


def potrs_no_shardmap(
    a: Array, b: Array, T_A: int, cyclic_1d: bool = False, axis_name="x"
 ) -> Union[Array, Tuple[Array, int]]:
    """
    Solves the linear system `a * x = b` for `x` using the Cholesky decomposition of a symmetric positive-definite matrix `a`
    on multiple GPUs via a distributed CUDA kernel through JAX FFI.

    This function calls a CusolverMg CUDA kernel to compute the solution.
    The input matrix `a` must be 2D, symmetric, and positive-definite,
    and the right-hand side `b` must be a 2D array with the same number of rows as `a`.

    The matrix `a` must be sharded along its columns using PartitionSpec P(None, str),
    and `b` must be replicated across all devices using PartitionSpec P(None, None).
    If `cyclic_1d` is True, the input arrays are assumed to be sharded in a cyclic 1D layout.

    If `a` is not positive-definite or not symmetric, CusolverMg will fail and
    return an error status. In both cases, the returned result will be NaN.

    Args:
        a: 2D JAX array representing the matrix to decompose. Must be symmetric and positive-definite.
        b: 2D JAX array representing the right-hand side of the linear system. Must have the same number of rows as `a`.
        T_A: Tile size for cyclic 1D layout. Only used if `cyclic_1d` is True.
        cyclic_1d: If True, input arrays are assumed to be sharded in a cyclic 1D manner.
            If False, arrays are sharded along columns for `a` and replicated for `b`.

    Returns:
        The solution `x` as a 2D array, replicated across all devices.
        The status of the solver, replicated across all devices.

    Raises:
        AssertionError: If `a` or `b` are not 2D, if their shapes do not match, or if `in_specs` is not of the correct length/type.
        ValueError: If the input arrays do not have the correct sharding.
    """
    assert a.shape[0] == b.shape[0], "A and b must have the same number of rows."
    assert a.ndim == 2, "a must be a 2D array."
    assert b.ndim == 2, "b must be a 2D array."

    ndev = int(os.environ["JAXMG_NUMBER_OF_DEVICES"])
    input_layouts = (
        (1, 0),
        (1, 0),
    )
    output_layouts = ((1, 0), (0,))
    out_type = (
        jax.ShapeDtypeStruct(b.shape, b.dtype),
        jax.ShapeDtypeStruct((1,), jnp.int32),
    )

    def impl(_a, _b):
        if not cyclic_1d and ndev > 1:
            _a = cyclic_1d_no_shardmap(_a, T_A=T_A, ndev=ndev, axis_name=axis_name)
        _b, status = jax.ffi.ffi_call(
            "potrs_mp_mg",
            out_type,
            input_layouts=input_layouts,
            output_layouts=output_layouts,
        )(_a, _b, T_A=T_A)
        return _b, status

    return impl(a, b)
