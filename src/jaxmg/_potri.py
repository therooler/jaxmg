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
from .utils import symmetrize


def potri(
    a: Array,
    T_A: int,
    mesh: Mesh,
    in_specs: Tuple[P] | P,
    cyclic_1d: bool = False,
    return_status: bool = False,
 ) -> Union[Array, Tuple[Array, int]]:
    """
    Computes the inverse of a symmetric positive-definite matrix `a` using a
    distributed multi-GPU CUDA kernel via JAX FFI.

    This function calls a the cuSolverMg CUDA library to compute the matrix inverse.

    The input matrix `a` must be 2D, symmetric, and positive-definite.

    The matrix must be sharded along its columns using PartitionSpec P(None, str).

    If `a` is not postive-definite, CusolverMg will fail and return an error status.
    If `a` is not symmetric, CusolverMg will fail and return an error status.
    In both cases, the returned result will be NaN.

    If `cyclic_1d` is set to True but the input arrays are not sharded in a cyclic 1d manner,
    the data layout will be wrong and the kernel will fail since `a` will likely not be positve definite with the
    given data layout.

    Args:
        a: 2D JAX array representing the matrix to invert. Must be symmetric and positive-definite.
        T_A: Tile size for cyclic 1D layout. Only used if `cyclic_1d` is True.
        mesh: JAX Mesh object describing the device mesh.
        in_specs: PartitionSpec or tuple of PartitionSpec describing the sharding of `a`.
        cyclic_1d: If True, input arrays are assumed to be sharded in a cyclic 1D manner. If False, arrays are sharded along columns.
        return_status: If True, returns a tuple (A_inv, status), where `status` is an integer indicating the success or failure of the computation.
            For CUDA Toolkit 12.8.0, status codes are defined in `cusolver_common.h`.

     Returns:
        If `return_status` is False: The inverse of `a` as a 2D array, sharded columnwise across all devices.
        If `return_status` is True: A tuple (A_inv, status), where `status` is an integer status code from the CUDA kernel.

    Raises:
        AssertionError: If `a` is not 2D or if `in_specs` is not of the correct length/type.
        ValueError: If the input array does not have the correct sharding.
    """

    assert a.ndim == 2, "a must be a 2D array."
    if isinstance(in_specs, (tuple, list)):
        assert len(in_specs) == 1, f"expected only one `in_specs`, received {in_specs}"
        (spec_a,) = in_specs
    else:
        spec_a = in_specs

    ndev = jax.local_device_count()
    if (spec_a._partitions[0] != None) or (spec_a._partitions[1] == None):
        raise ValueError(
            "`a` must be sharded along the columns with PartitionSpec P(None, str)."
        )
    shard_size = a.shape[0] // ndev
    shard_size_needed = shard_size + (T_A - (shard_size % T_A)) % T_A
    
    out_type = (
        jax.ShapeDtypeStruct((a.shape[0], shard_size_needed), a.dtype),
        jax.ShapeDtypeStruct((1,), jnp.int32),
    )
    input_layouts = ((1, 0),)
    output_layouts = ((1, 0), (0,))
    out_specs = (spec_a, P(spec_a._partitions[1]))

    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )
    def impl(_a):
        if not cyclic_1d and ndev > 1:
            _a = cyclic_1d_no_shardmap(_a, T_A=T_A, ndev=ndev, axis_name=spec_a._partitions[1])

        _a, status = jax.ffi.ffi_call(
            "potri_mg",
            out_type,
            input_layouts=input_layouts,
            output_layouts=output_layouts,
        )(_a, T_A=T_A)

        if not cyclic_1d and ndev > 1:
            _a = undo_cyclic_1d_no_shardmap(
                _a, T_A=T_A, ndev=ndev, axis_name=spec_a._partitions[1]
            )
        return _a, status

    out, status = impl(a)
    out = symmetrize(out[:,:a.shape[0]])
    if return_status:
        return out, status[0]
    else:
        return out
