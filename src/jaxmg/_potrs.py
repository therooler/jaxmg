import os
import ctypes
import jax

import jax.numpy as jnp
from jax import Array
from jax.sharding import PartitionSpec as P, Mesh

from typing import Tuple, List, Union
from functools import partial

from ._cyclic_1d import calculate_padding, pad_rows, unpad_rows

SHARED_LIBRARY_POTRS = os.path.join(os.path.dirname(__file__), "bin/libpotrs.so")
library_potrs = ctypes.cdll.LoadLibrary(SHARED_LIBRARY_POTRS)

# Register FFI targets
jax.ffi.register_ffi_target(
    "potrs_mg", jax.ffi.pycapsule(library_potrs.PotrsMgFFI), platform="CUDA"
)


def potrs(
    a: Array,
    b: Array,
    T_A: int,
    mesh: Mesh,
    in_specs: Tuple[P] | List[P],
    return_status: bool = False,
    pad=True,
) -> Union[Array, Tuple[Array, int]]:
    """
    Solve a * x = b using a multi-GPU Cholesky solve (cuSolverMg) via JAX FFI.

    This wrapper prepares inputs, optionally pads per-device tiles, and calls a
    CUDA shared library through jax.ffi. The heavy work runs on GPUs; this
    function handles sharding, padding, and remapping.

    Parameters
    ----------
    a : Array
        2D, symmetric positive-definite matrix. Expected to be sharded across the
        mesh with PartitionSpec P(<axis_name>, None) where <axis_name> is a named
        mesh axis (see in_specs). The function asserts a.ndim == 2.
    b : Array
        2D right-hand side, replicated across devices with PartitionSpec P(None, None).
        The function asserts b.ndim == 2 and a.shape[1] == b.shape[0].
    T_A : int
        Tile size used for the 1D block-cyclic layout required by cuSolverMg.
        Determines per-device padding.
    mesh : Mesh
        JAX Mesh describing the devices used by shard_map.
    in_specs : Tuple[PartitionSpec, PartitionSpec]
        Expected tuple describing sharding of (a, b). This function requires
        in_specs == (P(<axis_name>, None), P(None, None)). The named axis is used
        for shard_map and to compute per-device shard_size.
    return_status : bool, optional
        If True, return (x, status) where status is a host-replicated int32 scalar
        returned from the native solver. If False, return x only.
    pad : bool, optional
        If True (default) a is padded per-device to satisfy the tile size T_A.
        If False, the caller must ensure the global shape already matches required padding.

    Returns
    -------
    Array or (Array, int)
        The solution x (replicated across devices). If return_status is True, also
        returns the solver status (int32 scalar).

    Notes
    -----
    - The implementation uses shard_map + a jitted inner function that donates
      the 'a' buffer to enable zero-copy buffer sharing with the native library.
    - Inputs/outputs use column-major-friendly layouts expected by cuSolverMg.
    - If the native solver fails, the returned solution may contain NaNs and
      the status will be non-zero.
    - This function validates in_specs and will raise ValueError for unsupported sharding.
    """

    ndev = jax.local_device_count()

    assert isinstance(
        in_specs, (tuple, list)
    ), f"expected `in_specs` to be a tuple or list of `PartitionSpec` objects, received {in_specs}"
    assert len(in_specs) == 2, f"expected two `in_specs`, received {in_specs}"

    spec_a, spec_b = in_specs
    if (spec_a._partitions[0] == None) or (spec_a._partitions[1] != None):
        raise ValueError(
            "A must be sharded along the columns with PartitionSpec P(None, str)."
        )
    if spec_b != P(None, None):
        raise ValueError(
            "b must be replicated along all shards with PartitionSpec P(None, None)."
        )

    assert a.shape[1] == b.shape[0], "A and b must have the same number of columns."
    assert a.ndim == 2, "a must be a 2D array."
    assert b.ndim == 2, "b must be a 2D array."
    N_rows, N = a.shape
    axis_name = spec_a._partitions[0]
    shard_size = N_rows // ndev

    # Keep b in column-major layout
    input_layouts = ((0, 1), (1, 0))
    output_layouts = ((1, 0), (0,))

    padding = calculate_padding(shard_size, T_A)
    out_type = (
        jax.ShapeDtypeStruct(b.shape, b.dtype),
        jax.ShapeDtypeStruct((1,), jnp.int32),
    )

    # Prepare ffi call
    ffi_fn = partial(
        jax.ffi.ffi_call(
            "potrs_mg",
            out_type,
            input_layouts=input_layouts,
            output_layouts=output_layouts,
        ),
        T_A=T_A,
    )

    # Jit with donate_argnums=0 is crucial for buffer sharing
    @partial(jax.jit, donate_argnums=0)
    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(P(axis_name, None), P(None, None)),
        out_specs=(P(None, None), P(None)),
        check_vma=False,
    )
    def impl(_a, _b):
        _a, _status = ffi_fn(_a, _b)
        return _a, _status

    if not pad or padding == 0 or T_A >= N // ndev:
        if T_A < N // ndev:
            assert (
                N_rows == N + ndev * padding
            ), f"pad=False, but with T_A={T_A}, we need padding of {padding} rows per device."
            f"Expected {N + ndev * padding} rows, but received {N_rows}"
        # Identity padding
        pad_fn = lambda _a: _a

    else:
        # Make padding fns
        pad_fn = jax.shard_map(
            partial(pad_rows, padding=padding),
            mesh=mesh,
            in_specs=P(axis_name, None),
            out_specs=P(axis_name, None),
            check_vma=True,
        )

    def fn(_a, _b):
        _a = pad_fn(_a)
        _out, _status = impl(_a, _b)
        return _out, _status

    out, status = fn(a, b)
    if return_status:
        return out, status[0]
    else:
        return out


def potrs_no_shardmap():
    pass


# def potrs_no_shardmap(
#     a: Array, b: Array, T_A: int, cyclic_1d: bool = False, axis_name="x"
# ) -> Union[Array, Tuple[Array, int]]:
#     """
#     Solves the linear system `a * x = b` for `x` using the Cholesky decomposition of a symmetric positive-definite matrix `a`
#     on multiple GPUs via a distributed CUDA kernel through JAX FFI.

#     This function calls a CusolverMg CUDA kernel to compute the solution.
#     The input matrix `a` must be 2D, symmetric, and positive-definite,
#     and the right-hand side `b` must be a 2D array with the same number of rows as `a`.

#     The matrix `a` must be sharded along its columns using PartitionSpec P(None, str),
#     and `b` must be replicated across all devices using PartitionSpec P(None, None).
#     If `cyclic_1d` is True, the input arrays are assumed to be sharded in a cyclic 1D layout.

#     If `a` is not positive-definite or not symmetric, CusolverMg will fail and
#     return an error status. In both cases, the returned result will be NaN.

#     Args:
#         a: 2D JAX array representing the matrix to decompose. Must be symmetric and positive-definite.
#         b: 2D JAX array representing the right-hand side of the linear system. Must have the same number of rows as `a`.
#         T_A: Tile size for cyclic 1D layout. Only used if `cyclic_1d` is True.
#         cyclic_1d: If True, input arrays are assumed to be sharded in a cyclic 1D manner.
#             If False, arrays are sharded along columns for `a` and replicated for `b`.

#     Returns:
#         The solution `x` as a 2D array, replicated across all devices.
#         The status of the solver, replicated across all devices.

#     Raises:
#         AssertionError: If `a` or `b` are not 2D, if their shapes do not match, or if `in_specs` is not of the correct length/type.
#         ValueError: If the input arrays do not have the correct sharding.
#     """
#     assert a.shape[0] == b.shape[0], "A and b must have the same number of rows."
#     assert a.ndim == 2, "a must be a 2D array."
#     assert b.ndim == 2, "b must be a 2D array."

#     ndev = jax.local_device_count()
#     input_layouts = (
#         (1, 0),
#         (1, 0),
#     )
#     output_layouts = ((1, 0), (0,))
#     out_type = (
#         jax.ShapeDtypeStruct(b.shape, b.dtype),
#         jax.ShapeDtypeStruct((1,), jnp.int32),
#     )

#     def impl(_a, _b):
#         if not cyclic_1d and ndev > 1:
#             _a = cyclic_1d_no_shardmap(_a, T_A=T_A, ndev=ndev, axis_name=axis_name)
#         _b, status = jax.ffi.ffi_call(
#             "potrs_mg",
#             out_type,
#             input_layouts=input_layouts,
#             output_layouts=output_layouts,
#         )(_a, _b, T_A=T_A)
#         return _b, status

#     return impl(a, b)
