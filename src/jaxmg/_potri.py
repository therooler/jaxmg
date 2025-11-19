import os
import ctypes
import jax
import jax.numpy as jnp

from jax import Array
from jax.sharding import PartitionSpec as P, Mesh

from functools import partial
from typing import Tuple, Union

from ._cyclic_1d import calculate_padding, pad_rows, unpad_rows


def potri(
    a: Array,
    T_A: int,
    mesh: Mesh,
    in_specs: Tuple[P] | P,
    return_status: bool = False,
    pad=True,
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

    ndev = jax.local_device_count()
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
    axis_name = in_specs._partitions[0]
    N_rows, N = a.shape

    shard_size = N_rows // ndev

    input_layouts = ((0, 1),)

    # Calculate padding
    padding = calculate_padding(shard_size, T_A)

    if not pad or padding == 0 or T_A >= N // ndev:
        if T_A < N // ndev:
            assert (
                N_rows == N + ndev * padding
            ), f"pad=False, but with T_A={T_A}, we need padding of {padding} rows per device."
            f"Expected {N + ndev * padding} rows, but received {N_rows}"

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

    out_type = (
        jax.ShapeDtypeStruct((shard_size + padding, N), a.dtype),
        jax.ShapeDtypeStruct((1,), jnp.int32),
    )
    output_layouts = ((0, 1), (0,))
    out_specs = (
        P(axis_name, None),
        P(None),
    )
    # Prepare ffi call
    ffi_fn = partial(
        jax.ffi.ffi_call(
            "potri_mg",
            out_type,
            input_layouts=input_layouts,
            output_layouts=output_layouts,
            input_output_aliases={0: 0},  # Crucial for buffer sharing
        ),
        T_A=T_A,
    )

    # Jit with donate_argnums=0 is crucial for buffer sharing
    @partial(jax.jit, donate_argnums=0)
    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=P(axis_name, None),
        out_specs=out_specs,
        check_vma=False,
    )
    def impl(_a):
        return ffi_fn(_a)

    def fn(_a):
        _a = pad_fn(_a)
        _a, _status = impl(_a)
        return unpad_fn(_a), _status

    out, status = fn(a)
    out = symmetrize(out)
    if return_status:
        return out, status[0]
    else:
        return out


@partial(jax.jit, donate_argnums=0)
def symmetrize(_a):
    _a = jnp.triu(_a)
    return _a + _a.T.conj() - jnp.diag(jnp.diag(_a))
