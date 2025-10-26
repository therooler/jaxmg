import os

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax import Array


def random_psd(n, dtype, seed):
    """
    Generate a random n x n positive semidefinite matrix.
    """
    key = jax.random.key(seed)
    A = jax.random.normal(key, (n, n), dtype=dtype) / jnp.sqrt(n)
    return A @ A.T.conj() + jnp.eye(n, dtype=dtype) * 1e-5  # symmetric PSD


def get_mesh_and_spec_from_array(a: Array):
    sharding = a.sharding
    if isinstance(sharding, NamedSharding):
        return sharding.mesh, sharding.spec
    else:
        raise ValueError(
            "Array is not sharded with a NamedSharding, cannot extract mesh and spec."
        )


def maybe_real_dtype_from_complex(dtype):
    return (
        jnp.float32
        if dtype == jnp.complex64
        else (jnp.float64 if dtype == jnp.complex128 else dtype)
    )


def symmetrize(_a):
    _a = jnp.tril(_a)
    return _a + _a.T.conj() - jnp.diag(jnp.diag(_a))


class JaxMgWarning(UserWarning):
    """Warnings emitted by JaxMg."""


def determine_distributed_setup():
    n_proc = jax.process_count()
    n_devices_per_node = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    n_devices = jax.device_count()
    n_nodes = n_devices // n_devices_per_node
    n_devices_per_process = n_devices // n_proc
    if n_devices_per_process==n_devices_per_node:
        mode = "SPMD"
    elif n_devices_per_process==1: 
        mode = "MPMD"
    else:
        return n_nodes, n_devices_per_process, "UNKNOWN"
    
    return n_nodes, n_devices_per_node, n_devices_per_process, mode
