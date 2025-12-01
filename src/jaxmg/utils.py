import socket
import os
import hashlib
import warnings
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax import Array
from jax.experimental import multihost_utils as mh


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

class JaxMgWarning(UserWarning):
    """Warnings emitted by JaxMg."""

def numeric_machine_key():
    # 128-bit hash of hostname
    h = hashlib.blake2b(socket.gethostname().encode(), digest_size=16).digest()
    hi = int.from_bytes(h[:8], "big")
    lo = int.from_bytes(h[8:], "big")
    return jnp.array([hi, lo], dtype=jnp.uint64)

def determine_distributed_setup():
    n_proc = jax.process_count()
    all_ids = mh.process_allgather(
        numeric_machine_key()
    )  # list of len = num_processes, ordered by process_index
    all_ids = jnp.atleast_2d(all_ids)
    unique_ids = set(str(row[0])+str(row[1]) for row in all_ids)
    num_machines = len(unique_ids)
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES","")
    n_visisble_devices = len(cuda_visible_devices.split(","))
    n_devices = jax.device_count()
    n_devices_per_machine = n_devices // num_machines
    n_devices_per_process = n_devices // n_proc
    n_local_devices = jax.local_device_count()
    configuration_str = (
        f"\t{num_machines} machine(s), {n_proc} process(es), "
        f"{n_local_devices} local device(s), {n_visisble_devices} visible device(s)"
    )
    if n_proc == num_machines:
        if n_local_devices != n_visisble_devices:
            raise ValueError(
                f"Invalid distributed configuration detected: \n {configuration_str}\n"
                "The number of local devices cannot differ from the visible devices in SPMD mode"
            )
        mode = "SPMD"
    elif n_devices_per_process == 1:
        mode = "MPMD"
    else:
        raise ValueError(
            f"Invalid distributed configuration detected: \n {configuration_str}\n"
            "JAXMg requires either a single process per machine (SPMD)"
            "or a single process for each GPU (MPMD)"
        )

    return num_machines, n_devices_per_machine, n_devices_per_process, mode
