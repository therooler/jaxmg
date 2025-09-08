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
    return A @ A.T + jnp.eye(n, dtype=dtype) * 1e-3  # symmetric PSD


def get_mesh_and_spec_from_array(a: Array):
    sharding = a.sharding
    if isinstance(sharding, NamedSharding):
        return sharding.mesh, sharding.spec
    else:
        raise ValueError(
            "Array is not sharded with a NamedSharding, cannot extract mesh and spec."
        )


class JaxMgWarning(UserWarning):
    """Warnings emitted by JaxMg."""
