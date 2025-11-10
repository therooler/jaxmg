import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding
from jaxmg import potrs

def main():
    # Assumes we have at least one GPU available
    devices = jax.devices("gpu")
    assert len(devices) in [1, 2], "Example only works for 1 or 2 devices"
    N = 12
    T_A = 2
    dtype = jnp.float64
    # Create diagonal matrix and `b` all equal to one
    A = jnp.diag(jnp.arange(N, dtype=dtype) + 1)
    b = jnp.ones((N, 1), dtype=dtype)
    ndev = jax.device_count()
    # Make mesh and place data (columns sharded)
    mesh = jax.make_mesh((ndev,), ("x",))
    A = jax.device_put(A, NamedSharding(mesh, P(None, "x")))
    b = jax.device_put(b, NamedSharding(mesh, P(None, None)))
    # Call potrf
    out = potrs(A, b, T_A=T_A, mesh=mesh, in_specs=(P(None, "x"), P(None, None)))
    expected_out = 1.0 / (jnp.arange(N, dtype=dtype) + 1)
    return jnp.allclose(out.flatten(), expected_out)

if __name__=="__main__":
    import sys
    sys.exit(0 if main() else 1)   # 0 on success, 1 on failure
