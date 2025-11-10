import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding
from jaxmg import cyclic_1d_layout


def main():
    # Assumes we have at least one GPU available
    devices = jax.devices("gpu")
    assert len(devices) in [1, 2], "Example only works for 1 or 2 devices"
    N = 12
    T_A = 2
    dtype = jnp.float32
    ndev = jax.device_count()
    # Create diagonal matrix and `b` all equal to one
    A = jnp.diag(jnp.arange(N, dtype=dtype) + 1)
    mesh = jax.make_mesh((ndev,), ("x",))
    A = jax.device_put(A, NamedSharding(mesh, P(None, "x")))

    for dev, shard in enumerate(A.addressable_shards):
        print(f"dev {dev}: shard\n {shard.data}")
    A_bc = cyclic_1d_layout(A, T_A)

    for shard in A_bc.addressable_shards:
        print(f"dev {dev}: shard\n {shard.data}")

    return True

if __name__ == "__main__":
    import sys

    sys.exit(0 if main() else 1)  # 0 on success, 1 on failure
