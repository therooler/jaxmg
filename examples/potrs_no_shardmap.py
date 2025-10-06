import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jax.sharding import PartitionSpec as P, NamedSharding
from jaxmg import potrs_no_shardmap

from functools import partial


def main():
    # Assumes we have at least one GPU available
    devices = jax.devices("gpu")
    assert len(devices) in [1, 2], "Example only works for 1 or 2 devices"
    N = 8
    T_A = 2
    dtype = jnp.complex64
    # Create diagonal matrix and `b` all equal to one
    A = jnp.diag(jnp.arange(N, dtype=dtype) + 1)
    b = jnp.ones((N, 1), dtype=dtype)
    ndev = jax.device_count()
    # Make mesh and place data (columns sharded)
    mesh = jax.make_mesh((ndev,), ("x",))
    A = jax.device_put(A, NamedSharding(mesh, P(None, "x")))
    b = jax.device_put(b, NamedSharding(mesh, P(None, None)))
    diag_shift = 1e-1

    @partial(jax.jit, static_argnames=("_T_A",))
    def shift_and_solve(_a, _b, _ds, _T_A):
        idx = jnp.arange(_a.shape[0])
        shard_size = _a.shape[1]
        # Add shift based on index.
        _a = _a.at[idx + shard_size * jax.lax.axis_index("x"), idx].add(_ds)
        jax.debug.print("dev{}:_a=\n{}\n", jax.lax.axis_index("x"), _a)
        # Call solver in shardmap context
        return potrs_no_shardmap(_a, _b, _T_A)

    @partial(jax.jit, static_argnames=("_T_A",))
    def jitted_potrs(_a, _b, _ds, _T_A):
        out = jax.shard_map(
            partial(shift_and_solve, _T_A=_T_A),
            mesh=mesh,
            in_specs=(P(None, "x"), P(None, None), P()),
            out_specs=(P(None, None), P(None)),
            check_vma=False
        )(_a, _b, _ds)
        return out
    out, status = jitted_potrs(A, b, diag_shift, T_A)
    print(f"Status: {status}")
    expected_out = 1.0 / (jnp.arange(N, dtype=dtype) + 1+diag_shift)
    return jnp.allclose(out.flatten(), expected_out)

if __name__ == "__main__":
    import sys

    sys.exit(0 if main() else 1)  # 0 on success, 1 on failure
