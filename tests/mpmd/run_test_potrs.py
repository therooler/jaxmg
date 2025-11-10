import os
import sys
import json
import traceback
from typing import Callable, Dict, List

import jax
coord_addr = sys.argv[1]
proc_id = int(sys.argv[2])
num_procs = int(sys.argv[3])
selected_csv = sys.argv[4] if len(sys.argv) > 4 else ""  # empty -> all

# Initialize the GPU machines.
jax.distributed.initialize(
    coordinator_address=coord_addr,
    num_processes=num_procs,
    process_id=proc_id,
    local_device_ids=proc_id,
)
# Basic diagnostics for debugging
print("process id =", jax.process_index(), flush=True)
print("global devices =", jax.devices(), flush=True)
print("local devices =", jax.local_devices(), flush=True)
print("visible devices", os.environ.get("CUDA_VISIBLE_DEVICES", ""), flush=True)

def _println(prefix: str, payload: dict):
    """Print a single-line JSON payload with a stable prefix for log parsing."""
    print(f"{prefix} {json.dumps(payload, sort_keys=True)}", flush=True)

    
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from jaxmg import potrs
from functools import partial
from jaxmg.utils import random_psd
from jax.experimental import multihost_utils as mu
from itertools import product


# These will be initialized after jax.distributed.initialize()
devices = None
mesh = None

def cusolver_solve_arange(N, T_A, dtype):
    A = jnp.diag(jnp.arange(N, dtype=dtype) + 1)
    b = jnp.ones((N, 1), dtype=dtype)
    # Make mesh and place data
    A = jax.device_put(A, NamedSharding(mesh, P(None, "x")))
    b = jax.device_put(b, NamedSharding(mesh, P(None, None)))

    out = jax.jit(
        partial(potrs, mesh=mesh, in_specs=(P(None, "x"), P(None, None))),
        static_argnums=2,
    )(A, b, T_A)
    expected_out = 1.0 / (jnp.arange(N, dtype=dtype) + 1)
    assert jnp.allclose(out.flatten(), expected_out)
    print("Passed cusolver_solve_arange")

def cusolver_solve_non_psd(N, T_A, dtype):
    A = jnp.diag(jnp.arange(N, dtype=dtype) - 1)
    b = jnp.ones((N, 1), dtype=dtype)
    # Make mesh and place data
    A = jax.device_put(A, NamedSharding(mesh, P(None, "x")))
    b = jax.device_put(b, NamedSharding(mesh, P(None, None)))

    out, status= jax.jit(
        partial(potrs, mesh=mesh, in_specs=(P(None, "x"), P(None, None)), return_status=True),
        static_argnums=2,
    )(A, b, T_A)
    status.block_until_ready()
    out.block_until_ready()
    assert status==7
    assert jnp.all(jnp.isnan(out))
    print("Passed cusolver_solve_non_psd")

def cusolver_solve_non_symm(N, T_A, dtype):
    A = jnp.diag(jnp.arange(N, dtype=dtype) + 1)
    #TODO: For some reason the solver does not fail when we set this to 1.0.
    A = A.at[1,0].set(2.0)
    b = jnp.ones((N, 1), dtype=dtype)
    # Make mesh and place data
    A = jax.device_put(A, NamedSharding(mesh, P(None, "x")))
    b = jax.device_put(b, NamedSharding(mesh, P(None, None)))

    out, status= jax.jit(
        partial(potrs, mesh=mesh, in_specs=(P(None, "x"), P(None, None)), return_status=True),
        static_argnums=2,
    )(A, b, T_A)
    status.block_until_ready()
    out.block_until_ready()
    assert status==7
    assert jnp.all(jnp.isnan(out))
    print("Passed cusolver_solve_non_symm")

def cusolver_solve_psd(N, T_A, dtype):
    A = random_psd(N, dtype=dtype, seed=1234)
    b = jnp.ones((N, 1), dtype=dtype)
    cfac = jax.scipy.linalg.cho_factor(A)
    expected_out = jax.scipy.linalg.cho_solve(cfac, b)
    # Make mesh and place data
    _A = jax.device_put(A, NamedSharding(mesh, P(None, "x")))
    _b = jax.device_put(b, NamedSharding(mesh, P(None, None)))

    out = jax.jit(
        partial(potrs, mesh=mesh, in_specs=(P(None, "x"), P(None, None))),
        static_argnums=2,
    )(_A, _b, T_A)
    norm_scipy = jnp.linalg.norm(b - A @ expected_out)
    norm_potrf = jnp.linalg.norm(b - A @ out)
    assert jnp.isclose(norm_scipy, norm_potrf, rtol=10, atol=0.0)
    print("Passed cusolver_solve_psd")

def cusolver_solve_psd_sol_copy(N, T_A, dtype):
    A = random_psd(N, dtype=dtype, seed=1234)
    b = jnp.ones((N, 1), dtype=dtype)
    # Make mesh and place data
    _A = jax.device_put(A, NamedSharding(mesh, P(None, "x")))
    _b = jax.device_put(b, NamedSharding(mesh, P(None, None)))

    out = jax.jit(
        partial(potrs, mesh=mesh, in_specs=(P(None, "x"), P(None, None))),
        static_argnums=2,
    )(_A, _b, T_A)
    str_shards = []
    for shard in out.addressable_shards:
        str_shards.append(str(shard.data))
    print(str_shards)
    assert all(l == str_shards[0] for l in str_shards)
    print("Passed cusolver_solve_psd_sol_copy")


def _build_registry() -> Dict[str, Callable[[int, int, jnp.dtype], None]]:
    # Map test names to callables that accept (N, T_A, dtype)
    return {
        "arange": cusolver_solve_arange,
        "non_psd": cusolver_solve_non_psd,
        "non_symm": cusolver_solve_non_symm,
        "psd": cusolver_solve_psd,
        "psd_sol_copy": cusolver_solve_psd_sol_copy,
    }


def main(argv: List[str]):
    # Expected args: coord_addr, proc_id, num_procs, [tests_csv]
    if len(argv) < 4:
        print("Usage: run_test_potrs.py <coord_addr> <proc_id> <num_procs> [tests_csv]", flush=True)
        sys.exit(2)

    # Prepare global device topology now that distributed is active
    global devices, mesh
    devices = [d for d in jax.devices() if d.platform == "gpu"]
    mesh = jax.make_mesh((jax.device_count(),), ("x",))

    registry = _build_registry()
    selected = [s for s in (t.strip() for t in selected_csv.split(",")) if s] if selected_csv else list(registry.keys())

    # Parameter grids per requested process count
    # dtype strings for readability in logs
    dtypes = [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128]
    if num_procs == 1:
        Ns = [4, 8, 10, 12]
        TAs = [1, 2, 3]
    elif num_procs == 2:
        Ns = [8, 10, 12]
        TAs = [1, 2, 3]
    elif num_procs == 4:
        Ns = [48, 60]
        TAs = [1, 2, 4]

    params_summary = {
        "N": Ns,
        "T_A": TAs,
        "dtype": [jnp.dtype(dt).name for dt in dtypes],
    }

    # Report discovered/selected tests and parameter grids
    _println(
        "MPTEST_DISCOVER",
        {
            "proc": proc_id,
            "available": sorted(registry.keys()),
            "selected": selected,
            "params": params_summary,
        },
    )

    n_ok = 0
    n_fail = 0

    ndev = jax.device_count()
    for name in selected:
        fn = registry.get(name)
        if fn is None:
            _println(
                "MPTEST_RESULT",
                {"proc": proc_id, "name": name, "status": "skip", "message": f"unknown test '{name}'"},
            )
            continue
        for N, T_A, dt in product(Ns, TAs, dtypes):
            dtype_name = jnp.dtype(dt).name
            try:
                fn(N, T_A, dt)
                _println(
                    "MPTEST_RESULT",
                    {"proc": proc_id, "name": name, "status": "ok", "params": {"N": N, "T_A": T_A, "dtype": dtype_name}},
                )
                n_ok += 1
            except Exception:
                tb = traceback.format_exc(limit=40)
                _println(
                    "MPTEST_RESULT",
                    {
                        "proc": proc_id,
                        "name": name,
                        "status": "fail",
                        "params": {"N": N, "T_A": T_A, "dtype": dtype_name},
                        "traceback": tb,
                    },
                )
                n_fail += 1
                _println("MPTEST_SUMMARY", {"proc": proc_id, "ok": n_ok, "fail": n_fail, "total": n_ok + n_fail})

                return 1
                

    _println("MPTEST_SUMMARY", {"proc": proc_id, "ok": n_ok, "fail": n_fail, "total": n_ok + n_fail})

    # Don't raise here; we let the pytest parent parse logs and assert
    # Exiting 0 ensures all processes finish and results are captured.
    return 0


main(sys.argv)
