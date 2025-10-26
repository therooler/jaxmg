import importlib
import pathlib
import ctypes
import warnings
import sys

from .utils import JaxMgWarning

if not sys.platform.startswith("linux"):
    raise RuntimeError(f"Unsupported platform {sys.platform}, only Linux is supported.")


def _load(module, libraries):
    try:
        m = importlib.import_module(f"nvidia.{module}")
    except ImportError:
        m = None

    for lib in libraries:
        if m is not None:
            path = pathlib.Path(m.__path__[0]) / "lib" / lib
            try:
                ctypes.cdll.LoadLibrary(path)
                continue
            except OSError as e:
                raise OSError(
                    f"Unable to load CUDA library {lib}, is jax built with GPU support?"
                ) from e


# _load("cuda", ["libcuda.so.1"])
_load("cusolver", ["libcusolver.so.11"])
_load("cusolver", ["libcusolverMg.so.11"])

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from .utils import determine_distributed_setup

if any("gpu" == d.platform for d in jax.devices()):

    n_nodes, n_devices_per_node, n_devices_per_process, mode = (
        determine_distributed_setup()
    )
    if n_nodes > 1:
        warnings.warn(
            f"Computation seems to be running on multiple nodes.\n"
            "Ensure that jaxmg is only called over a local device mesh, otherwise process might hang.\n"
            "See examples for how this can be safely achieved.",
            JaxMgWarning,
            stacklevel=2,
        )
    if mode == "SPMD":
        from .potrs import potrs, potrs_no_shardmap
        from .potri import potri
        from .syevd import syevd
    elif mode == "MPMD":
        from .potrs_mp import potrs, potrs_no_shardmap
    else:
        raise ValueError(
            f"You have {n_nodes} nodes with {n_devices_per_process} devices per process.\n"
            "JAXMg only supports one process for all devices per node or one process per device across all nodes"
        )

else:
    warnings.warn(
        f"No GPUs found, only use this mode for testing.",
        JaxMgWarning,
        stacklevel=2,
    )
from .cyclic_1d import (
    cyclic_1d_no_shardmap,
    cyclic_1d_layout,
    undo_cyclic_1d_layout,
    manual_cyclic_1d_layout,
    calculate_padding,
    calculate_valid_T_A,
    calculate_all_valid_T_A,
)

__all__ = [
    "potrs",
    "potrs_no_shardmap" "potri",
    "syevd",
    "cyclic_1d_layout",
    "cyclic_1d_no_shardmap",
    "undo_cyclic_1d_layout",
    "manual_cyclic_1d_layout",
    "calculate_padding",
    "calculate_valid_T_A",
    "calculate_all_valid_T_A",
    "determine_distributed_setup",
]
