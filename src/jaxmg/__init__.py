import importlib
import pathlib
import ctypes
import warnings
import sys

from functools import partial
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


_load("cusolver", ["libcusolver.so.11"])
_load("cusolver", ["libcusolverMg.so.11"])

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

if any("gpu" == d.platform for d in jax.devices()):

    @partial(jax.pmap, axis_name="d")
    def warmup(x):
        return jax.lax.psum(x, "d")

    if jax.local_device_count() > 1:
        warnings.warn(
            f"Multiple GPUs detected, initializing communication primitives...",
            JaxMgWarning,
            stacklevel=2,
        )
        jax.block_until_ready(
            warmup(jnp.ones((jax.local_device_count(),)))
        )  # triggers NCCL setup

from .potrs import potrs, potrs_no_shardmap
from .potri import potri
from .syevd import syevd
from .cyclic_1d import (
    cyclic_1d_layout,
    undo_cyclic_1d_layout,
    manual_cyclic_1d_layout,
    calculate_padding,
    calculate_valid_T_A,
)

__all__ = [
    "potrs",
    "potrs_no_shardmap"
    "potri",
    "syevd",
    "cyclic_1d_layout",
    "undo_cyclic_1d_layout",
    "manual_cyclic_1d_layout",
    "calculate_padding",
    "calculate_valid_T_A",
]
