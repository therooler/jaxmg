import importlib
import pathlib
import ctypes
import warnings

from functools import partial
from .utils import JaxMgWarning

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
                raise OSError(f"Unable to load CUDA library {lib}") from e


_load("cusolver", ["libcusolver.so.11"])
_load("cusolver", ["libcusolverMg.so.11"])

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

if any("gpu" == d.platform for d in jax.devices()):
    @partial(jax.pmap, axis_name="d")
    def warmup(x):
        return jax.lax.psum(x, "d")
    if len(jax.devices("gpu")) > 1:
      warnings.warn(
          f"Multiple GPUs detected, initializing communication primitives...",          JaxMgWarning,
          stacklevel=2,
      )
      jax.block_until_ready(
          warmup(jnp.ones((jax.local_device_count(),)))
      )  # triggers NCCL setup
