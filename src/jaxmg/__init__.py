import importlib
import pathlib
import ctypes
import warnings
import sys
import os

from .utils import JaxMgWarning

if not sys.platform.startswith("linux"):
    warnings.warn(
        f"Unsupported platform {sys.platform}, only Linux is supported. Non-Linux only works for docs.",
        JaxMgWarning,
        stacklevel=2,
    )


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

    n_machines, n_devices_per_node, n_devices_per_process, mode = (
        determine_distributed_setup()
    )
    os.environ["JAXMG_NUMBER_OF_DEVICES"] = str(n_devices_per_node)
    if n_machines > 1:
        warnings.warn(
            f"Computation seems to be running on multiple machines.\n"
            "Ensure that jaxmg is only called over a local device mesh, otherwise process might hang.\n"
            "See examples for how this can be safely achieved.",
            JaxMgWarning,
            stacklevel=2,
        )
    if mode == "SPMD":
        # Load the shared libraries
        SHARED_LIBRARY_POTRS = os.path.join(os.path.dirname(__file__), "bin/libpotrs.so")
        library_potrs = ctypes.cdll.LoadLibrary(SHARED_LIBRARY_POTRS)
        SHARED_LIBRARY_POTRI = os.path.join(os.path.dirname(__file__), "bin/libpotri.so")
        library_potri = ctypes.cdll.LoadLibrary(SHARED_LIBRARY_POTRI)
        SHARED_LIBRARY_SYEVD = os.path.join(os.path.dirname(__file__), "bin/libsyevd.so")
        library_syevd = ctypes.cdll.LoadLibrary(SHARED_LIBRARY_SYEVD)
        SHARED_LIBRARY_SYEVD_NO_V = os.path.join(os.path.dirname(__file__), "bin/libsyevd_no_V.so")
        library_syevd_no_V = ctypes.cdll.LoadLibrary(SHARED_LIBRARY_SYEVD_NO_V)
        # Register FFI targets
        jax.ffi.register_ffi_target(
            "potrs_mg", jax.ffi.pycapsule(library_potrs.PotrsMgFFI), platform="CUDA"
        )
        jax.ffi.register_ffi_target(
            "potri_mg", jax.ffi.pycapsule(library_potri.PotriMgFFI), platform="CUDA"
        )
        jax.ffi.register_ffi_target(
            "syevd_mg", jax.ffi.pycapsule(library_syevd.SyevdMgFFI), platform="CUDA"
        )
        jax.ffi.register_ffi_target(
            "syevd_no_V_mg", jax.ffi.pycapsule(library_syevd_no_V.SyevdMgFFI), platform="CUDA"
        )
        from ._potrs import potrs, potrs_no_shardmap
        from ._potri import potri
        from ._syevd import syevd
    else:
        # Load the shared library
        SHARED_LIBRARY_POTRS_MP = os.path.join(os.path.dirname(__file__), "bin/libpotrs_mp.so")
        library_potrs_mp = ctypes.cdll.LoadLibrary(SHARED_LIBRARY_POTRS_MP)
        # Register FFI targets
        jax.ffi.register_ffi_target(
            "potrs_mp_mg", jax.ffi.pycapsule(library_potrs_mp.PotrsMgMpFFI), platform="CUDA"
        )
        from ._potrs_mp import potrs, potrs_no_shardmap
        
else:
    warnings.warn(
        f"No GPUs found, only use this mode for testing or docs.",
        JaxMgWarning,
        stacklevel=2,
    )
    from ._potrs import potrs, potrs_no_shardmap
    from ._potri import potri
    from ._syevd import syevd
    from ._potrs_mp import potrs, potrs_no_shardmap

    os.environ["JAXMG_NUMBER_OF_DEVICES"] = str(jax.device_count())

from ._cyclic_1d import (
    cyclic_1d_no_shardmap,
    cyclic_1d_layout,
    undo_cyclic_1d_layout,
    manual_cyclic_1d_layout,
    calculate_padding,
    calculate_valid_T_A,
    calculate_all_valid_T_A,
)

__all__ = [
    "_potrs",
    "potrs_no_shardmap",
    "_potri",
    "_syevd",
    "cyclic_1d_layout",
    "cyclic_1d_no_shardmap",
    "undo_cyclic_1d_layout",
    "manual_cyclic_1d_layout",
    "calculate_padding",
    "calculate_valid_T_A",
    "calculate_all_valid_T_A",
    "determine_distributed_setup",
]
