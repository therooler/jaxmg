import importlib
import pathlib
import ctypes

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
