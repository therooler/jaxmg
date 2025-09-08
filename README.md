# Distributed Linear Solver with cuSolverMg and JAX

This repository provides a C++ interface between [JAX](https://github.com/google/jax) and **cuSolverMg**, NVIDIA’s distributed linear solver.  

To use cuSolverMg, matrices must be stored in **1D block-cyclic, column-major form**. This package handles that transformation on the JAX side with a single **all-to-all** call in combination with `jax.shard_map`.

<img src="mat.png" alt="Matrix layout illustration" width="800">

The provided binary is compiled with:
- **GCC**: 11.5.0  
- **CUDA**: 12.8.0  
- **cuDNN**: 9.2.0.82-12  

> **Note:** JAX ships with CUDA 12.x binaries, which this package relies on.

---

## Installation

Clone the repository and install with:

```bash
pip install .
```

## Simple example

Consider the case where we have 2 GPUs available and we are trying to solve the linear 
system $A\cdot x =b$, where $A$ is an $12\times12$, positive-definite matrix and $b$ corresponds to a vector of ones. Every shard on each GPU will be of size $12\times 6$.
We require a cyclic 1D tiling with tile size `T_A=2` for `cusolverMg` to work. This 
results in the following layout:

<img src="mat_example.png" alt="Matrix layout illustration" width="800">

In order to interweave the blocks, we need to ensure that each shard is a multiple of
`ndev * T_A = 4`, so that we can reshape to `(ndev, T_A, ...)` and exchange the blocks via `jax.lax.all-to-all`. As a result, we add zero padding of 2 columns to each shard.


```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding
from jaxmg import potrf

# Assumes we have at least one GPU available
devices = jax.devices("gpu")
N = 12
T_A = 2
dtype = jnp.float64
# Create diagonal matrix and `b` all equal to one
A = jnp.diag(jnp.arange(N, dtype=dtype) + 1)
b = jnp.ones((N, 1), dtype=dtype)
ndev = len(devices)
# Make mesh and place data (columns sharded)
mesh = jax.make_mesh((ndev,), ("x",))
A = jax.device_put(A, NamedSharding(mesh, P(None, "x")))
b = jax.device_put(b, NamedSharding(mesh, P(None, None)))
# Call potrf
out = potrf(A, b, T_A=T_A)
expected_out = 1.0 / (jnp.arange(N, dtype=dtype) + 1)
print(jnp.allclose(out.flatten(), expected_out))
```
which should print
```bash
True
```

## Testing

To verify the installation (requires at least one GPU):

```bash
pytest 
```

CPU-only tests: The block-cyclic remapping is checked by simulating multiple CPU devices.
Multi-GPU tests: Requires multiple available GPUs.

## Development

To build from source:

```bash
mkdir build
cd build
cmake ..
cmake --build . --target install
```

This installs the CUDA binaries into src/jaxmg/bin.

Dependencies are managed with [CPM-CMAKE](https://github.com/cpm-cmake/CPM.cmake),
including **abseil-cpp**, **jaxlib**, **XLA** for compilation. Compilation requires C++17 or later.
To build specific targets only, for example potrf:
```bash
cmake ..
cmake --build . --target potrf
cmake --install .
```


## Citation
(Citation details will be available soon.)

## Acknowledgements
I acknowledge support from the Flatiron Institute. The Flatiron Institute is a
division of the Simons Foundation.