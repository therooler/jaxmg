<p align="center">
    <img src="docs/_static/logo.png" alt="Jaxmg" width="300">
</p>

# JAXMg: A distributed linear solver in JAX with cuSolverMg

[![Docs](https://img.shields.io/badge/docs-site-blue?style=flat-square)](https://therooler.github.io/jaxmg/)
[![Releases](https://img.shields.io/github/v/release/therooler/jaxmg?style=flat-square)](https://github.com/therooler/jaxmg/releases)

[![Continuous integration](https://github.com/therooler/jaxmg/actions/workflows/ci-build.yaml/badge.svg)](https://github.com/therooler/jaxmg/actions/workflows/ci-build.yaml)

This repository provides a C++ interface between [JAX](https://github.com/google/jax) and [cuSolverMg](https://docs.nvidia.com/cuda/cusolver/index.html#using-the-cuSolverMg-api), NVIDIA’s multi-GPU linear solver.  

The python package `jaxmg` provides a jittable API for the following routines.

- `cusolverMgPotrs`: Solves the system of linear equations: $Ax=b$ where $A$ is an $N\times N$ symmetric (Hermitian) positive-definite matrix via a Cholesky decomposition.
- `cusolverMgPotri`: Computes the inverse of an $N\times N$ symmetric (Hermitian) positive-definite matrix via a Cholesky decomposition.
- `cusolverMgSyevd`: Computes eigenvalues and eigenvectors of an $N\times N$ symmetric (Hermitian) matrix.

This README.md will focus on the `jaxmg.potrs` API, but we provide examples for calling `jaxmg.potri` and `jaxmg.syevd` in the both the `/examples` and `/tests` folders.

The provided binary is compiled with:
- **GCC**: 11.5.0  
- **CUDA**: 12.8.0  
- **cuDNN**: 9.2.0.82-12  

> **Note:** JAX ships with CUDA 12.x binaries, which this package relies on. No local version of CUDA is required.

<!-- ### Contents
* [Installation](#installation) -->


## Installation

Clone the repository and install with:

```bash
pip install .
```

To verify the installation (requires at least one GPU):

```bash
pytest 
```
There are three types of tests:

1. CPU-only tests: The block-cyclic remapping is checked by simulating multiple CPU devices.
2. Single-GPU tests: A single GPU. 
3. Multi-GPU tests: Requires multiple available GPUs.

If there are not multiple GPUs availble we skip the tests that require multiple GPUs.

<!-- ## Examples

### Block-cyclic data layout

To use cuSolverMg, matrices must be stored in **1D block-cyclic, column-major form**. The reason for this is to ensure that all devices participating in a specific routine can perform computations without being blocked by other parts of the computation (see Dongarra 1996). In `jaxmg`, we handles this transformation on the JAX side with a single **all-to-all** within a `jax.shard_map` context.

Consider the case where we have 2 GPUs available and we are trying to solve the linear 
system $A\cdot x =b$, where $A$ is an $12\times12$, positive-definite matrix and $b$ corresponds to a vector of ones. Every shard on each GPU will be of size $12\times 6$.
We require a cyclic 1D tiling with tile size `T_A=2` for `cuSolverMg` to work:

<img src="docs/_static/mat_example.png" alt="Matrix layout illustration" width="800">


In order to interweave the blocks, we need to ensure that each shard is a multiple of
`ndev * T_A = 2`, so that we can reshape to `(ndev, T_A, ...)` and exchange the blocks via `jax.lax.all_to_all`. We therefore add zero padding of 2 columns to each shard (see top figure). After interweaving the blocks, we are left with extra padding on the right, which we ignore in the solver itself. After the solver is called, we again use a
single `jax.lax.all_to_all` call to remap the data back to block-sharded form. 

We can achieve this layout in `jaxmg` with the following code:

```python
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding
from jaxmg import cyclic_1d_layout

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
A_bc = cyclic_1d_layout(A, T_A)

for shard in A_bc.addressable_shards:
    print(f"dev {dev}: shard\n {shard.data}")
```
which prints
```bash
dev 0: shard
 [ 0.  2.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.]
 [ 0.  0.  5.  0.  0.  0.]
 [ 0.  0.  0.  6.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  9.  0.]
 [ 0.  0.  0.  0.  0. 10.]
 [ 0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.]]
dev 1: shard
 [[ 0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.]
 [ 3.  0.  0.  0.  0.  0.]
 [ 0.  4.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.]
 [ 0.  0.  7.  0.  0.  0.]
 [ 0.  0.  0.  8.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0. 11.  0.]
 [ 0.  0.  0.  0.  0. 12.]]
```

A more involved example is the case where we have 4 GPUS, `N=100` and we want a tiling of `T_A=4`. Now we need a padding of 7 on each GPU in order to perform data remappping (produced with above code):

<img src="docs/_static/mat.png" alt="Matrix layout illustration" width="800">

- Dongarra, J.J., and D.W. Walker. *The Design of Linear Algebra Libraries for High Performance Computers.* Office of Scientific and Technical Information (OSTI), August 1, 1993. https://doi.org/10.2172/10184308.

### A simple example: `jaxmg.potrs`

In practice, the cyclic relayout is taken care of when you call any of the solvers in `jaxmg`. Here, we give an example of calling `jax.potrs`, which solves the linear system of equations $Ax=b$ for symmetric, positive-definite $A$ via a Cholesky decomposition.

The interface of `jaxmg.potrs` is simple to use; one needs to supply to underlying mesh of the sharded data and specify the input shardings:

```python
# examples/readme.py
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding
from jaxmg import potrs

# Assumes we have at least one GPU available
devices = jax.devices("gpu")
assert len(devices) in [1, 2], "Example only works for 1 or 2 devices"
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
out = potrs(A, b, T_A=T_A, mesh=mesh, in_specs=(P(None, "x"), P(None, None)))
expected_out = 1.0 / (jnp.arange(N, dtype=dtype) + 1)
print(jnp.allclose(out.flatten(), expected_out))
```
which should print
```bash
True
```
Note that we did not have to perform the cyclic relayout here since `jaxmg.potrs` calls `cyclic_1d_layout` before calling the solver.

> **Note:** If the user can ensure that the matrix `A` is already in block cyclic form, then `jaxmg.potrs` can be called with the argument `cyclic_1d=True` (`False` by default). If the data is not laid out correctly, then calling `jaxmg.potrs` will result in an array of `NaN`s and a nonzero status, indicating the failure of solver.



### Calling the solver in a `jax.shard_map` context

The `potrs` interface uses a call to `jax.shard_map` to relayout the data in 1D cyclic form and call the underlying cuSolverMg API. In practice, one may have a more complicated jitted function that manipulates shards in a shard_map context already, which 
requires calling the solver within this function on individual shards.

To allow for this use case, we also provide an API that has to be called in a shard_map context. 
Here, we rely on the user to correctly call `jax.shard_map`, passing the correct in and out shardings to their own function.

In the example below, we use this API for a trivial matrix, now with `complex64` data type, where we apply a diagonal shift to to the 
matrix `A` before handing it to the solver:

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding
from jaxmg import potrs_no_shardmap
from functools import partial

# Assumes we have at least one GPU available
devices = jax.devices("gpu")
assert len(devices) in [1, 2], "Example only works for 1 or 2 devices"
N = 8
T_A = 2
dtype = jnp.complex64
# Create diagonal matrix and `b` all equal to one
A = jnp.diag(jnp.arange(N, dtype=dtype) + 1)
b = jnp.ones((N, 1), dtype=dtype)
ndev = len(devices)
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
    # Call solver in shard_map context
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
expected_out = 1.0 / (jnp.arange(N, dtype=dtype) + 1 + diag_shift)
print(jnp.allclose(out.flatten(), expected_out))
```
for two devices, this will print
```bash
dev0:_a=
[[1.1+0.j 0. +0.j 0. +0.j 0. +0.j]
 [0. +0.j 2.1+0.j 0. +0.j 0. +0.j]
 [0. +0.j 0. +0.j 3.1+0.j 0. +0.j]
 [0. +0.j 0. +0.j 0. +0.j 4.1+0.j]
 [0. +0.j 0. +0.j 0. +0.j 0. +0.j]
 [0. +0.j 0. +0.j 0. +0.j 0. +0.j]
 [0. +0.j 0. +0.j 0. +0.j 0. +0.j]
 [0. +0.j 0. +0.j 0. +0.j 0. +0.j]]

dev1:_a=
[[0. +0.j 0. +0.j 0. +0.j 0. +0.j]
 [0. +0.j 0. +0.j 0. +0.j 0. +0.j]
 [0. +0.j 0. +0.j 0. +0.j 0. +0.j]
 [0. +0.j 0. +0.j 0. +0.j 0. +0.j]
 [5.1+0.j 0. +0.j 0. +0.j 0. +0.j]
 [0. +0.j 6.1+0.j 0. +0.j 0. +0.j]
 [0. +0.j 0. +0.j 7.1+0.j 0. +0.j]
 [0. +0.j 0. +0.j 0. +0.j 8.1+0.j]]

Status: [0]
True
```

> **Note:** `potrs_no_shardmap` always returns a status.

> **Note:** Jax will complain about replication errors if you do not pass `check_vma=True`. This is likely because it cannot infer the output sharding from the ffi call.

### SPMD and MPMD support

We support two modes of distributed computing. Single Process Multiple Devices mode (SPMD), where we have
a single process per node that potentially manages multiple devices. We also support MPMD mode (MPMD). Here the user needs to
use one process for each GPU. When `jaxmg` is imported, we attempt to verify the user's distributed setup to not go out beyond these two modes of computation.

`jaxmg` supports multi-process `jax.distributed` environments but cuSolverMg **can only run on a single node**. There are some technical reasons for this (see below) that will hopefully be resolved in a future release.

To circumvent this limitation, one can perform a computation over all global devices, replicate the results over all host by gathering the data and calling the solver only on the process-local devices. 

Here we provide an example of using `jaxmg` in a context where we have 2 nodes, each with 4 GPUs.
In order to use the solver, we will have to gather the results onto each node by making use of a 2D `Mesh`. To illustrate the data layout, we will simply work with CPUs which allows us to run this code on a local machine.


```python
# Call ./examples/multi_process.sh to launch this code!
import os
import sys

proc_id = int(sys.argv[1])
num_procs = int(sys.argv[2])

# initialize the distributed system
import jax

jax.config.update("jax_platform_name", "cpu")
jax.distributed.initialize("localhost:6000", num_procs, proc_id)

from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import jax.numpy as jnp
import numpy as np


def get_device_grid():
    by_proc = {}
    for d in jax.devices():
        by_proc.setdefault(d.process_index, []).append(d)
    hosts = sorted(by_proc)
    return np.array(
        [[by_proc[h][x] for x in range(jax.local_device_count())] for h in hosts]
    )


def create_2d_mesh():
    dev_grid = get_device_grid()
    return Mesh(dev_grid, ("x", "y"))


def create_1d_mesh():
    dev_grid = get_device_grid()
    return Mesh(dev_grid.flatten(), ("y",))

print(f"Rank {proc_id}")
print(f"Local devices {jax.local_device_count()}")
print(f"Global devices {jax.device_count()}")
print(f"World size {num_procs}")
print(f"Device grid\n {get_device_grid()}")
```
When we launch this code like this:
```bash
#!/bin/bash

export JAX_NUM_CPU_DEVICES=4
num_processes=2

range=$(seq 0 $(($num_processes - 1)))

for i in $range; do
  python multi_process.py $i $num_processes > /tmp/multi_process_$i.out &
done

wait

for i in $range; do
  echo "=================== process $i output ==================="
  cat /tmp/multi_process_$i.out
  echo
done
```
we see
```
=================== process 0 output ===================
Rank 0
Local devices 4
Global devices 8
World size 2
Device grid
 [[CpuDevice(id=0) CpuDevice(id=1) CpuDevice(id=2) CpuDevice(id=3)]
 [CpuDevice(id=131072) CpuDevice(id=131073) CpuDevice(id=131074) CpuDevice(id=131075)]]
 =================== process 1 output ===================
Rank 1
Local devices 4
Global devices 8
World size 2
Device grid
 [[CpuDevice(id=0) CpuDevice(id=1) CpuDevice(id=2) CpuDevice(id=3)]
 [CpuDevice(id=131072) CpuDevice(id=131073) CpuDevice(id=131074) CpuDevice(id=131075)]]
```
We can then construct a matrix that has its columns sharded over all global devices, and gather the columns onto each host:

```python
mesh2d = create_2d_mesh()

A = jax.device_put(
    jnp.diag(jnp.arange(1, jax.device_count() + 1, dtype=jnp.float32)),
    NamedSharding(mesh2d, P(None, ("x", "y"))),
)

for shard in A.addressable_shards:
    print(f"shard\n {shard.data}")

# Gather over the number of hosts
A = jax.lax.with_sharding_constraint(A, NamedSharding(mesh2d, P(None, "y")))

for shard in A.addressable_shards:
    print(f"shard\n {shard.data}")
```
which prints
```
=================== process 0 output ===================
Rank 0
Local devices 4
Global devices 8
World size 2
Device grid
 [[CpuDevice(id=0) CpuDevice(id=1) CpuDevice(id=2) CpuDevice(id=3)]
 [CpuDevice(id=131072) CpuDevice(id=131073) CpuDevice(id=131074)
  CpuDevice(id=131075)]]
shard
 [[1.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]]
...
shard
 [[0.]
 [0.]
 [0.]
 [4.]
 [0.]
 [0.]
 [0.]
 [0.]]
shard
 [[1. 0.]
 [0. 2.]
 [0. 0.]
 [0. 0.]
 [0. 0.]
 [0. 0.]
 [0. 0.]
 [0. 0.]]
...
shard
 [[0. 0.]
 [0. 0.]
 [0. 0.]
 [0. 0.]
 [0. 0.]
 [0. 0.]
 [7. 0.]
 [0. 8.]]

=================== process 1 output ===================
...
shard
 [[0.]
 [0.]
 [0.]
 [0.]
 [5.]
 [0.]
 [0.]
 [0.]]
shard
 ...
shard
 [[0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [0.]
 [8.]]
shard
 [[1. 0.]
 [0. 2.]
 [0. 0.]
 [0. 0.]
 [0. 0.]
 [0. 0.]
 [0. 0.]
 [0. 0.]]
...
shard
 [[0. 0.]
 [0. 0.]
 [0. 0.]
 [0. 0.]
 [0. 0.]
 [0. 0.]
 [7. 0.]
 [0. 8.]]
```

We went from a matrix that was column sharded over all 8 global devices, to a matrix that was column sharded over the 4 gpus in each process.

In this host-replicated layout we can safely call `jaxmg.potrs` on the array with a 2D mesh (the code below only works if we are actually performing this computation with access to GPUs):

```python
from jaxmg import potrs
out= potrs(
    A,
    jnp.ones((jax.device_count(), 1), dtype=jnp.float32),
    T_A=256,
    mesh=mesh2d,
    in_specs=(P(None, "T"), P(None, None)),
)
```

## Sharp bits


- **Potential invalid tilings:** It is possible that for a given $N\times N$ matrix the provided `T_A` does not allow one to use a single `jax.lax.all_to_all` call to bring the matrix to cyclic 1D form. In this case we raise an error, and suggest both a smaller and larger `T_A` that would enable the data remapping. This problem mostly occurs for small matrices, where the number of tiles is small and `T_A` is close to the shard size. To calculate all available tilings, we provide a function `jaxmg.calculate_all_valid_T_A` that will return all possible valid tilings.

- **Maximum tilings:** If the tiling `T_A` is too small, the solver can slow down significantly. In the cuSolverMg documentation, the recommended value for `T_A` is "256 or above". There is no maximum value of `T_A` for `jaxmg.potrs` and `jaxmg.potri`. However, for the symmetric eigensolver `jaxmg.syevd`, the maximum value of `T_A` is 1024.

- **Maximum number of GPUs:** According to the cuSolverMg documentation, the current maximum number of GPUs is hardcoded to be 16. Going beyond this value will raise a an error from within CUDA code.

## Technical details of implementation

### SPMD mode

When `potrs.cu` is called in a `jax.shard_map` context through the `jax.ffi` API with a single process for multiple devices,

```python
_out, status = jax.ffi.ffi_call(
            "potrs_mg",
            out_type,
            input_layouts=input_layouts,
            output_layouts=output_layouts,
        )(_a, _b, T_A=T_A)
```

a thread will spawn for each available GPU that executes the code in `potrs.cu`. Each thread will only have access to its local shard in GPU memory through a device pointer. The `cuSolverMgPotrf` API must be called in a single thread and requires an array of **all** device pointers containing the shards on each GPU. 

This raises the following two issues.

1. We need to synchronise the threads to set up `cuSolverMgPotrf` and the data. We then need to execute the solver in thread 0 and have the other threads wait for it to finish. However, JAX has spawned the threads and we do not have any explict control over the thread  syncronization.
2. Since each thread only has access to its local shard, we need to somehow make thread 0 aware of the device pointers across all other threads.

We solver the first issue by initializing a global barrier via `std::unique_ptr<std::barrier<>> barrier_ptr`. Here `std::unique_ptr` takes care of deleting the barrier when it goes out of scope (when the FFI call finishes). Then, in `potrs.cu` we use 
```C++
static std::once_flag barrier_initialized;
std::call_once(barrier_initialized, [&](){ sync_point.initialize(nbGpus); });
```
to initialize the barrier across all threads. The `std::once_flag` ensures that the barrier is initialized exactly once so that all threads see the same barrier.

We share device pointers between threads through the creation of shared memory:

```C++
data_type **shmA = get_shm_device_ptrs<data_type>(currentDevice,sync_point, shminfoA, "shmA"); 
```

In each thread, we then assign the device pointer of the local shard to this shared memory:

```C++
shmA[currentDevice] = array_data_A;
```

which we can safely pass to `cuSolverMgPotrf`:
```C++
cusolver_status = cusolverMgPotrs(cusolverH, CUBLAS_FILL_MODE_LOWER, N,NRHS, 
                                  reinterpret_cast<void **>(shmA), IA, JA, descrA,
                                  reinterpret_cast<void **>(shmB), IB, JB, descrB,
                                  compute_type,
                                  reinterpret_cast<void **>(shmwork), *shmlwork,
                                  &info);
```

### MPMD mode

In a multi-process context it is not as straightforward to setup memory sharing between processes, especially when it comes to passing around device pointers which are bound to a specific CUDA context. 

The solution used here is to make use of the cudaIPC documentation, which allows one to export handles to device memory to
different processes. In `potrs_mp.cu`, we achieve this again through shared memory, although now we share the cudaIPC memory
handles:

```cpp
ipcGetHandleAndOffset(array_data_A, shmAipc[currentDevice], shmoffsetA[currentDevice]);
```

A significant complication is that JAX' memory allocation is managed by XLA, which means that device pointers are actually
base pointers together with some offset. cudaIPC only exports the base-pointer, so we have to manually pass around the 
offset and extract the true pointer:

```cpp
opened_ptrs_A = ipcGetDevicePointers<data_type>(currentDevice, nbGpus, shmAipc, shmoffsetA);
```

We gather all the pointers in process 0 and set up the solver in the same way as before. After completion, it is essential
to close the memory handles

```cpp
ipcCloseDevicePointers(currentDevice, opened_ptrs_A.bases, nbGpus);
```

to avoid memory leaks.

> **Note:** If you've made it this far into the README.md and have experience or thoughts on this, please reach out!
 -->

### cuSolverMp
As of CUDA 13, there is a new distributed linear algebra library called [cuSolverMp](https://docs.nvidia.com/cuda/cusolvermp/) with similar capabilities as cuSolverMg, that does support multi-node computations as well as >16 devices. Given the similarities in syntax, it should be straightforward to eventually switch to this API. This will require sharding data into a cyclic 2D form and handling the solver orchestration with MPI.

## Citations
(Citation details will be available soon.)

## Acknowledgements
I acknowledge support from the Flatiron Institute. The Flatiron Institute is a
division of the Simons Foundation.