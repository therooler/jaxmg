# JAXMg
JAXMg provides a C++ interface between [JAX](https://github.com/google/jax) and [cuSolverMg](https://docs.nvidia.com/cuda/cusolver/index.html#using-the-cuSolverMg-api), NVIDIA’s multi-GPU linear solver.  We provide a jittable API for the following routines.

- [cusolverMgPotrs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolvermgpotrs-deprecated): Solves the system of linear equations: $Ax=b$ where $A$ is an $N\times N$ symmetric (Hermitian) positive-definite matrix via a Cholesky decomposition 
- [cusolverMgPotrs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolvermgpotri-deprecated): Computes the inverse of an $N\times N$ symmetric (Hermitian) positive-definite matrix via a Cholesky decomposition.
- [cusolverMgPotrs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolvermgsyevd-deprecated): Computes eigenvalues and eigenvectors of an $N\times N$ symmetric (Hermitian) matrix.

For more details, see the [API](api/potrs.md).

The provided binary is compiled with:

| Component | Version |
|---|---:|
| **GCC** | 11.5.0 |
| **CUDA** | 12.8.0 |
| **cuDNN** | 9.2.0.82-12 |

!!! Compatibility
    We require JAX>=0.6.0, since it ships with CUDA 12.x binaries, which this package relies on. No local version of CUDA is required.
