JAXMg provides a C++ interface between [JAX](https://github.com/google/jax) and [cuSolverMg](https://docs.nvidia.com/cuda/cusolver/index.html#using-the-cuSolverMg-api), NVIDIA’s multi-GPU linear solver.  We provide a jittable API for the following routines.

- `cusolverMgPotrs`: Solves the system of linear equations: $Ax=b$ where $A$ is an $N\times N$ symmetric (Hermitian) positive-definite matrix via a Cholesky decomposition.
- `cusolverMgPotri`: Computes the inverse of an $N\times N$ symmetric (Hermitian) positive-definite matrix via a Cholesky decomposition.
- `cusolverMgSyevd`: Computes eigenvalues and eigenvectors of an $N\times N$ symmetric (Hermitian) matrix.

The provided binary is compiled with:
- **GCC**: 11.5.0  
- **CUDA**: 12.8.0  
- **cuDNN**: 9.2.0.82-12  

> **Note:** JAX>=0.6.0 ships with CUDA 12.x binaries, which this package relies on. No local version of CUDA is required.
