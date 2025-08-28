# Distributed Linear Solver with CusolveMg and Jax

This code provides a C++ interface between CusolveMg, a distributed linear solver
provided by NVIDIA. Calling the distributed solver requires laying out matrices in
1D block cyclic form, which we handle on the Jax side with a single all-to-all call in
combination with `jax.shard_map`. 

We require `jax>=0.6.1` and building from source requires a C++17 compiler, CUDA Toolkit >=12.3