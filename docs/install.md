# Installation
Clone the repository and install with:

```bash
pip install jaxmg
```

This will install a GPU compatible version of JAX. 

To verify the installation (requires at least one GPU) run

```bash
pytest 
```
There are two types of tests:

1. SPMD tests: Single Process Multiple GPU tests.
3. MPMD: Multiple Processes Multiple GPU tests.