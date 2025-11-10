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