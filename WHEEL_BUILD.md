# Building Wheels for jaxmg

This document explains how to build distributable wheels for jaxmg with CUDA support.

## Overview

The project uses `cibuildwheel` to build manylinux wheels with pre-compiled CUDA libraries. This allows users to install jaxmg without needing to compile the CUDA code themselves.

## CI Build Process

Wheels are automatically built on GitHub Actions when:
- A tag matching `v*` is pushed (e.g., `v0.1.0`)
- The workflow is manually triggered via workflow_dispatch
- Changes are made to the build configuration files

### What the CI does:

1. **Sets up build environment**: Uses manylinux_2_28 Docker image with GCC 11
2. **Installs CUDA Toolkit 12.8** and cuDNN 9.2
3. **Compiles CUDA libraries**: Runs CMake to build the 6 shared libraries (.so files)
4. **Creates wheels**: Packages the compiled libraries with the Python code
5. **Repairs wheels**: Uses auditwheel to create manylinux-compliant wheels, excluding CUDA runtime libraries
6. **Uploads artifacts**: Wheels are uploaded as GitHub Actions artifacts (30-day retention)

### Supported configurations:

- **Python versions**: 3.9, 3.10, 3.11, 3.12
- **Platform**: Linux x86_64 only
- **CUDA version**: 12.8 (compatible with JAX CUDA 12.x)

## Local Testing

To test the wheel build process locally:

```bash
# Install cibuildwheel
pip install cibuildwheel==2.16.2

# Build wheels (requires Docker)
cibuildwheel --platform linux --output-dir wheelhouse

# This will take significant time on first run as it downloads CUDA
```

## Using Pre-built Wheels

Once wheels are built in CI:

1. Go to the GitHub Actions run
2. Download the `jaxmg-wheels-*` artifact
3. Extract and install:

```bash
pip install jaxmg-0.1.0-cp311-cp311-manylinux_2_28_x86_64.whl
```

Or install with CUDA dependencies:

```bash
pip install jaxmg-0.1.0-cp311-cp311-manylinux_2_28_x86_64.whl
pip install "jax[cuda12]>=0.6.2"
```

## Important Notes

### CUDA Runtime Dependencies

The wheels **do not include** CUDA runtime libraries. Users must have CUDA libraries available through one of:

1. **JAX CUDA installation** (recommended):
   ```bash
   pip install "jax[cuda12]>=0.6.2"
   ```
   JAX ships with CUDA 12.x runtime libraries.

2. **System CUDA installation**: CUDA 12.1+ installed on the system

3. **NVIDIA Container Runtime**: When running in Docker with NVIDIA runtime

### Excluded Libraries

The following CUDA libraries are excluded from the wheel (users get them via JAX or system CUDA):

- `libcusolver.so.12`
- `libcusolverMg.so.12`
- `libcudart.so.12`
- `libcublas.so.12`
- `libcublasLt.so.12`
- `libcupti.so.12`
- `libcuda.so.1`
- `libcudnn.so.9`

### Wheel Size

Without bundling CUDA libraries, wheels are approximately 12-15 MB each (for the 6 compiled .so files). With CUDA libraries bundled, they would be 400+ MB.

## Future Improvements

- [ ] Add wheel building for CUDA 13 when JAX migrates
- [ ] Consider building wheels for multiple CUDA versions (12.x, 13.x)
- [ ] Add automated PyPI publishing (currently only GitHub artifacts)
- [ ] Add wheel verification tests

## Troubleshooting

### cuDNN Download Issues

The current workflow downloads cuDNN 9.2 from NVIDIA. If this URL becomes unavailable, you may need to:

1. Host cuDNN in a more permanent location
2. Use a different cuDNN version (ensure compatibility)
3. Modify the `CIBW_BEFORE_ALL_LINUX` step in the workflow

### Build Failures

Common issues:

- **CUDA not found**: Check that CUDA_ROOT and CUDNN_ROOT are set correctly
- **GCC version**: Must use GCC 11 for C++20 support
- **CMake version**: Use cmake3 on manylinux images
- **Missing symlinks**: CMake build creates symlinks to CUDA/cuDNN, ensure they're created successfully

### Testing Wheels

To test a wheel without GPU access:

```bash
# Install the wheel
pip install wheelhouse/jaxmg-*.whl

# Import should work (though running code requires GPU)
python -c "import jaxmg; print(jaxmg.__version__)"
```

For full testing, you need a system with NVIDIA GPU and CUDA runtime.
