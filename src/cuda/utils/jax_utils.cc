/* Copyright 2024 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Own code
#include "utils/jax_utils.h"
// Abseil
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
// JAXlib
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/gpu/handle_pool.h"
// CUDA
#include "third_party/gpus/cuda/include/cusolverMg.h"

namespace jax
{

  template <>
  /*static*/ absl::StatusOr<SolverHandlePool::Handle> SolverHandlePool::Borrow(
      gpuStream_t stream)
  {
    SolverHandlePool *pool = Instance();
    absl::MutexLock lock(&pool->mu_);
    cusolverMgHandle_t handle;
    if (pool->handles_[stream].empty())
    {
      // std::printf("Hitting cusolverMgCreate");
      JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cusolverMgCreate(&handle)));
    }
    else
    {
      handle = pool->handles_[stream].back();
      pool->handles_[stream].pop_back();
    }
    // #TODO Need to handle the case where the stream is nonempty, which should never happen.
    //   if (stream) {
    // std::printf("Hitting this if statement");
    // JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpusolverDnSetStream(handle, stream)));
    //   }
    return Handle(pool, handle, stream);
  }

} // namespace jax

void workspaceAlloc(int num_devices, const int *deviceIdA, /* <int> dimension num_devices */
                    size_t sizeInBytes,                    /* number of bytes per device */
                    void **array_d_work                    /* <t> num_devices, host array */
                                                           /* array_d_work[j] points to device workspace of device j */
)
{
  int currentDev = 0; /* record current device ID */
  CUDA_CHECK(cudaGetDevice(&currentDev));

  for (int idx = 0; idx < num_devices; idx++)
  {
    int deviceId = deviceIdA[idx];
    /* WARNING: we need to set device before any runtime API */
    CUDA_CHECK(cudaSetDevice(deviceId));

    void *d_workspace = NULL;

    CUDA_CHECK(cudaMalloc(&d_workspace, sizeInBytes));
    array_d_work[idx] = d_workspace;
  }
  CUDA_CHECK(cudaSetDevice(currentDev));
}

void workspaceFree(int num_devices, const int *deviceIdA, /* <int> dimension num_devices */
                   void **array_d_work                    /* <t> num_devices, host array */
                                                          /* array_d_work[j] points to device workspace of device j */
)
{
  int currentDev = 0; /* record current device ID */
  CUDA_CHECK(cudaGetDevice(&currentDev));

  for (int idx = 0; idx < num_devices; idx++)
  {
    int deviceId = deviceIdA[idx];
    /* WARNING: we need to set device before any runtime API */
    CUDA_CHECK(cudaSetDevice(deviceId));

    if (NULL != array_d_work[idx])
    {
      CUDA_CHECK(cudaFree(array_d_work[idx]));
    }
  }
  CUDA_CHECK(cudaSetDevice(currentDev));
}
