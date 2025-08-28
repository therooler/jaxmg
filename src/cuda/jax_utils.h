#include "jaxlib/ffi_helpers.h"
#include "xla/ffi/api/ffi.h"

#include "absl/status/statusor.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/gpu/handle_pool.h"
#include "third_party/gpus/cuda/include/cusolverMg.h"

#ifndef JAXLIB_GPU_SOLVER_HANDLE_POOL_H_
#define JAXLIB_GPU_SOLVER_HANDLE_POOL_H_

namespace jax
{

  using SolverHandlePool = HandlePool<cusolverMgHandle_t, gpuStream_t>;

  template <>
  absl::StatusOr<SolverHandlePool::Handle> SolverHandlePool::Borrow(
      gpuStream_t stream);

  template <typename T>
  inline absl::StatusOr<T *> AllocateWorkspaceBytes(
      ::xla::ffi::ScratchAllocator &scratch, int64_t n_bytes,
      std::string_view name)
  {
    auto maybe_workspace = scratch.Allocate(n_bytes);
    if (!maybe_workspace.has_value())
    {
      return absl::Status(
          absl::StatusCode::kResourceExhausted,
          absl::StrFormat("Unable to allocate workspace for %s", name));
    }
    return static_cast<T *>(maybe_workspace.value());
  }
} // namespace jax

#endif // JAXLIB_GPU_SOLVER_HANDLE_POOL_H_