/*
 * Copyright 2020 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

// C++
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <mutex>
#include <string>
#include <cstdio>
#include <iostream>
// Abseil
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
// Jaxlib
#include "jaxlib/ffi_helpers.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/vendor.h"
// XLA
#include "xla/ffi/api/ffi.h"
// CUDA
#include "third_party/gpus/cuda/include/cusolverMg.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/gpus/cuda/include/cuda_runtime.h"
// My Code
#include "jax_utils.h"
#include "cusolver_utils.h"
#include "shm.h"

DynamicBarrier sync_point;

namespace jax
{
    namespace JAX_GPU_NAMESPACE
    {
        namespace ffi = ::xla::ffi;

#define SOLVER_DISPATCH_IMPL(impl, ...)   \
    switch (dataType)                     \
    {                                     \
    case ffi::F32:                        \
        return impl<float>(__VA_ARGS__);  \
    case ffi::F64:                        \
        return impl<double>(__VA_ARGS__); \
    default:                              \
        break;                            \
    }
        template <typename data_type>
        ffi::Error PotrsMgImpl(int64_t N, int64_t NRHS, int64_t batch_a,
                               gpuStream_t stream, ffi::ScratchAllocator &scratch,
                               ffi::AnyBuffer a, ffi::AnyBuffer b, int64_t tile_size,
                               ffi::Result<ffi::AnyBuffer> out, ffi::Result<ffi::Buffer<ffi::S32>> status)
        {
            /* misc */
            bool VERBOSE = false;                 // print matrices for debugging
            const std::string &source = __FILE__; // file name for error messages

            /* GPU */
            const int MAX_NUM_DEVICES = 16; // cusolverMg can handle 16 GPUs at most
            int nbGpus = 0;                 // number of GPUs to use
            int currentDevice = 0;          // current GPU
            CUDA_CHECK_OR_RETURN(cudaGetDeviceCount(&nbGpus));
            CUDA_CHECK_OR_RETURN(cudaGetDevice(&currentDevice));
            if (nbGpus > MAX_NUM_DEVICES)
            {
                return ffi::Error::InvalidArgument(
                    absl::StrFormat("%s: Number of Gpus must be <=16, received %d", source, nbGpus));
            }
            std::vector<int> deviceList(nbGpus); // list of device IDs

            /* data */
            auto array_data_A = static_cast<data_type *>(a.untyped_data()); // XLA device pointer for a
            auto array_data_b = static_cast<data_type *>(b.untyped_data());
            auto out_data = static_cast<data_type *>(out->untyped_data());

            /* Tiling sizes */
            const int IA = 1; // index within a global matrix, base-1 (not used)
            const int JA = 1;
            const int T_A = std::min(tile_size, batch_a); // tile size of A
            const int lda = N;                            // leading dimension of local A

            const int IB = 1; // index within a global matrix, base-1 (not used)
            const int JB = 1;

            if (NRHS > 256)
            {
                return ffi::Error::InvalidArgument(
                    absl::StrFormat("%s: Number of right hand sides must be <=256, received %d, "
                                    "this may be improved in the next release",
                                    source, NRHS));
            }
            const int T_B = static_cast<int>(NRHS); // tile size of B
            const int ldb = N;                      // leading dimension of local b

            /* CUDA */
            cudaDataType compute_type = traits<data_type>::cuda_data_type; // Data type for computation
            cudaLibMgMatrixDesc_t descrA; // CusolverMg matrix descriptors
            cudaLibMgMatrixDesc_t descrB;
            cudaLibMgGrid_t gridA; // CusolverMg grid descriptors
            cudaLibMgGrid_t gridB;
            cusolverMgGridMapping_t mapping = CUDALIBMG_GRID_MAPPING_COL_MAJOR;         // Column major a la Scalapack
            FFI_ASSIGN_OR_RETURN(auto cusolverHPool, SolverHandlePool::Borrow(stream)); // Assign a cusolver handle from the pool
            cusolverMgHandle_t cusolverH = cusolverHPool.get();
            int info = 0;                            // Info used by cusolverMg calls
            cusolverStatus_t cusolver_status;        // Return status of cusolverMg calls
            int cusolver_status_host;
            auto status_data = status->typed_data(); // Status returned by potrf
            int64_t lwork_potrf = 0;                 // Workspace size used by cusolverMg calls
            int64_t lwork_potrs = 0;

            /* Shared memory */
            static std::once_flag barrier_initialized; // Initialize barrier once between threads
            std::call_once(barrier_initialized, [&]()
                           { sync_point.initialize(nbGpus); });
            sync_point.arrive_and_wait();
            sharedMemoryInfo shminfoA; // Shared memory info for device pointers to local matrices
            sharedMemoryInfo shminfoB;
            sharedMemoryInfo shminfowork;
            sharedMemoryInfo shminfolwork; // Shared memory info for lwork space nbytes

            data_type **shmA = get_shm_device_ptrs<data_type>(currentDevice, sync_point, shminfoA, "shmA"); // Actual shared memory
            data_type **shmB = get_shm_device_ptrs<data_type>(currentDevice, sync_point, shminfoB, "shmB");
            data_type **shmwork = get_shm_device_ptrs<data_type>(currentDevice, sync_point, shminfowork, "shmwork");
            int64_t *shmlwork = (int64_t *)get_shm_lwork_ptr(currentDevice, sync_point, shminfolwork, "shmlwork");

            if (currentDevice == 0)
            {
                // std::printf("Step 1: Create Mg handle and select devices (thread 0 only)\n");
                CUSOLVER_CHECK_OR_RETURN(cusolverMgCreate(&cusolverH));

                for (int j = 0; j < nbGpus; j++)
                {
                    deviceList[j] = j;
                    cudaDeviceProp prop;
                    CUDA_CHECK_OR_RETURN(cudaGetDeviceProperties(&prop, j));
                    if (VERBOSE)
                    {
                        std::printf("\tThere are %d GPUs \n", nbGpus);
                        std::printf("\tDevice %d, %s, cc %d.%d \n", j, prop.name, prop.major, prop.minor);
                        std::printf("T_A: %d \n", T_A);
                        std::printf("T_B: %d \n", T_B);
                    }
                }

                CUSOLVER_CHECK_OR_RETURN(cusolverMgDeviceSelect(cusolverH, nbGpus, deviceList.data()));

                // std::printf("Step 2: Enable peer access \n");
                // enablePeerAccess(nbGpus, deviceList.data());

                // std::printf("Step 3: Create matrix descriptors for A and D \n");

                CUSOLVER_CHECK_OR_RETURN(cusolverMgCreateDeviceGrid(&gridA, 1, nbGpus, deviceList.data(), mapping));
                CUSOLVER_CHECK_OR_RETURN(cusolverMgCreateDeviceGrid(&gridB, 1, nbGpus, deviceList.data(), mapping));

                /* (global) A is N-by-N */
                CUSOLVER_CHECK_OR_RETURN(cusolverMgCreateMatrixDesc(&descrA, N, /* number of rows of (global) A */
                                                                    N,          /* number of columns of (global) A */
                                                                    N,          /* number or rows in a tile */
                                                                    T_A,        /* number of columns in a tile */
                                                                    compute_type, gridA));

                /* (global) B is N-by-NRHS */
                CUSOLVER_CHECK_OR_RETURN(cusolverMgCreateMatrixDesc(&descrB, N, /* number of rows of (global) B */
                                                                    NRHS,       /* number of columns of (global) B */
                                                                    N,          /* number or rows in a tile */
                                                                    T_B,        /* number of columns in a tile */
                                                                    compute_type, gridB));
            }
            if (VERBOSE)
            {
                std::printf("Step 3: Print data on host \n");
                std::vector<data_type> A(batch_a * N, 0);
                gpuMemcpy(A.data(), array_data_A, batch_a * N * sizeof(data_type), gpuMemcpyDeviceToHost);
                for (int i = 0; i < batch_a * N; i++)
                {
                    std::cout << A[i] << std::endl;
                }

                std::vector<data_type> B(batch_a * NRHS, 0);
                gpuMemcpy(B.data(), array_data_b, batch_a * NRHS * sizeof(data_type), gpuMemcpyDeviceToHost);

                std::printf("%d: A = matlab base-1\n", currentDevice);

                print_matrix(N, batch_a, A.data(), N);
                std::printf("%d: b = matlab base-1\n", currentDevice);
                print_matrix(NRHS, batch_a, B.data(), NRHS);
            }
            // std::printf("Step 4: Allocate distributed matrices A and B \n");

            // int64_t nbytes_A = getWorkspaceBytesT_A<data_type>(nbGpus, N, T_A, lda);
            // int64_t nbytes_B = getWorkspaceBytesT_A<data_type>(nbGpus, N, T_B, ldb);

            /* A := 0 */

            // FFI_ASSIGN_OR_RETURN(auto workspaceA, AllocateWorkspaceBytes<data_type>(scratch, nbytes_A, "workspaceA"));
            // CUDA_CHECK_OR_RETURN(cudaMemset(workspaceA, 0, nbytes_A));
            // shmA[currentDevice] = workspaceA;
            /* B := 0 */
            // FFI_ASSIGN_OR_RETURN(auto workspaceB, AllocateWorkspaceBytes<data_type>(scratch, nbytes_B, "workspaceB"));
            // CUDA_CHECK_OR_RETURN(cudaMemset(workspaceA, 0, nbytes_A));
            // shmB[currentDevice] = workspaceB;
            // CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
            // sync_point.arrive_and_wait();
            /*
            The example has the NxN matrix in host memory and then distributes it in chunks over the GPUs
            We have chunks of size NxBatch in device memory, and need to somehow move this to the expected layout
            for PotRf.
            */
            // std::printf("Step 8: Relayout data \n");
            memcpyCyclicShard<data_type>(nbGpus, deviceList.data(), N, batch_a,
                                         /* input */
                                         array_data_A, lda,
                                         /* output */
                                         N,           /* number of columns of global A */
                                         T_A,         /* number of columns per column tile */
                                         lda,         /* leading dimension of local A */
                                         array_data_A /* device pointer for shard on device */
            );
            shmA[currentDevice] = array_data_A;
            if (currentDevice == 0)
            {
                // std::cout << "memcpyH2D b" << std::endl;
                memcpyCyclicShard<data_type>(nbGpus, deviceList.data(), N, NRHS,
                                             /* input */
                                             array_data_b, ldb,
                                             /* output */
                                             1,           /* number of columns of global A */
                                             T_B,         /* number of columns per column tile */
                                             ldb,         /* leading dimension of local A */
                                             array_data_b /* device pointer for shard on device */
                );
                shmB[currentDevice] = array_data_b;
            }

            CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
            sync_point.arrive_and_wait();

            // std::printf("Step 9: Allocate workspace \n");
            if (currentDevice == 0)
            {
                CUSOLVER_CHECK_OR_RETURN(cusolverMgPotrf_bufferSize(cusolverH, CUBLAS_FILL_MODE_LOWER, N,
                                                                    //   reinterpret_cast<void **>(array_d_A.data()), IA, /* base-1 */
                                                                    reinterpret_cast<void **>(shmA), IA, /* base-1 */
                                                                    JA,                                  /* base-1 */
                                                                    descrA, compute_type, &lwork_potrf));

                CUSOLVER_CHECK_OR_RETURN(cusolverMgPotrs_bufferSize(cusolverH, CUBLAS_FILL_MODE_LOWER, N, NRHS, /* NRHS */
                                                                                                                //   reinterpret_cast<void **>(array_d_A.data()), IA, JA,
                                                                    reinterpret_cast<void **>(shmA), IA, JA,
                                                                    //   descrA, reinterpret_cast<void **>(array_d_B.data()),
                                                                    descrA, reinterpret_cast<void **>(shmB),
                                                                    IB, JB, descrB, compute_type,
                                                                    &lwork_potrs));
                *shmlwork = std::max(lwork_potrf, lwork_potrs);
            }
            sync_point.arrive_and_wait();
            // std::printf("\t%d: Allocate device workspace, lwork = %lld \n", currentDevice, static_cast<long long>(*shmlwork));

            /* array_d_work[j] points to device workspace of device j */
            FFI_ASSIGN_OR_RETURN(auto workspace, AllocateWorkspaceBytes<data_type>(scratch, sizeof(data_type) * (*shmlwork), "workspace_potrf"));
            shmwork[currentDevice] = workspace;

            /* sync all devices */
            CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
            sync_point.arrive_and_wait();

            if (currentDevice == 0)
            {
                // std::printf("Step 10: Solve A*X = B by POTRF and POTRS \n");
                cusolver_status = cusolverMgPotrf(
                    cusolverH, CUBLAS_FILL_MODE_LOWER, N,
                    reinterpret_cast<void **>(shmA), IA, JA,
                    descrA, compute_type,
                    reinterpret_cast<void **>(shmwork), *shmlwork, &info);

                if (cusolver_status != CUSOLVER_STATUS_SUCCESS)
                {
                    cusolver_status_host = static_cast<int>(cusolver_status);
                    JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpyAsync(
                        status_data, &cusolver_status_host, sizeof(cusolver_status_host), gpuMemcpyHostToDevice, stream));
                }

                /* sync all devices */
                CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());

                /* check if A is singular */
                if (0 > info)
                {
                    return ffi::Error::Internal(
                        absl::StrFormat("unexpected error in cusolverMgPotrf, %d-th input parameter is wrong \n", -info));
                }

                cusolver_status = cusolverMgPotrs(cusolverH, CUBLAS_FILL_MODE_LOWER, N, NRHS, /* NRHS */
                                                  reinterpret_cast<void **>(shmA), IA, JA, descrA,
                                                  reinterpret_cast<void **>(shmB), IB, JB, descrB,
                                                  compute_type,
                                                  reinterpret_cast<void **>(shmwork), *shmlwork,
                                                  &info);

                if (cusolver_status != CUSOLVER_STATUS_SUCCESS)
                {
                    cusolver_status_host = -static_cast<int>(cusolver_status);
                    JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpyAsync(
                        status_data, &cusolver_status_host, sizeof(cusolver_status_host), gpuMemcpyHostToDevice, stream));
                }
                /* sync all devices */
                CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());

                /* check if parameters are valid */
                if (0 > info)
                {
                    return ffi::Error::Internal(
                        absl::StrFormat("unexpected error in cusolverMgPotrs, %d-th input parameter is wrong \n", -info));
                }
            }
            /* sync all devices */
            CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
            sync_point.arrive_and_wait();

            // std::printf("Step 11: Solution vector B \n");

            JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpyAsync(
                out_data, shmB[0], b.size_bytes(), gpuMemcpyDeviceToDevice, stream));

            if (currentDevice == 0)
            {
                // std::printf("Step 12: Free resources \n");

                CUSOLVER_CHECK_OR_RETURN(cusolverMgDestroyMatrixDesc(descrA));
                CUSOLVER_CHECK_OR_RETURN(cusolverMgDestroyMatrixDesc(descrB));

                CUSOLVER_CHECK_OR_RETURN(cusolverMgDestroyGrid(gridA));
                CUSOLVER_CHECK_OR_RETURN(cusolverMgDestroyGrid(gridB));

                CUSOLVER_CHECK_OR_RETURN(cusolverMgDestroy(cusolverH));
                if (currentDevice == 0)
                {
                    sharedMemoryClose(&shminfoA);
                    sharedMemoryClose(&shminfoB);
                    sharedMemoryClose(&shminfowork);
                    sharedMemoryClose(&shminfolwork);
                    // std::printf("%d: Shared memory destroyed\n", currentDevice);
                }
            }
            CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
            sync_point.arrive_and_wait();

            return ffi::Error::Success();
        }

        ffi::Error PotrsMgDispatch(gpuStream_t stream, ffi::ScratchAllocator scratch,
                                   ffi::AnyBuffer a, ffi::AnyBuffer b, int64_t tile_size,
                                   ffi::Result<ffi::AnyBuffer> out, ffi::Result<ffi::Buffer<ffi::S32>> status)
        {
            auto dataType = a.element_type();

            // Columns are batched
            FFI_ASSIGN_OR_RETURN((const auto [N, batch_a]), SplitBatch1D(a.dimensions()));
            FFI_ASSIGN_OR_RETURN((const auto [N_b, NRHS]), SplitBatch1D(b.dimensions()));
            FFI_RETURN_IF_ERROR(CheckShape(b.dimensions(), {N, NRHS}, "b", "potrf"));

            if (dataType != out->element_type())
            {
                return ffi::Error::InvalidArgument(
                    "The input and output to getrf must have the same element type");
            }
            FFI_RETURN_IF_ERROR(CheckShape(status->dimensions(), 1, "status", "potrf"));

            SOLVER_DISPATCH_IMPL(PotrsMgImpl, N, NRHS, batch_a, stream, scratch, a, b, tile_size, out, status);

            return ffi::Error::InvalidArgument(absl::StrFormat(
                "Unsupported data type%s for potrf", absl::FormatStreamed(dataType)));
        }

        XLA_FFI_DEFINE_HANDLER_SYMBOL(PotrsMgFFI, PotrsMgDispatch,
                                      ffi::Ffi::Bind()
                                          .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                          .Ctx<ffi::ScratchAllocator>()
                                          .Arg<ffi::AnyBuffer>()        // A
                                          .Arg<ffi::AnyBuffer>()        // b
                                          .Attr<int64_t>("T_A")         // tile size
                                          .Ret<ffi::AnyBuffer>()        // x
                                          .Ret<ffi::Buffer<ffi::S32>>() // status
        );

#undef SOLVER_DISPATCH_IMPL

    } // namespace JAX_GPU_NAMESPACE
} // namespace jax