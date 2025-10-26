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
#include "xla/ffi/api/c_api.h"
// CUDA
#include "third_party/gpus/cuda/include/cusolverMg.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/gpus/cuda/include/cuda_runtime.h"
#include "third_party/gpus/cuda/include/cuda.h"
// My Code
#include "jax_utils.h"
#include "cusolver_utils.h"
#include "shm.h"
#include "process_barrier.h"
#include "ipc_utils.h"

namespace jax
{
    namespace JAX_GPU_NAMESPACE
    {
        namespace ffi = ::xla::ffi;

#define SOLVER_DISPATCH_IMPL(impl, ...)            \
    switch (dataType)                              \
    {                                              \
    case ffi::F32:                                 \
        return impl<float>(__VA_ARGS__);           \
    case ffi::F64:                                 \
        return impl<double>(__VA_ARGS__);          \
    case ffi::C64:                                 \
        return impl<cuFloatComplex>(__VA_ARGS__);  \
    case ffi::C128:                                \
        return impl<cuDoubleComplex>(__VA_ARGS__); \
    default:                                       \
        break;                                     \
    }
        template <typename data_type>
        ffi::Error PotrsMgImpl(int64_t N, int64_t NRHS, int64_t batch_a,
                               gpuStream_t stream, ffi::ScratchAllocator &scratch,
                               ffi::AnyBuffer a, ffi::AnyBuffer b, int64_t tile_size,
                               ffi::Result<ffi::AnyBuffer> out, ffi::Result<ffi::Buffer<ffi::S32>> status)
        {
            /* misc */
            const std::string &source = __FILE__; // file name for error messages

            /* GPU */
            const int MAX_NUM_DEVICES = 16; // cusolverMg can handle 16 GPUs at most
            int nbGpus = 0;                 // number of GPUs to use
            int currentDevice = 0;          // current GPU
            CUDA_CHECK_OR_RETURN(cudaGetDeviceCount(&nbGpus));
            CUDA_CHECK_OR_RETURN(cudaGetDevice(&currentDevice));
            std::printf("Number of GPUs: %d\n", nbGpus);
            std::printf("currentDevice: %d\n", currentDevice);

            print_current_context();
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
            cudaLibMgMatrixDesc_t descrA;                                  // CusolverMg matrix descriptors
            cudaLibMgMatrixDesc_t descrB;
            cudaLibMgGrid_t gridA; // CusolverMg grid descriptors
            cudaLibMgGrid_t gridB;
            cusolverMgGridMapping_t mapping = CUDALIBMG_GRID_MAPPING_COL_MAJOR;         // Column major a la Scalapack
            FFI_ASSIGN_OR_RETURN(auto cusolverHPool, SolverHandlePool::Borrow(stream)); // Assign a cusolver handle from the pool
            cusolverMgHandle_t cusolverH = cusolverHPool.get();
            int info = 0;                            // Info used by cusolverMg calls
            cusolverStatus_t cusolver_status;        // Return status of cusolverMg calls
            auto status_data = status->typed_data(); // Status returned by potrf
            int64_t lwork_potrf = 0;                 // Workspace size used by cusolverMg calls
            int64_t lwork_potrs = 0;

            /* Shared memory */
            DynamicBarrier sync_point(nbGpus, "/jaxmgbarrier");
            sync_point.arrive_and_wait();
            CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());

            sharedMemoryInfo shminfoA;    // Shared memory info for device pointers to local matrices
            sharedMemoryInfo shminfoAipc; // Shared memory info for device pointers to local matrices
            sharedMemoryInfo shminfoB;
            sharedMemoryInfo shminfowork;
            sharedMemoryInfo shminfolwork; // Shared memory info for lwork space nbytes
            sharedMemoryInfo shmcsh;       // Shared memory info for cusolver status
            sharedMemoryInfo shminfoshmoffsetA;

            std::vector<data_type *> shmA(nbGpus, nullptr);
            cudaIpcMemHandle_t *shmAipc = get_shm_ipc_handles(currentDevice, sync_point, shminfoAipc, "shmAipc");
            size_t *shmoffsetA = get_shm_lwork_ptr<size_t>(currentDevice, sync_point, shminfoshmoffsetA, "shmoffsetA");

            ipcGetHandleAndOffset(array_data_A, shmAipc[currentDevice], shmoffsetA[currentDevice]);

            CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
            sync_point.arrive_and_wait();
            if (currentDevice == 0)
            {
                shmA = ipcGetDevicePointers<data_type>(currentDevice, nbGpus, shmAipc, shmoffsetA);
                shmA[0] = array_data_A;
                // printf("DEV %d\n_________\n", 0);
                // shmA[0] = array_data_A;
                // print_pointer_info(shmA[0]);
                // for (int dev = 1; dev < nbGpus; dev++)
                // {
                //     void *opened = nullptr;
                //     CUDA_CHECK_OR_RETURN(cudaIpcOpenMemHandle(&opened, *(cudaIpcMemHandle_t *)&shmAipc[dev], cudaIpcMemLazyEnablePeerAccess));
                //     CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
                //     shmA[dev] = reinterpret_cast<data_type *>(reinterpret_cast<char *>(opened) + 2304);
                //     printf("DEV %d\n_________\n", dev);
                //     print_pointer_info(shmA[dev]);
                //     // Printer
                //     std::vector<typename traits<data_type>::T> host(N * batch_a);
                //     size_t numBytes = sizeof(data_type) * N * batch_a;
                //     CUDA_CHECK_OR_RETURN(cudaMemcpy(host.data(), shmA[dev], numBytes, cudaMemcpyDeviceToHost));
                //     CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
                //     print_matrix(N, batch_a, host.data(), N);
                // }
            }
            // std::vector<typename traits<data_type>::T> host(N * batch_a);
            // size_t numBytes = sizeof(data_type) * N * batch_a;
            // CUDA_CHECK_OR_RETURN(cudaMemcpy(host.data(), shmA[currentDevice], numBytes, cudaMemcpyDeviceToHost));
            // CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
            // print_matrix(N, batch_a, host.data(), N);
            // barrierWait(&shm->barrier, &shm->sense, (unsigned int)nbGpus);

            // WORKS, but can we avoid the copy? //
            // void *device_array = nullptr;
            // cudaMalloc(reinterpret_cast<void **>(&device_array), numBytes);
            // CUDA_CHECK_OR_RETURN(cudaMemcpy(device_array, array_data_A, numBytes, cudaMemcpyDeviceToDevice));

            // // DOES NOT WORK, cannot just cast to new pointer.
            // // void *device_array = array_data_A;
            // CUmemorytype memType{};
            // int is_managed = 0;
            // int is_legacy_ipc = 0;
            // CUmemoryPool pool = nullptr;
            // CUdeviceptr canonical = 0;
            // CUdeviceptr range_base = 0;
            // size_t range_size = 0;
            // int owning_device;
            // // Check attributes
            // cuPointerGetAttribute(&memType, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (CUdeviceptr)device_array);
            // cuPointerGetAttribute(&is_managed, CU_POINTER_ATTRIBUTE_IS_MANAGED, (CUdeviceptr)device_array);
            // cuPointerGetAttribute(&is_legacy_ipc, CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE, (CUdeviceptr)device_array);
            // cuPointerGetAttribute(&pool, CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE, (CUdeviceptr)device_array);
            // // Get underlying device ptr
            // cuPointerGetAttribute(&canonical, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, (CUdeviceptr)device_array);
            // cuPointerGetAttribute(&range_base, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, (CUdeviceptr)canonical);
            // cuPointerGetAttribute(&range_size, CU_POINTER_ATTRIBUTE_RANGE_SIZE, (CUdeviceptr)canonical);
            // cuPointerGetAttribute(&owning_device, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, (CUdeviceptr)device_array);
            // size_t offset = static_cast<size_t>(
            //     (uintptr_t)canonical - (uintptr_t)range_base);
            // printf("Owning: device=%d\n", owning_device);
            // printf("A ptr orig=%p canonical=%p\n", (void *)device_array, (void *)canonical);
            // printf("size=%zu offset=%zu\n", range_size, offset);
            // printf("memType=%d - (DEVICE=%d) managed=%d legacy_ipc=%d mempool=%p canonical=0x%llx\n",
            //        (int)memType, (int)CU_MEMORYTYPE_DEVICE, is_managed, is_legacy_ipc, (void *)pool,
            //        (unsigned long long)canonical);

            // cudaDeviceSynchronize();
            // // Shared memory
            // volatile shmStruct *shm = NULL;
            // sharedMemoryInfo structinfo;
            // if (currentDevice == 0)
            // {
            //     if (sharedMemoryCreate("TEST", sizeof(*shm), &structinfo) != 0)
            //     {
            //         printf("Failed to create shared memory slab\n");
            //         exit(EXIT_FAILURE);
            //     }
            //     shm = (volatile shmStruct *)structinfo.addr;
            //     memset((void *)shm, 0, sizeof(*shm));
            // }
            // else
            // {
            //     sleep(1);
            //     if (sharedMemoryOpen("TEST", sizeof(shmStruct), &structinfo) != 0)
            //     {
            //         printf("Failed to create shared memory slab\n");
            //         exit(EXIT_FAILURE);
            //     }
            //     shm = (volatile shmStruct *)structinfo.addr;
            // }

            // // Assigning handles to shared memory
            // cudaError_t ev = cudaIpcGetMemHandle((cudaIpcMemHandle_t *)&shm->memHandle[currentDevice], device_array);
            // barrierWait(&shm->barrier, &shm->sense, (unsigned int)nbGpus);
            // if (ev == cudaSuccess)
            // {
            //     printf("cudaIpcGetMemHandle succesful\n");
            // }
            // if (currentDevice == 0)
            // {
            //     shmA[0] = array_data_A;
            //     printf("Device %d is going to open handle 1\n", currentDevice);
            //     void *opened = nullptr;
            //     CUDA_CHECK_OR_RETURN(cudaIpcOpenMemHandle(&opened, *(cudaIpcMemHandle_t *)&shm->memHandle[1], cudaIpcMemLazyEnablePeerAccess));
            //     int owner = -1;
            //     CUdeviceptr range_start = 0;
            //     size_t range_size = 0;

            //     CUresult rc;
            //     rc = cuPointerGetAttribute(&owner, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, (CUdeviceptr)opened);
            //     rc = cuPointerGetAttribute(&range_start, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, (CUdeviceptr)opened);
            //     rc = cuPointerGetAttribute(&range_size, CU_POINTER_ATTRIBUTE_RANGE_SIZE, (CUdeviceptr)opened);

            //     printf("owner device=%d  range=[%p, +%zu)\n", owner, (void *)range_start, range_size);
            //     cudaPointerAttributes open_attr{};
            //     cudaPointerGetAttributes(&open_attr, opened);
            //     printf("opened pointer: device=%d type=%d\n", open_attr.device, open_attr.type);
            //     shmA[1] = static_cast<data_type *>(opened);
            //     std::vector<typename traits<data_type>::T> host_array_opened(N * batch_a);
            //     CUDA_CHECK_OR_RETURN(cudaMemcpy(host_array_opened.data(), opened, numBytes, cudaMemcpyDeviceToHost));
            //     cudaDeviceSynchronize();
            //     print_matrix(N, batch_a, host_array_opened.data(), N);
            //     CUDA_CHECK_OR_RETURN(cudaMemcpyPeer(
            //         /*dst*/ device_array, currentDevice,
            //         /*src*/ opened, 1,
            //         numBytes));
            //     cudaDeviceSynchronize();
            // }

            // std::vector<typename traits<data_type>::T> host_array_swapped(N * batch_a);
            // CUDA_CHECK_OR_RETURN(cudaMemcpy(host_array_swapped.data(), device_array, numBytes, cudaMemcpyDeviceToHost));
            // cudaDeviceSynchronize();
            // print_matrix(N, batch_a, host_array_swapped.data(), N);
            // // Cleanup
            // cudaFree(device_array);

            // data_type **shmA = get_shm_device_ptrs<data_type>(currentDevice, sync_point, shminfoA, "shmA"); // Actual shared memory
            // data_type **shmB = get_shm_device_ptrs<data_type>(currentDevice, sync_point, shminfoB, "shmB");
            // data_type **shmwork = get_shm_device_ptrs<data_type>(currentDevice, sync_point, shminfowork, "shmwork");

            // int32_t *cusolver_status_host = get_shm_lwork_ptr<int32_t>(currentDevice, sync_point, shmcsh, "shmcsh");
            // int64_t *shmlwork = get_shm_lwork_ptr<int64_t>(currentDevice, sync_point, shminfolwork, "shmlwork");

            // if (currentDevice == 0)
            // {
            //     CUSOLVER_CHECK_OR_RETURN(cusolverMgCreate(&cusolverH));
            //     for (int j = 0; j < nbGpus; j++)
            //     {
            //         deviceList[j] = j;
            //         cudaDeviceProp prop;
            //         CUDA_CHECK_OR_RETURN(cudaGetDeviceProperties(&prop, j));
            //         std::printf("\tThere are %d GPUs \n", nbGpus);
            //         std::printf("\tDevice %d, %s, cc %d.%d \n", j, prop.name, prop.major, prop.minor);
            //     }

            //     CUSOLVER_CHECK_OR_RETURN(cusolverMgDeviceSelect(cusolverH, nbGpus, deviceList.data()));

            //     CUSOLVER_CHECK_OR_RETURN(cusolverMgCreateDeviceGrid(&gridA, 1, nbGpus, deviceList.data(), mapping));
            //     CUSOLVER_CHECK_OR_RETURN(cusolverMgCreateDeviceGrid(&gridB, 1, nbGpus, deviceList.data(), mapping));

            //     /* (global) A is N-by-N */
            //     CUSOLVER_CHECK_OR_RETURN(cusolverMgCreateMatrixDesc(&descrA, N, /* number of rows of (global) A */
            //                                                         N,          /* number of columns of (global) A */
            //                                                         N,          /* number or rows in a tile */
            //                                                         T_A,        /* number of columns in a tile */
            //                                                         compute_type, gridA));

            //     /* (global) B is N-by-NRHS */
            //     CUSOLVER_CHECK_OR_RETURN(cusolverMgCreateMatrixDesc(&descrB, N, /* number of rows of (global) B */
            //                                                         NRHS,       /* number of columns of (global) B */
            //                                                         N,          /* number or rows in a tile */
            //                                                         T_B,        /* number of columns in a tile */
            //                                                         compute_type, gridB));
            // }
            // volatile shmStruct *shm = NULL;
            // sharedMemoryInfo structinfo;
            // if (currentDevice == 0)
            // {
            //     if (sharedMemoryCreate("TEST", sizeof(*shm), &structinfo) != 0)
            //     {
            //         printf("Failed to create shared memory slab\n");
            //         exit(EXIT_FAILURE);
            //     }
            //     shm = (volatile shmStruct *)structinfo.addr;
            //     memset((void *)shm, 0, sizeof(*shm));
            // }
            // else
            // {
            //     sleep(1);
            //     if (sharedMemoryOpen("TEST", sizeof(shmStruct), &structinfo) != 0)
            //     {
            //         printf("Failed to create shared memory slab\n");
            //         exit(EXIT_FAILURE);
            //     }
            //     shm = (volatile shmStruct *)structinfo.addr;
            // }

            // CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
            // // sync_point.arrive_and_wait();
            // barrierWait(&shm->barrier, &shm->sense, (unsigned int)nbGpus);
            // memcpyCyclicShard<data_type>(nbGpus, stream, deviceList.data(), N, batch_a,
            //                              /* input */
            //                              array_data_A, lda,
            //                              /* output */
            //                              N,           /* number of columns of global A */
            //                              T_A,         /* number of columns per column tile */
            //                              lda,         /* leading dimension of local A */
            //                              array_data_A /* device pointer for shard on device */
            // );
            // // physical device owning the allocation
            // cudaDeviceProp prop{};
            // CUDA_CHECK_OR_RETURN(cudaGetDeviceProperties(&prop, currentDevice));
            // if (!prop.unifiedAddressing)
            // {
            //     return ffi::Error::Internal(absl::StrFormat(
            //         "Device %d has no UVA; CUDA IPC memory unsupported", currentDevice));
            // }

            // // sharedMemoryInfo shminfoAIPC;                                                                          // Shared memory info for device pointers to local matrices
            // // cudaIpcMemHandle_t *shmA_IPC = get_shm_ipc_handles(currentDevice, sync_point, shminfoAIPC, "shmAIPC"); //
            // // cudaIpcMemHandle_t myAHandle;
            // pid_t pid;
            // pid = getppid();
            // char pidString[20] = {0};
            // snprintf(pidString, sizeof(pidString), "%d", pid);
            // printf("PPID: %s\n", pidString);
            // printf("PID: %d\n", (int)getpid());

            // // CUDA_CHECK_OR_RETURN(cudaMalloc(&ptr, 128));
            // cudaPointerAttributes a_attr{};
            // cudaPointerGetAttributes(&a_attr, array_data_A);
            // printf("local A owner dev = %d\n", a_attr.device);
            // printf("dev %d: local array_data_A=%p\n", currentDevice, (const void *)array_data_A);
            // std::printf("Allocating at index %d", currentDevice);
            // cudaDeviceProp p;
            // cudaGetDeviceProperties(&p, currentDevice);
            // printf("dev %d: pciBusId=%02x, pciDeviceId=%02x, uuid=",
            //        currentDevice, (unsigned)p.pciBusID, (unsigned)p.pciDeviceID);
            // for (int i = 0; i < 16; ++i)
            // {
            //     printf("%02x", (unsigned char)p.uuid.bytes[i]);
            // }
            // printf("\n");
            // cudaEvent_t ready;
            // CUDA_CHECK_OR_RETURN(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming | cudaEventInterprocess));
            // CUDA_CHECK_OR_RETURN(cudaEventRecord(ready, /*stream=*/0));
            // CUDA_CHECK_OR_RETURN(cudaIpcGetEventHandle((cudaIpcEventHandle_t *)&shm->eventHandle[currentDevice], ready));

            // // Export memory handle (check errors)
            // cudaError_t ev = cudaIpcGetMemHandle((cudaIpcMemHandle_t *)&shm->memHandle[currentDevice], array_data_A);
            // if (ev != cudaSuccess)
            // {
            //     std::fprintf(stderr, "cudaIpcGetMemHandle(dev=%d) failed: %s\n", currentDevice, cudaGetErrorString(ev));
            //     return ffi::Error::Internal("cudaIpcGetMemHandle failed");
            // }

            // CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
            // if (currentDevice == 0)
            // {
            //     printf("Waiting for a bit here");
            //     sleep(1);
            //     printf("Slept for 1[s]\n");
            // }
            // std::vector<typename traits<data_type>::T> ev_print(N * batch_a);

            // JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpy(
            //     ev_print.data(), array_data_A, sizeof(data_type) * N * batch_a, gpuMemcpyDeviceToHost));
            // CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
            // // sync_point.arrive_and_wait();
            // print_matrix(N, batch_a, ev_print.data(), N);
            // // sync_point.arrive_and_wait();
            // barrierWait(&shm->barrier, &shm->sense, (unsigned int)nbGpus);

            // if (currentDevice == 0)
            // {
            //     // shmA[currentDevice] = array_data_A;
            //     // Open handles for all other GPUs (guard by actual nbGpus)
            //     for (int peer = 1; peer < nbGpus; ++peer)
            //     {
            //         cudaSetDevice(peer);
            //         void *opened = nullptr;
            //         CUDA_CHECK_OR_RETURN(cudaIpcOpenMemHandle(
            //             &opened, *(cudaIpcMemHandle_t *)&shm->memHandle[peer], cudaIpcMemLazyEnablePeerAccess));

            //         cudaEvent_t peerReady;
            //         CUDA_CHECK_OR_RETURN(cudaIpcOpenEventHandle(&peerReady, *(cudaIpcEventHandle_t *)&shm->eventHandle[peer]));

            //         cudaStream_t s;
            //         CUDA_CHECK_OR_RETURN(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
            //         CUDA_CHECK_OR_RETURN(cudaStreamWaitEvent(s, peerReady, 0));
            //         cudaPointerAttributes open_attr{};
            //         cudaPointerGetAttributes(&open_attr, opened);
            //         printf("opened pointer: device=%d type=%d\n", open_attr.device, open_attr.type);
            //         CUDA_CHECK_OR_RETURN(cudaMemcpyPeer(
            //             /*dst*/ array_data_A, currentDevice,
            //             /*src*/ opened, peer,
            //             sizeof(data_type) * N * batch_a));
            //         // Copy just the shard
            //         std::vector<typename traits<data_type>::T> host(N * batch_a);
            //         CUDA_CHECK_OR_RETURN(cudaMemcpyAsync(host.data(), opened, sizeof(data_type) * N * batch_a,
            //                                              cudaMemcpyDeviceToHost, s));
            //         CUDA_CHECK_OR_RETURN(cudaStreamSynchronize(s));
            //         print_matrix(N, batch_a, host.data(), N);

            //         CUDA_CHECK_OR_RETURN(cudaEventDestroy(peerReady));
            //         CUDA_CHECK_OR_RETURN(cudaIpcCloseMemHandle(opened));
            //         CUDA_CHECK_OR_RETURN(cudaStreamDestroy(s));
            //     }
            // }
            // cudaSetDevice(currentDevice);
            // printf("printing copied data");
            // JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpy(
            //     ev_print.data(), array_data_A, sizeof(data_type) * N * batch_a, gpuMemcpyDeviceToHost));
            // CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
            // // sync_point.arrive_and_wait();
            // print_matrix(N, batch_a, ev_print.data(), N);
            // CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
            // sync_point.arrive_and_wait();
            // std::vector<typename traits<data_type>::T> ev_print(N * batch_a);
            // JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpy(
            //     ev_print.data(), shmA[currentDevice], sizeof(data_type) * N * batch_a, gpuMemcpyDeviceToHost));
            // CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
            // sync_point.arrive_and_wait();
            // print_matrix(N, batch_a, ev_print.data(), N);
            // std::vector<typename traits<data_type>::T> ev_print(N * batch_a);
            // for (int dev = 0; dev < nbGpus; dev++)
            // {
            //     JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpy(
            //         ev_print.data(), shmA[dev], sizeof(data_type) * N * batch_a, gpuMemcpyDeviceToHost));
            //     CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
            //     sync_point.arrive_and_wait();
            //     print_matrix(N, batch_a, ev_print.data(), N);
            // }
            // // asign B on every device, even though solution will only be on device 0
            // memcpyShard<data_type>(nbGpus, N, NRHS,
            //                        /* input */
            //                        array_data_b, ldb,
            //                        /* output */
            //                        1,           /* number of columns of global A */
            //                        ldb,         /* leading dimension of local A */
            //                        array_data_b /* device pointer for shard on device */
            // );
            // shmB[currentDevice] = array_data_b;

            // CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
            // sync_point.arrive_and_wait();

            // if (currentDevice == 0)
            // {
            //     CUSOLVER_CHECK_OR_RETURN(cusolverMgPotri_bufferSize(cusolverH, CUBLAS_FILL_MODE_LOWER, N,
            //                                                         //   reinterpret_cast<void **>(array_d_A.data()), IA, /* base-1 */
            //                                                         reinterpret_cast<void **>(shmA), IA, /* base-1 */
            //                                                         JA,                                  /* base-1 */
            //                                                         descrA, compute_type, &lwork_potrf));
            //     printf("lworkf:%d", lwork_potrf);
            // CUSOLVER_CHECK_OR_RETURN(cusolverMgPotrs_bufferSize(cusolverH, CUBLAS_FILL_MODE_LOWER, N, NRHS, /* NRHS */
            //                                                                                                 //   reinterpret_cast<void **>(array_d_A.data()), IA, JA,
            //                                                     reinterpret_cast<void **>(shmA), IA, JA,
            //                                                     //   descrA, reinterpret_cast<void **>(array_d_B.data()),
            //                                                     descrA, reinterpret_cast<void **>(shmB),
            //                                                     IB, JB, descrB, compute_type,
            //                                                     &lwork_potrs));
            // for (int dev = 0; dev < nbGpus; dev++)
            // {
            //     shmlwork[dev] = std::max(lwork_potrf, lwork_potrs);
            // }
            // }
            // sync_point.arrive_and_wait();
            // CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());

            // /* array_d_work[j] points to device workspace of device j */
            // FFI_ASSIGN_OR_RETURN(auto workspace, AllocateWorkspaceBytes<data_type>(scratch, sizeof(data_type) * (lwork_potrf), "workspace_potrf"));
            // shmwork[currentDevice] = workspace;

            // /* sync all devices */
            // if (currentDevice == 0){
            // JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpy(out_data, shmA[0], b.size_bytes(), gpuMemcpyDeviceToDevice));
            // }
            // CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());

            // sync_point.arrive_and_wait();

            // if (currentDevice == 0)
            // {
            //     cusolver_status = cusolverMgPotrf(
            //         cusolverH, CUBLAS_FILL_MODE_LOWER, N,
            //         reinterpret_cast<void **>(shmA), IA, JA,
            //         descrA, compute_type,
            //         reinterpret_cast<void **>(shmwork), shmlwork[currentDevice], &info);
            //     /* sync all devices */
            //     CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
            //     }
            //     // Copy status to all devices
            //     for (int dev = 0; dev < nbGpus; dev++)
            //     {
            //         cusolver_status_host[dev] = static_cast<int32_t>(cusolver_status);
            //     }
            //     /* check if A is singular */
            //     if (0 > info)
            //     {
            //         return ffi::Error::Internal(
            //             absl::StrFormat("unexpected error in cusolverMgPotrf, %d-th input parameter is wrong \n", -info));
            //     }
            //     // Check status, if 0, continue with Potrs
            //     if (cusolver_status_host[0] == 0)
            //     {
            //         cusolver_status = cusolverMgPotrs(cusolverH, CUBLAS_FILL_MODE_LOWER, N, NRHS, /* NRHS */
            //                                           reinterpret_cast<void **>(shmA), IA, JA, descrA,
            //                                           reinterpret_cast<void **>(shmB), IB, JB, descrB,
            //                                           compute_type,
            //                                           reinterpret_cast<void **>(shmwork), *shmlwork,
            //                                           &info);

            //         /* sync all devices */
            //         CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());

            //         for (int dev = 0; dev < nbGpus; dev++)
            //         {
            //             cusolver_status_host[dev] = static_cast<int32_t>(cusolver_status);
            //         }
            //         /* check if parameters are valid */
            //         if (0 > info)
            //         {
            //             return ffi::Error::Internal(
            //                 absl::StrFormat("unexpected error in cusolverMgPotrs, %d-th input parameter is wrong \n", -info));
            //         }
            //     }
            //     /* check if A is singular */
            // }
            // /* sync all devices */
            // CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
            // sync_point.arrive_and_wait();

            // // Write status data
            // int32_t status_val = static_cast<int32_t>(cusolver_status_host[currentDevice]);
            // JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpy(status_data, &status_val, sizeof(status_val), gpuMemcpyHostToDevice));
            // // Write solution to all shmBs
            // if (currentDevice == 0)
            // {
            //     for (int dev = 1; dev < nbGpus; dev++)
            //     {
            //         JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpy(shmB[dev], shmB[0], b.size_bytes(), gpuMemcpyDeviceToDevice));
            //     }
            // }
            // CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
            // sync_point.arrive_and_wait();
            // // Collect solutions, fill nans if solver failed
            // if (cusolver_status_host[currentDevice] == 0)
            // {
            //     JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpy(out_data, shmB[currentDevice], b.size_bytes(), gpuMemcpyDeviceToDevice));
            // }
            // else
            // {
            //     std::vector<typename traits<data_type>::T> host_nan(N * NRHS, traits<data_type>::nan());
            //     JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpy(out_data, host_nan.data(), sizeof(data_type) * N * NRHS, gpuMemcpyHostToDevice));
            // }
            // CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
            // sync_point.arrive_and_wait();
            // if (currentDevice == 0)
            // {
            //     CUSOLVER_CHECK_OR_RETURN(cusolverMgDestroyMatrixDesc(descrA));
            //     CUSOLVER_CHECK_OR_RETURN(cusolverMgDestroyMatrixDesc(descrB));

            //     CUSOLVER_CHECK_OR_RETURN(cusolverMgDestroyGrid(gridA));
            //     CUSOLVER_CHECK_OR_RETURN(cusolverMgDestroyGrid(gridB));

            //     CUSOLVER_CHECK_OR_RETURN(cusolverMgDestroy(cusolverH));

            sharedMemoryClose(&shminfoA);
            // sharedMemoryClose(&structinfo);
            sharedMemoryClose(&shminfoB);
            sharedMemoryClose(&shminfowork);
            sharedMemoryClose(&shmcsh);
            sharedMemoryClose(&shminfolwork);
            // }
            CUDA_CHECK_OR_RETURN(cudaDeviceSynchronize());
            sync_point.arrive_and_wait();
            // barrierWait(&shm->barrier, &shm->sense, (unsigned int)nbGpus);
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
                    "The input and output to potrs must have the same element type");
            }
            if (dataType != b.element_type())
            {
                return ffi::Error::InvalidArgument(
                    "The input matrix a and output x of potrs must have the same element type");
            }
            FFI_RETURN_IF_ERROR(CheckShape(status->dimensions(), 1, "status", "potrf"));

            SOLVER_DISPATCH_IMPL(PotrsMgImpl, N, NRHS, batch_a, stream, scratch, a, b, tile_size, out, status);

            return ffi::Error::InvalidArgument(absl::StrFormat(
                "Unsupported data type%s for potrs", absl::FormatStreamed(dataType)));
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