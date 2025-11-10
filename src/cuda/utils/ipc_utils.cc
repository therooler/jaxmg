// C++
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdio>
#include <algorithm>
// CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuComplex.h"
#include "third_party/gpus/cuda/include/cuda_runtime.h"
#include "third_party/gpus/cuda/include/cuda.h"
// Own code
#include "utils/ipc_utils.h"

template <typename T>
void ipcGetHandleAndOffset(T *array_data_A, cudaIpcMemHandle_t &handle, uintptr_t &offset)
{

    CUdeviceptr base = 0;
    uintptr_t size = 0;
    CUresult r = cuMemGetAddressRange(&base, &size,
                                      reinterpret_cast<CUdeviceptr>(array_data_A));
    if (r != CUDA_SUCCESS)
    {
        const char *name = nullptr;
        const char *desc = nullptr;
        cuGetErrorName(r, &name);
        cuGetErrorString(r, &desc);
        throw std::runtime_error(std::string("cuMemGetAddressRange failed: ") +
                                 (name ? name : "UNKNOWN") + " - " +
                                 (desc ? desc : ""));
    }

    // Compute offset = pointer - base (in bytes)
    const std::uintptr_t p = reinterpret_cast<std::uintptr_t>(array_data_A);
    const std::uintptr_t b = static_cast<std::uintptr_t>(base);
    cudaIpcGetMemHandle(&handle, reinterpret_cast<void *>(base));
    offset = static_cast<uintptr_t>(p - b);
}

template void ipcGetHandleAndOffset<float>(float *array_data_A, cudaIpcMemHandle_t &handle, uintptr_t &offset);
template void ipcGetHandleAndOffset<double>(double *array_data_A, cudaIpcMemHandle_t &handle, uintptr_t &offset);
template void ipcGetHandleAndOffset<cuFloatComplex>(cuFloatComplex *array_data_A, cudaIpcMemHandle_t &handle, uintptr_t &offset);
template void ipcGetHandleAndOffset<cuDoubleComplex>(cuDoubleComplex *array_data_A, cudaIpcMemHandle_t &handle, uintptr_t &offset);

template <typename T>
IpcOpenResult<T> ipcGetDevicePointers(
    int currentDevice,           // this process's device ordinal
    int nbGpus,                  // number of GPUs participating
    cudaIpcMemHandle_t *shmAipc, // per-device IPC handles (exported by each process)
    uintptr_t *shmoffsetA        // per-device byte offsets (relative to each base)
)
{
    const int MAX_NUM_DEVICES = 16;
    IpcOpenResult<T> out;
    out.ptrs.assign(MAX_NUM_DEVICES, nullptr);
    out.bases.assign(MAX_NUM_DEVICES, nullptr);
    // Return vector sized to 16.
    cudaError_t status;
    // Only the coordinator (dev 0) opens peers (matches your current pattern).
    for (int dev = 0; dev < nbGpus; ++dev)
    {
        if (dev == currentDevice)
        {
            continue;
        }
        cudaSetDevice(dev);
        void *opened_base = nullptr;
        status = cudaIpcOpenMemHandle(&opened_base, shmAipc[dev], cudaIpcMemLazyEnablePeerAccess);
        const char *err_name = cudaGetErrorName(status);
        const char *err_str = cudaGetErrorString(status);
        if (status != cudaSuccess)
        {
            printf("cudaIpcOpenMemHandle(dev=%d) -> %d (%s): %s\n",
                   dev, static_cast<int>(status),
                   err_name ? err_name : "UNKNOWN",
                   err_str ? err_str : "UNKNOWN");
            throw std::runtime_error(std::string("cudaIpcOpenMemHandle failed"));
        }
        // Apply the peer's byte offset to get the logical subarray pointer.
        out.bases[dev] = opened_base;
        auto *bytes = static_cast<unsigned char *>(opened_base) + static_cast<std::uintptr_t>(shmoffsetA[dev]);
        out.ptrs[dev] = reinterpret_cast<T *>(bytes);
    }
    cudaSetDevice(currentDevice);
    return out;
}

template IpcOpenResult<float> ipcGetDevicePointers<float>(int currentDevice, int nbGpus, cudaIpcMemHandle_t *shmAipc, uintptr_t *shmoffsetA);
template IpcOpenResult<double> ipcGetDevicePointers<double>(int currentDevice, int nbGpus, cudaIpcMemHandle_t *shmAipc, uintptr_t *shmoffsetA);
template IpcOpenResult<cuFloatComplex> ipcGetDevicePointers<cuFloatComplex>(int currentDevice, int nbGpus, cudaIpcMemHandle_t *shmAipc, uintptr_t *shmoffsetA);
template IpcOpenResult<cuDoubleComplex> ipcGetDevicePointers<cuDoubleComplex>(int currentDevice, int nbGpus, cudaIpcMemHandle_t *shmAipc, uintptr_t *shmoffsetA);

void ipcCloseDevicePointers(int currentDevice, const std::vector<void *> &bases, int nbGpus)
{
    for (int dev = 0; dev < nbGpus; ++dev)
    {
        if (dev == currentDevice)
        {
            continue;
        }
        if (bases[dev])
        {
            cudaIpcCloseMemHandle(bases[dev]);
        }
    }
}
void print_current_context()
{
    cuInit(0); // safe if already initialized
    CUcontext ctx = nullptr;
    CUdevice dev;
    cuCtxGetCurrent(&ctx); // <-- the current CUcontext (may be NULL)
    cuCtxGetDevice(&dev);

    char name[128];
    cuDeviceGetName(name, sizeof(name), dev);

    char busId[32];
    cuDeviceGetPCIBusId(busId, sizeof(busId), dev);

    printf("driver: ctx=%p dev=%d name=%s pciBusId=%s\n",
           (void *)ctx, (int)dev, name, busId);

    // If you want to know whether you're using the device's primary context:
    CUcontext primary = nullptr;
    cuDevicePrimaryCtxRetain(&primary, dev);
    printf("primary=%p (current==primary? %s)\n",
           (void *)primary, (primary == ctx ? "yes" : "no"));
    cuDevicePrimaryCtxRelease(dev);
}

void print_pointer_info(void *ptr)
{
    CUmemorytype memType{};
    int is_managed = -1;
    int is_legacy_ipc = -1;
    CUmemoryPool pool = nullptr;
    CUdeviceptr canonical = 0;
    CUdeviceptr range_base = 0;
    size_t range_size = 0;
    int owning_device = -1;
    // Check attributes
    cuPointerGetAttribute(&memType, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (CUdeviceptr)ptr);
    cuPointerGetAttribute(&is_managed, CU_POINTER_ATTRIBUTE_IS_MANAGED, (CUdeviceptr)ptr);
    cuPointerGetAttribute(&is_legacy_ipc, CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE, (CUdeviceptr)ptr);
    cuPointerGetAttribute(&pool, CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE, (CUdeviceptr)ptr);
    // Get underlying device ptr
    cuPointerGetAttribute(&canonical, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, (CUdeviceptr)ptr);
    cuPointerGetAttribute(&range_base, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, (CUdeviceptr)ptr);
    cuPointerGetAttribute(&range_size, CU_POINTER_ATTRIBUTE_RANGE_SIZE, (CUdeviceptr)canonical);
    cuPointerGetAttribute(&owning_device, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, (CUdeviceptr)ptr);
    uintptr_t offset = static_cast<uintptr_t>(
        (uintptr_t)canonical - (uintptr_t)range_base);
    printf("Owning: device=%d\n", owning_device);
    printf("A ptr orig=%p canonical=%p\n", (void *)ptr, (void *)canonical);
    printf("size=%zu offset=%zu\n", range_size, offset);
    printf("memType=%d - (DEVICE=%d) managed=%d legacy_ipc=%d mempool=%p canonical=0x%llx\n",
           (int)memType, (int)CU_MEMORYTYPE_DEVICE, is_managed, is_legacy_ipc, (void *)pool,
           (unsigned long long)canonical);
}