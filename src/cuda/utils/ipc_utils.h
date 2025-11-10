#ifndef CUDA_IPC_UTILS_H_
#define CUDA_IPC_UTILS_H_

// C++
#include <cstdint>
#include <stdexcept>
#include <string>
// CUDA
#include "third_party/gpus/cuda/include/cuda_runtime.h"

template <typename T>
void ipcGetHandleAndOffset(T *array_data_A, cudaIpcMemHandle_t &handle, uintptr_t &offset);

template <typename T>
struct IpcOpenResult
{
    std::vector<T *> ptrs;     // offset-applied logical pointers (what you use)
    std::vector<void *> bases; // base pointers from cudaIpcOpenMemHandle (what you close)
};

template <typename T>
IpcOpenResult<T> ipcGetDevicePointers(int currentDevice, int nbGpus, cudaIpcMemHandle_t *shmAipc, uintptr_t *shmoffsetA);

void ipcCloseDevicePointers(int currentDevice, const std::vector<void *> &bases, int nbGpus);

void print_current_context();

void print_pointer_info(void *ptr);

#endif