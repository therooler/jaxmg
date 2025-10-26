#ifndef CUDA_IPC_UTILS_H_
#define CUDA_IPC_UTILS_H_

// C++
#include <cstdint>
#include <stdexcept>
#include <string>
// CUDA
#include "third_party/gpus/cuda/include/cuda_runtime.h"

template <typename T>
void ipcGetHandleAndOffset(T *array_data_A, cudaIpcMemHandle_t &handle, size_t &offset);

template <typename T>
std::vector<T *> ipcGetDevicePointers(int currentDevice, int nbGpus, cudaIpcMemHandle_t *shmAipc, size_t *shmoffsetA);

void print_current_context();

void print_pointer_info(void *ptr);

#endif