// Own code
#include "shm.h"
// C++
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>
#include <barrier>
// JAXlib
#include "jaxlib/gpu/vendor.h"
// CUDA
#include <third_party/gpus/cuda/include/cuComplex.h>
#include "third_party/gpus/cuda/include/cuda_runtime.h"
// Own code
#include "process_barrier.h"
#include "thread_barrier.h"

int sharedMemoryCreate(const char *name, size_t sz, sharedMemoryInfo *info)
{

  int status = 0;

  info->size = sz;

  info->shmFd = shm_open(name, O_RDWR | O_CREAT, 0777);
  if (info->shmFd < 0)
  {
    return errno;
  }

  status = ftruncate(info->shmFd, sz);
  if (status != 0)
  {
    return status;
  }

  info->addr = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_SHARED, info->shmFd, 0);
  if (info->addr == NULL)
  {
    return errno;
  }

  return 0;
}

int sharedMemoryOpen(const char *name, size_t sz, sharedMemoryInfo *info)
{

  info->size = sz;

  info->shmFd = shm_open(name, O_RDWR, 0777);
  if (info->shmFd < 0)
  {
    return errno;
  }

  info->addr = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_SHARED, info->shmFd, 0);
  if (info->addr == NULL)
  {
    return errno;
  }

  return 0;
}

void sharedMemoryClose(sharedMemoryInfo *info)
{

  if (info->addr)
  {
    munmap(info->addr, info->size);
  }
  if (info->shmFd)
  {
    close(info->shmFd);
  }
}

template <typename T>
T **get_shm_device_ptrs(int currentDevice, ThreadBarrier &sync_point, sharedMemoryInfo &info, const char *shmName)
{
  T **shm = nullptr;
  pid_t pid = getppid();
  char lshmName[40] = {0};
  const int MAX_NUM_DEVICES = 16;
  strcat(lshmName, "jaxmg_");
  strcat(lshmName, shmName);

  size_t shmSize = MAX_NUM_DEVICES * sizeof(T *);

  if (currentDevice == 0)
  {
    if (sharedMemoryCreate(lshmName, shmSize, &info) != 0)
    {
      printf("Failed to create shared memory\n");
      exit(EXIT_FAILURE); // #TODO: replace this with proper JAX error handling
    }
    // printf("Created shm in device %d\n", currentDevice);
    shm = (T **)info.addr;
    memset((void *)shm, 0, shmSize);
  }

  sync_point.arrive_and_wait(); // Barrier sync

  if (currentDevice != 0)
  {
    if (sharedMemoryOpen(lshmName, shmSize, &info) != 0)
    {
      printf("Failed to open shared memory in dev %d\n", currentDevice);
      exit(EXIT_FAILURE);
    }
    shm = (T **)info.addr;
  }

  sync_point.arrive_and_wait();

  return shm;
}

template float **get_shm_device_ptrs<float>(int, ThreadBarrier &, sharedMemoryInfo &, const char *);
template double **get_shm_device_ptrs<double>(int, ThreadBarrier &, sharedMemoryInfo &, const char *);
template cuFloatComplex **get_shm_device_ptrs<cuFloatComplex>(int, ThreadBarrier &, sharedMemoryInfo &, const char *);
template cuDoubleComplex **get_shm_device_ptrs<cuDoubleComplex>(int, ThreadBarrier &, sharedMemoryInfo &, const char *);

template <typename T, typename barrier>
T *get_shm_lwork_ptr(int currentDevice, barrier &sync_point, sharedMemoryInfo &info, const char *shmName)
{
  pid_t pid = getppid();
  char lshmName[40] = {0};
  const int MAX_NUM_DEVICES = 16;
  strcat(lshmName, "jaxmg_");
  strcat(lshmName, shmName);

  size_t shmSize = MAX_NUM_DEVICES * sizeof(T);
  T *shm = nullptr;
  if (currentDevice == 0)
  {
    if (sharedMemoryCreate(lshmName, shmSize, &info) != 0)
    {
      printf("Failed to create shared memory\n");
      exit(EXIT_FAILURE); // #TODO: Replace this with proper JAX error handling
    }
    shm = reinterpret_cast<T *>(info.addr);
    ;
    memset(shm, 0, shmSize);
  }

  sync_point.arrive_and_wait(); // Barrier sync

  if (currentDevice != 0)
  {
    if (sharedMemoryOpen(lshmName, shmSize, &info) != 0)
    {
      printf("Failed to open shared memory\n");
      exit(EXIT_FAILURE);
    }
    shm = reinterpret_cast<T *>(info.addr);
  }

  sync_point.arrive_and_wait();

  return shm;
}

template int64_t *get_shm_lwork_ptr<int64_t, DynamicBarrier>(int, DynamicBarrier &, sharedMemoryInfo &, const char *);
template int32_t *get_shm_lwork_ptr<int32_t, DynamicBarrier>(int, DynamicBarrier &, sharedMemoryInfo &, const char *);
template size_t *get_shm_lwork_ptr<size_t, DynamicBarrier>(int, DynamicBarrier &, sharedMemoryInfo &, const char *);

template int64_t *get_shm_lwork_ptr<int64_t, ThreadBarrier>(int, ThreadBarrier &, sharedMemoryInfo &, const char *);
template int32_t *get_shm_lwork_ptr<int32_t, ThreadBarrier>(int, ThreadBarrier &, sharedMemoryInfo &, const char *);
template size_t *get_shm_lwork_ptr<size_t, ThreadBarrier>(int, ThreadBarrier &, sharedMemoryInfo &, const char *);

cudaIpcMemHandle_t *get_shm_ipc_handles(int currentDevice, DynamicBarrier &sync_point, sharedMemoryInfo &info, const char *shmName)
{
  const int MAX_NUM_DEVICES = 16;

  // Use a stable, shared name (not PID)
  std::string name = std::string("/") + shmName;            // e.g., "/shmA" – include your session suffix if you have one
  size_t sz = sizeof(cudaIpcMemHandle_t) * MAX_NUM_DEVICES; // nbGpus must be visible here

  if (currentDevice == 0)
  {
    int rc = sharedMemoryCreate(name.c_str(), sz, &info);
    if (rc)
    {

      printf("Failed to open shared memory IPC\n");
      std::abort();
    }
    memset(info.addr, 0, sz);
  }

  sync_point.arrive_and_wait();

  if (currentDevice != 0)
  {
    int rc = sharedMemoryOpen(name.c_str(), sz, &info);
    if (rc)
    {
      printf("Failed to open shared memory IPC\n");
      std::abort();
    }
  }

  sync_point.arrive_and_wait();
  return static_cast<cudaIpcMemHandle_t *>(info.addr);
}

