
#include "shm.h"        

#include <cerrno>       
#include <cstdio>       
#include <cstring>      
#include <cstdlib>      

#include <fcntl.h>      
#include <sys/mman.h>  
#include <sys/types.h>  
#include <unistd.h> 
#include <barrier>    

#include "jaxlib/gpu/vendor.h"
#include <third_party/gpus/cuda/include/cuComplex.h>


int sharedMemoryCreate(const char *name, size_t sz, sharedMemoryInfo *info) {

  int status = 0;

  info->size = sz;

  info->shmFd = shm_open(name, O_RDWR | O_CREAT, 0777);
  if (info->shmFd < 0) {
    return errno;
  }

  status = ftruncate(info->shmFd, sz);
  if (status != 0) {
    return status;
  }

  info->addr = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_SHARED, info->shmFd, 0);
  if (info->addr == NULL) {
    return errno;
  }

  return 0;

}

int sharedMemoryOpen(const char *name, size_t sz, sharedMemoryInfo *info) {

  info->size = sz;

  info->shmFd = shm_open(name, O_RDWR, 0777);
  if (info->shmFd < 0) {
    return errno;
  }

  info->addr = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_SHARED, info->shmFd, 0);
  if (info->addr == NULL) {
    return errno;
  }

  return 0;
}

void sharedMemoryClose(sharedMemoryInfo *info) {

  if (info->addr) {
    munmap(info->addr, info->size);
  }
  if (info->shmFd) {
    close(info->shmFd);
  }
}

// Definitions for DynamicBarrier methods
void DynamicBarrier::initialize(int thread_count)
{
    barrier_ptr = std::make_unique<std::barrier<>>(thread_count);
}

void DynamicBarrier::arrive_and_wait()
{
    if (barrier_ptr)
    {
        barrier_ptr->arrive_and_wait();
    }
}

template <typename T>
T **get_shm_device_ptrs(int currentDevice, DynamicBarrier &sync_point, sharedMemoryInfo &info, const char *shmName)
{
    // static const char shmName[] = "shmA";
    T **shm = nullptr;
    pid_t pid = getppid();
    char pidString[20] = {0};
    char lshmName[40] = {0};
    const int MAX_NUM_DEVICES = 16;
    snprintf(pidString, sizeof(pidString), "%d", pid);
    strcat(lshmName, shmName);
    strcat(lshmName, pidString);

    size_t shmSize = MAX_NUM_DEVICES * sizeof(T *);

    if (currentDevice == 0)
    {
        if (sharedMemoryCreate(lshmName, shmSize, &info) != 0)
        {
            printf("Failed to create shared memory\n");
            exit(EXIT_FAILURE); // You can later replace this with proper JAX error handling
        }
        shm = (T **)info.addr;
        memset((void *)shm, 0, shmSize);
        // printf("%d: Shared memory initialized\n", currentDevice);
    }

    sync_point.arrive_and_wait(); // Barrier sync

    if (currentDevice != 0)
    {
        if (sharedMemoryOpen(lshmName, shmSize, &info) != 0)
        {
            printf("Failed to open shared memory\n");
            exit(EXIT_FAILURE);
        }
        shm = (T **)info.addr;
        // printf("%d: Shared memory opened\n", currentDevice);
    }

    sync_point.arrive_and_wait();
    
    return shm;
}

template float  **get_shm_device_ptrs<float >(int, DynamicBarrier&, sharedMemoryInfo&, const char*);
template double **get_shm_device_ptrs<double>(int, DynamicBarrier&, sharedMemoryInfo&, const char*);
template cuFloatComplex **get_shm_device_ptrs<cuFloatComplex>(int, DynamicBarrier&, sharedMemoryInfo&, const char*);
template cuDoubleComplex **get_shm_device_ptrs<cuDoubleComplex>(int, DynamicBarrier&, sharedMemoryInfo&, const char*);

template <typename T>
T* get_shm_lwork_ptr(int currentDevice, DynamicBarrier &sync_point, sharedMemoryInfo &info, const char *shmName)
{
    // static const char shmName[] = "shmA";
    pid_t pid = getppid();
    char pidString[20] = {0};
    char lshmName[40] = {0};
    const int MAX_NUM_DEVICES = 16;
    snprintf(pidString, sizeof(pidString), "%d", pid);
    strcat(lshmName, shmName);
    strcat(lshmName, pidString);

    size_t shmSize = MAX_NUM_DEVICES*sizeof(T);
    T* shm = nullptr;
    if (currentDevice == 0)
    {
        if (sharedMemoryCreate(lshmName, shmSize, &info) != 0)
        {
            printf("Failed to create shared memory\n");
            exit(EXIT_FAILURE); // You can later replace this with proper JAX error handling
        }
        shm = reinterpret_cast<T*>(info.addr);;
        memset(shm, 0, shmSize);
        // printf("%d: Shared memory initialized\n", currentDevice);
    }

    sync_point.arrive_and_wait(); // Barrier sync

    if (currentDevice != 0)
    {
        if (sharedMemoryOpen(lshmName, shmSize, &info) != 0)
        {
            printf("Failed to open shared memory\n");
            exit(EXIT_FAILURE);
        }
        shm = reinterpret_cast<T*>(info.addr);
    }

    sync_point.arrive_and_wait();
    
    return shm;
}

template int64_t* get_shm_lwork_ptr<int64_t>(int, DynamicBarrier&, sharedMemoryInfo&, const char*);
template int32_t* get_shm_lwork_ptr<int32_t>(int, DynamicBarrier&, sharedMemoryInfo&, const char*);


