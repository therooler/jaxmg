
#include <cstdlib>
#include <string>
#include "shm.h"


int sharedMemoryCreate(const char *name, size_t sz, sharedMemoryInfo *info) {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  info->size = sz;
  info->shmHandle = CreateFileMapping(INVALID_HANDLE_VALUE, NULL,
                                      PAGE_READWRITE, 0, (DWORD)sz, name);
  if (info->shmHandle == 0) {
    return GetLastError();
  }

  info->addr = MapViewOfFile(info->shmHandle, FILE_MAP_ALL_ACCESS, 0, 0, sz);
  if (info->addr == NULL) {
    return GetLastError();
  }

  return 0;
#else
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
#endif
}

int sharedMemoryOpen(const char *name, size_t sz, sharedMemoryInfo *info) {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  info->size = sz;

  info->shmHandle = OpenFileMapping(FILE_MAP_ALL_ACCESS, FALSE, name);
  if (info->shmHandle == 0) {
    return GetLastError();
  }

  info->addr = MapViewOfFile(info->shmHandle, FILE_MAP_ALL_ACCESS, 0, 0, sz);
  if (info->addr == NULL) {
    return GetLastError();
  }

  return 0;
#else
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
#endif
}

void sharedMemoryClose(sharedMemoryInfo *info) {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  if (info->addr) {
    UnmapViewOfFile(info->addr);
  }
  if (info->shmHandle) {
    CloseHandle(info->shmHandle);
  }
#else
  if (info->addr) {
    munmap(info->addr, info->size);
  }
  if (info->shmFd) {
    close(info->shmFd);
  }
#endif
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
 
double **get_shm_device_ptrs(int currentDevice, DynamicBarrier &sync_point, sharedMemoryInfo &info, const char *shmName)
{
    // static const char shmName[] = "shmA";
    double **shm = nullptr;
    pid_t pid = getppid();
    char pidString[20] = {0};
    char lshmName[40] = {0};
    const int MAX_NUM_DEVICES = 16;
    snprintf(pidString, sizeof(pidString), "%d", pid);
    strcat(lshmName, shmName);
    strcat(lshmName, pidString);

    size_t shmSize = MAX_NUM_DEVICES * sizeof(double *);

    if (currentDevice == 0)
    {
        if (sharedMemoryCreate(lshmName, shmSize, &info) != 0)
        {
            printf("Failed to create shared memory\n");
            exit(EXIT_FAILURE); // You can later replace this with proper JAX error handling
        }
        shm = (double **)info.addr;
        memset((void *)shm, 0, shmSize);
        printf("%d: Shared memory initialized\n", currentDevice);
    }

    sync_point.arrive_and_wait(); // Barrier sync

    if (currentDevice != 0)
    {
        if (sharedMemoryOpen(lshmName, shmSize, &info) != 0)
        {
            printf("Failed to open shared memory\n");
            exit(EXIT_FAILURE);
        }
        shm = (double **)info.addr;
        printf("%d: Shared memory opened\n", currentDevice);
    }

    sync_point.arrive_and_wait();
    
    return shm;
}

int64_t get_shm_lwork_ptr(int currentDevice, DynamicBarrier &sync_point, sharedMemoryInfo &info, const char *shmName)
{
    // static const char shmName[] = "shmA";
    int64_t shm = 0;
    pid_t pid = getppid();
    char pidString[20] = {0};
    char lshmName[40] = {0};
    const int MAX_NUM_DEVICES = 16;
    snprintf(pidString, sizeof(pidString), "%d", pid);
    strcat(lshmName, shmName);
    strcat(lshmName, pidString);

    size_t shmSize = sizeof(int64_t);

    if (currentDevice == 0)
    {
        if (sharedMemoryCreate(lshmName, shmSize, &info) != 0)
        {
            printf("Failed to create shared memory\n");
            exit(EXIT_FAILURE); // You can later replace this with proper JAX error handling
        }
        shm = (int64_t )info.addr;
        memset((void *)shm, 0, shmSize);
        printf("%d: Shared memory initialized\n", currentDevice);
    }

    sync_point.arrive_and_wait(); // Barrier sync

    if (currentDevice != 0)
    {
        if (sharedMemoryOpen(lshmName, shmSize, &info) != 0)
        {
            printf("Failed to open shared memory\n");
            exit(EXIT_FAILURE);
        }
        shm = (int64_t )info.addr;
        printf("%d: Shared memory opened\n", currentDevice);
    }

    sync_point.arrive_and_wait();
    
    return shm;
}

