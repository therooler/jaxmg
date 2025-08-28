
#ifndef HELPER_MULTIPROCESS_H
#define HELPER_MULTIPROCESS_H

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <iostream>
#include <stdio.h>
#include <tchar.h>
#include <strsafe.h>
#include <sddl.h>
#include <aclapi.h>
#include <winternl.h>
#else
#include <stdio.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <errno.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <memory.h>
#include <sys/un.h>
#endif
#include <vector>
#include <barrier>

#define MAX_DEVICES 16

typedef struct sharedMemoryInfo_st
{
    void *addr;
    size_t size;
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    HANDLE shmHandle;
#else
    int shmFd;
#endif
} sharedMemoryInfo;

int sharedMemoryCreate(const char *name, size_t sz, sharedMemoryInfo *info);

int sharedMemoryOpen(const char *name, size_t sz, sharedMemoryInfo *info);

void sharedMemoryClose(sharedMemoryInfo *info);

typedef struct shmStruct_st
{
    int64_t lwork = 0;              // Work buff
    int devices[MAX_DEVICES];       // Participating device IDs
    double *device_ptrs[MAX_DEVICES]; // IPC memory handles
} shmStruct;

class DynamicBarrier
{
private:
    std::unique_ptr<std::barrier<>> barrier_ptr;

public:
    void initialize(int thread_count);
    void arrive_and_wait();
};

// volatile shmStruct *get_shm_device_ptrs(int currentDevice, int nbGpus, DynamicBarrier &sync_point, sharedMemoryInfo &info, const char *shmName);

double **get_shm_device_ptrs(int currentDevice, DynamicBarrier &sync_point, sharedMemoryInfo &info, const char *shmName);
int64_t get_shm_lwork_ptr(int currentDevice, DynamicBarrier &sync_point, sharedMemoryInfo &info, const char *shmName);

#endif // HELPER_MULTIPROCESS_H
