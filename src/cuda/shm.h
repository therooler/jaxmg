
#ifndef SHM_MULTIPROCESS_H
#define SHM_MULTIPROCESS_H

// C++
#include <vector>
#include <barrier>
#include <memory>
#include <cstddef>
#include <cstdint>

#define MAX_DEVICES 16

typedef struct sharedMemoryInfo_st
{
    void *addr;
    size_t size;
    int shmFd;
} sharedMemoryInfo;

int sharedMemoryCreate(const char *name, size_t sz, sharedMemoryInfo *info);

int sharedMemoryOpen(const char *name, size_t sz, sharedMemoryInfo *info);

void sharedMemoryClose(sharedMemoryInfo *info);

typedef struct shmStruct_st
{
    int64_t lwork = 0;                // Work buff
    int devices[MAX_DEVICES];         // Participating device IDs
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

template <typename T>
T **get_shm_device_ptrs(int currentDevice, DynamicBarrier &sync_point, sharedMemoryInfo &info, const char *shmName);

template <typename T>
T *get_shm_lwork_ptr(int currentDevice, DynamicBarrier &sync_point, sharedMemoryInfo &info, const char *shmName);

#endif // SHM_MULTIPROCESS_H
