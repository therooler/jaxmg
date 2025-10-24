#ifndef JAXMG_PROCESS_BARRIER_H
#define JAXMG_PROCESS_BARRIER_H

#include <semaphore.h>
#include <fcntl.h>
#include <string>
#include <sys/mman.h> // for mmap, munmap, shm_open, shm_unlink
#include <sys/stat.h> // for file mode constants (used in shm_open)
#include <unistd.h>   // for ftruncate, getpid
#include <cstdlib>    // for nullptr (if using older C++ standard)

class DynamicBarrier
{
    sem_t *barrier_sem = SEM_FAILED;
    sem_t *counter_sem = SEM_FAILED;
    int *counter = nullptr;
    int world_size;
    std::string barrier_name, shm_name, counter_sem_name;
    bool i_created = false; // I created the objects?

public:
    DynamicBarrier(int world_size, int rank)
        : world_size(world_size)
    {
        barrier_name = "/jaxmg_barrier";
        counter_sem_name = "/jaxmg_csem";
        shm_name = "/jaxmg_counter";

        if (rank == 0)
        {
            // Try to create; if already exists, just open.
            errno = 0;
            barrier_sem = sem_open(barrier_name.c_str(), O_CREAT | O_EXCL, 0644, 0);
            if (barrier_sem == SEM_FAILED && errno == EEXIST)
                barrier_sem = sem_open(barrier_name.c_str(), 0);
            else
                i_created = true;

            errno = 0;
            counter_sem = sem_open(counter_sem_name.c_str(), O_CREAT | O_EXCL, 0644, 1);
            if (counter_sem == SEM_FAILED && errno == EEXIST)
                counter_sem = sem_open(counter_sem_name.c_str(), 0);
            else
                i_created = true;

            int shm_fd = shm_open(shm_name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0644);
            if (shm_fd == -1 && errno == EEXIST)
                shm_fd = shm_open(shm_name.c_str(), O_RDWR, 0644);
            else
            {
                i_created = true;
                ftruncate(shm_fd, sizeof(int));
            }
            counter = (int *)mmap(nullptr, sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
            close(shm_fd);
            if (i_created)
                *counter = 0;
        }
        else
        {
            // Wait until rank 0 has created things.
            while ((barrier_sem = sem_open(barrier_name.c_str(), 0)) == SEM_FAILED && errno == ENOENT)
                usleep(1000);
            while ((counter_sem = sem_open(counter_sem_name.c_str(), 0)) == SEM_FAILED && errno == ENOENT)
                usleep(1000);

            int shm_fd = -1;
            while ((shm_fd = shm_open(shm_name.c_str(), O_RDWR, 0644)) == -1 && errno == ENOENT)
                usleep(1000);
            counter = (int *)mmap(nullptr, sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
            close(shm_fd);
        }
    }

    void arrive_and_wait()
    {
        sem_wait(counter_sem);
        (*counter)++;
        if (*counter == world_size)
        {
            // last to arrive
            for (int i = 0; i < world_size - 1; ++i)
            {
                sem_post(barrier_sem);
            }
            *counter = 0;          // reset
            sem_post(counter_sem); // hand over for next round
        }
        else
        {
            sem_post(counter_sem);
            sem_wait(barrier_sem);
        }
    }

    ~DynamicBarrier()
    {
        if (barrier_sem != SEM_FAILED)
            sem_close(barrier_sem);
        if (counter_sem != SEM_FAILED)
            sem_close(counter_sem);
        if (counter)
            munmap(counter, sizeof(int));
        if (i_created)
        { // only the creator unlinks
            sem_unlink(barrier_name.c_str());
            sem_unlink(counter_sem_name.c_str());
            shm_unlink(shm_name.c_str());
        }
    }
};

#if defined(__linux__)
#define cpu_atomic_add32(a, x) __sync_add_and_fetch(a, x)
#elif defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define cpu_atomic_add32(a, x) InterlockedAdd((volatile LONG *)a, x)
#else
#error Unsupported system
#endif


static void barrierWait(volatile int *barrier, volatile int *sense, unsigned int n)
{
    int count;

    // Check-in
    count = cpu_atomic_add32(barrier, 1);
    if (count == n) // Last one in
        *sense = 1;
    while (!*sense)
        ;

    // Check-out
    count = cpu_atomic_add32(barrier, -1);
    if (count == 0) // Last one out
        *sense = 0;
    while (*sense)
        ;
}

#endif // JAXMG_PROCESS_BARRIER_H