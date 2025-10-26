#ifndef JAXMG_PROCESS_BARRIER_H
#define JAXMG_PROCESS_BARRIER_H


#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>
#include <cstdio>
#include <pthread.h>

class DynamicBarrier {
    struct Shared {
        pthread_mutex_t mtx;
        pthread_cond_t  cv;
        int n;          // total participants
        int count;      // remaining to arrive in this generation
        int gen;        // generation number
        int inited;     // 0 until fully initialized (published last)
    };

    std::string shm_name;
    Shared* s = nullptr;
    bool i_created = false;
    int world_size = 0;

    static void throw_errno(const char* what) {
        char buf[256];
        snprintf(buf, sizeof(buf), "%s: %s", what, strerror(errno));
        throw std::runtime_error(buf);
    }

public:
    // Any process can call this with the same shm 'name' and the same 'world_size'.
    // The first creator sets world_size; others verify it.
    DynamicBarrier(int world_size, const std::string& name="/mpbarrier")
        : shm_name(name), world_size(world_size)
    {
        int fd = shm_open(shm_name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0644);
        if (fd >= 0) {
            i_created = true;
            if (ftruncate(fd, sizeof(Shared)) == -1) { close(fd); throw_errno("ftruncate"); }
        } else {
            if (errno != EEXIST) throw_errno("shm_open(O_CREAT|O_EXCL)");
            fd = shm_open(shm_name.c_str(), O_RDWR, 0644);
            if (fd == -1) throw_errno("shm_open");
        }

        void* addr = mmap(nullptr, sizeof(Shared), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        if (addr == MAP_FAILED) throw_errno("mmap");
        s = static_cast<Shared*>(addr);

        if (i_created) {
            // Set up process-shared primitives
            pthread_mutexattr_t ma; pthread_condattr_t ca;
            if (pthread_mutexattr_init(&ma)) throw std::runtime_error("pthread_mutexattr_init");
            if (pthread_mutexattr_setpshared(&ma, PTHREAD_PROCESS_SHARED))
                throw std::runtime_error("pthread_mutexattr_setpshared");
            if (pthread_condattr_init(&ca)) throw std::runtime_error("pthread_condattr_init");
            if (pthread_condattr_setpshared(&ca, PTHREAD_PROCESS_SHARED))
                throw std::runtime_error("pthread_condattr_setpshared");

            if (pthread_mutex_init(&s->mtx, &ma)) throw std::runtime_error("pthread_mutex_init");
            if (pthread_cond_init(&s->cv, &ca))   throw std::runtime_error("pthread_cond_init");
            pthread_mutexattr_destroy(&ma);
            pthread_condattr_destroy(&ca);

            s->n      = world_size;
            s->count  = world_size;
            s->gen    = 0;

            // Publish last: mark initialized
            __sync_synchronize();
            s->inited = 1;
        } else {
            // Wait until the creator finishes initializing
            while (__atomic_load_n(&s->inited, __ATOMIC_ACQUIRE) == 0) {
                // simple spin; in practice this is a very short window
                sched_yield();
            }
            if (s->n != world_size)
                throw std::runtime_error("Barrier world_size mismatch across processes.");
        }
    }

    // Reusable barrier
    void arrive_and_wait() {
        pthread_mutex_lock(&s->mtx);
        int my_gen = s->gen;
        if (--s->count == 0) {
            // Last to arrive: advance generation, reset, wake everyone
            s->gen++;
            s->count = s->n;
            pthread_cond_broadcast(&s->cv);
            pthread_mutex_unlock(&s->mtx);
        } else {
            // Sleep until next generation begins
            while (my_gen == s->gen)
                pthread_cond_wait(&s->cv, &s->mtx);
            pthread_mutex_unlock(&s->mtx);
        }
    }

    ~DynamicBarrier() {
        if (s) munmap(s, sizeof(Shared));
    }
};

#endif // JAXMG_PROCESS_BARRIER_H