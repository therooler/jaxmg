#ifndef JAXMG_THREAD_BARRIER_H
#define JAXMG_THREAD_BARRIER_H

#include <barrier>
#include <memory>
#include <cstddef>

class ThreadBarrier
{
private:
    std::unique_ptr<std::barrier<>> barrier_ptr;

public:
    void initialize(int thread_count)
    {
        barrier_ptr = std::make_unique<std::barrier<>>(static_cast<std::ptrdiff_t>(thread_count));
    }

    void arrive_and_wait()
    {
        if (barrier_ptr)
        {
            barrier_ptr->arrive_and_wait();
        }
    }
};

#endif // JAXMG_THREAD_BARRIER_H
