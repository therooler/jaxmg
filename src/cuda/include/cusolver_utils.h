/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef CUSOLVER_GPU_UTILS_H_
#define CUSOLVER_GPU_UTILS_H_

#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <cstdio>
#include <cstddef>
#include <vector>
#include <unordered_set>
#include <cstdlib>
#include <unordered_map>
#include <library_types.h>
// Cusolver
#include <third_party/gpus/cuda/include/cuComplex.h>
#include <third_party/gpus/cuda/include/cusolverDn.h>

inline bool g_cusolver_utils_verbose = []()
{
    const char *env = std::getenv("JAXMG_CUSOLVER_UTILS_VERBOSE");
    if (!env)
        return false;
    char c = env[0];
    if (c == '\0')
        return false;
    switch (c)
    {
    case '1':
    case 'y':
    case 'Y':
    case 't':
    case 'T':
        return true;
    case '0':
    case 'n':
    case 'N':
    case 'f':
    case 'F':
        return false;
    default:
        return true;
    }
}();

// Returns an unordered_map mapping global_col_src -> (global_col_dst, visited=false)
std::unordered_map<int, std::pair<int, bool>> get_col_maps(int N, int N_batch, int T_A, int num_devices)
{
    std::unordered_map<int, std::pair<int, bool>> col_map;

    // int num_devices_active = (N + N_batch - 1) / N_batch; // ceil(N / N_batch)
    // if (g_cusolver_utils_verbose)
    //     std::printf("num_devices_active=%d\n", num_devices_active);
    int shard_size = N / num_devices;
    // dst_cols initialized with size num_devices to mirror the python file
    std::vector<int> dst_cols(num_devices, 0);
    int dst_dev = -1;
    int offset = N_batch - (N / num_devices);
    if (g_cusolver_utils_verbose)
        std::printf("N_batch=%d offset=%d\n", N_batch, offset);

    for (int col = 0; col < N; ++col)
    {
        if (col % T_A == 0)
        {
            dst_dev = (dst_dev + 1) % num_devices;
        }
        int num_offsets = col / shard_size;
        int global_col_src = col + offset * num_offsets;
        int global_col_dst = dst_cols[dst_dev] + dst_dev * N_batch;
        col_map[global_col_src] = std::make_pair(global_col_dst, false);
        int src_dev = global_col_src / N_batch;
        if (g_cusolver_utils_verbose)
            std::printf("src_dev %d, dst_dev %d, src=%d, dst=%d, visited=%s\n",
                        src_dev,
                        dst_dev,
                        global_col_src,
                        col_map[global_col_src].first,
                        col_map[global_col_src].second ? "true" : "false");
        dst_cols[dst_dev] += 1;
    }
    return col_map;
}

// get_cycles: given tile_list mapping src_col -> (dst_col, visited),
// returns a map of cycle_starts -> cycle_vector. Mutates visited flags.
std::unordered_map<int, std::vector<int>> get_cycles(
    std::unordered_map<int, std::pair<int, bool>> &col_map)
{

    std::unordered_map<int, std::vector<int>> cycles;

    for (auto it = col_map.begin(); it != col_map.end(); ++it)
    {
        int key = it->first;
        // if already visited, skip
        if (it->second.second)
            continue;

        int dst = it->second.first;
        // trivial self-mapping: mark visited and continue
        if (dst == key)
        {
            it->second.second = true;
            if (g_cusolver_utils_verbose)
                std::printf("%d -> %d (no copy)\n", key, dst);
            continue;
        }

        // start a new cycle
        std::vector<int> cycle;
        cycle.push_back(key);
        it->second.second = true;
        if (g_cusolver_utils_verbose)
            std::printf("starting cycle %d\n", key);

        // Follow the chain.
        // There are three cases possible
        // 1. The chain reaches it's start. We break.
        // 2. The chain reaches a node that's been visited before. We merge the chains and break.
        // 3. We hit padding. We break.
        // Note that this will always result in disjoint cycles. Case 1 is clearly always disjoint.
        // For Case 2 and 3 assume a chain 2 15 75 86 (99) where (indicates padding). A new chain
        // will start: 3 16 9 2 (hit). Since we already visited 2, we merge: 3 16 9 2 15 75 86.
        while (true)
        {
            auto dst_it = col_map.find(dst);
            if (dst_it != col_map.end())
            {
                if (g_cusolver_utils_verbose)
                {
                    std::printf("current chain:");
                    for (int v : cycle)
                        std::printf(" %d", v);
                    std::printf("\n");
                }
                // Case 1: Cycle closed
                if (dst_it->second.first == key)
                {
                    if (g_cusolver_utils_verbose)
                        printf("dst=%d == key=%d\n, breaking", dst, key);
                    cycle.push_back(dst);
                    break;
                }
                // Case 2: We hit a column that we've visited before
                if (col_map[dst].second)
                {
                    if (g_cusolver_utils_verbose)
                        std::printf("dst = %d was already visited, stopping cycle\n", dst);
                    auto cycle_it = cycles.find(dst);

                    if (g_cusolver_utils_verbose)
                        std::printf("Merging cycles %d and %d\n", key, dst);
                    // Merge chains
                    for (int v : cycle_it->second)
                    {
                        cycle.push_back(v);
                    }
                    cycles.erase(dst);

                    break;
                }
                // Add to chain
                if (g_cusolver_utils_verbose)
                    printf("push_back dst=%d\n", dst);
                cycle.push_back(dst);
                // Mark as visited
                col_map[dst].second = true;
                // Get next column
                dst = dst_it->second.first;
            }
            // Case 3: We hit padding.
            else
            {
                if (g_cusolver_utils_verbose)
                    std::printf("dst = %d is padding", dst);
                cycle.push_back(dst);
                if (g_cusolver_utils_verbose)
                    std::printf(", stopping cycle\n");
                break;
            }
        }
        if (g_cusolver_utils_verbose)
        {
            std::printf("cycle result:");
            for (int v : cycle)
                std::printf(" %d", v);
            std::printf("\n");
        }
        if (cycle.size() > 1)
            cycles.emplace(key, std::move(cycle));
    }

    return cycles;
}

template <typename T_ELEM>
static void memcpyCyclicShard(int num_devices, gpuStream_t stream, const int *deviceIdA, /* <int> dimension num_devices */
                              int N,                                                     /* number of rows in local A, B */
                              int N_batch,                                               /* number of columns in local A, B */
                              int T_A,                                                   /* number of columns per column tile */
                              /* input */
                              T_ELEM **array_d_A /* array of device array, array_d_A is N-by-N_batch with leading dimension LLD_A  */
)
{
    if (num_devices < 2)
    {
        return;
    }

    int currentDev = 0; /* record current device id */
    CUDA_CHECK(cudaGetDevice(&currentDev));

    // Lightweight diagnostics: report matrix/tile parameters
    if (g_cusolver_utils_verbose)
        printf("memcpyCyclicShard: N_A=%d, T_A=%d,  N_batch=%d\n", N, T_A, N_batch);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Get tiles
    std::unordered_map<int, std::pair<int, bool>> col_map = get_col_maps(N, N_batch, T_A, num_devices);
    // Get permutation cycles
    std::unordered_map<int, std::vector<int>> cycles = get_cycles(col_map);
    std::printf("Permutation cycles (count=%zu):\n", cycles.size());
    if (g_cusolver_utils_verbose)
    {
        for (const auto &entry : cycles)
        {
            std::printf("cycle start %d:", entry.first);
            for (int v : entry.second)
                std::printf(" %d", v);
            std::printf("\n");
        }
    }
    // Implement Python-style cycle moves using two staging buffers. Leave TODOs
    // for the actual buffer allocation / copy operations so you can fill them in.

    std::vector<T_ELEM *> buffers(2, nullptr);
    const size_t num_bytes_per_col = static_cast<size_t>(N) * sizeof(T_ELEM);
    for (int b = 0; b < 2; ++b)
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&buffers[b]), num_bytes_per_col));
        if (g_cusolver_utils_verbose)
            std::printf("memcpyCyclicShard: allocated staging buffer [%d] \n", b);
    }
    int bit = 0;

    for (const auto &entry : cycles)
    {
        const std::vector<int> &cycle = entry.second;

        if (g_cusolver_utils_verbose)
        {
            std::printf("cycle:");
            for (int v : cycle)
                std::printf(" %d", v);
            std::printf("\n");
        }

        int j = -1;
        for (size_t move = 0; move + 1 < cycle.size(); ++move)
        {
            int i = cycle[move];
            j = cycle[move + 1];

            // buffers[bit] = A[:, i].copy()
            const int src_device_i = i / N_batch;
            if (g_cusolver_utils_verbose)
                printf("i=%d->j=%d, src dev: %d\n", i, j, src_device_i);
            T_ELEM *dst_ptr_b0 = buffers[bit];
            T_ELEM *src_ptr_i = array_d_A[src_device_i] + static_cast<size_t>(i % N_batch) * static_cast<size_t>(N);
            CUDA_CHECK(cudaMemcpyPeerAsync(
                /*dst*/ dst_ptr_b0, /*dstDevice*/ currentDev,
                /*src*/ src_ptr_i, /*srcDevice*/ src_device_i,
                /*count*/ num_bytes_per_col,
                /*stream*/ stream));

            if (move > 0)
            {
                // A[:, i] = buffers[bit ^ 1].copy()
                const int dst_device = i / N_batch;
                T_ELEM *dst_ptr = array_d_A[dst_device] + static_cast<size_t>(i % N_batch) * static_cast<size_t>(N);
                T_ELEM *src_ptr_b1 = buffers[bit ^ 1];
                CUDA_CHECK(cudaMemcpyPeerAsync(
                    /*dst*/ dst_ptr, /*dstDevice*/ dst_device,
                    /*src*/ src_ptr_b1, /*srcDevice*/ currentDev,
                    /*count*/ num_bytes_per_col,
                    /*stream*/ stream));
            }
            // buffers[bit ^ 1] = A[:, j].copy()
            const int src_device_j = j / N_batch;
            T_ELEM *dst_ptr_b1 = buffers[bit ^ 1];
            T_ELEM *src_ptr_j = array_d_A[src_device_j] + static_cast<size_t>(j % N_batch) * static_cast<size_t>(N);
            CUDA_CHECK(cudaMemcpyPeerAsync(
                /*dst*/ dst_ptr_b1, /*dstDevice*/ currentDev,
                /*src*/ src_ptr_j, /*srcDevice*/ src_device_j,
                /*count*/ num_bytes_per_col,
                /*stream*/ stream));

            bit ^= 1;
        }

        if (j != -1)
        {
            // A[:, j] = buffers[bit ^ 1].copy()
            const int dst_device = j / N_batch;
            T_ELEM *dst_ptr = array_d_A[dst_device] + static_cast<size_t>(j % N_batch) * static_cast<size_t>(N);
            T_ELEM *src_ptr_b1 = buffers[bit ^ 1];
            CUDA_CHECK(cudaMemcpyPeerAsync(
                /*dst*/ dst_ptr, /*dstDevice*/ dst_device,
                /*src*/ src_ptr_b1, /*srcDevice*/ currentDev,
                /*count*/ num_bytes_per_col,
                /*stream*/ stream));
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int b = 0; b < 2; ++b)
    {
        CUDA_CHECK(cudaFree(buffers[b]));
        if (g_cusolver_utils_verbose)
            std::printf("memcpyCyclicShard: freed staging [%d] \n", b);
    }
    // Restore original device
    CUDA_CHECK(cudaSetDevice(currentDev));
    CUDA_CHECK(cudaDeviceSynchronize());
}

// type traits
template <typename T>
struct traits;

template <>
struct traits<float>
{
    // scalar type
    typedef float T;
    typedef T S;

    static constexpr T zero = 0.f;
    static constexpr cudaDataType cuda_data_type = CUDA_R_32F;
#if CUDART_VERSION >= 11000
    static constexpr cusolverPrecType_t cusolver_precision_type = CUSOLVER_R_32F;
#endif
    static inline T nan()
    {
        return NAN;
    }
    inline static S abs(T val) { return fabs(val); }

    template <typename RNG>
    inline static T rand(RNG &gen) { return (S)gen(); }

    inline static T add(T a, T b) { return a + b; }

    inline static T mul(T v, S f) { return v * f; }
};

template <>
struct traits<double>
{
    // scalar type
    typedef double T;
    typedef T S;

    static constexpr T zero = 0.;
    static constexpr cudaDataType cuda_data_type = CUDA_R_64F;
#if CUDART_VERSION >= 11000
    static constexpr cusolverPrecType_t cusolver_precision_type = CUSOLVER_R_64F;
#endif

    static inline T nan()
    {
        return NAN;
    }
    inline static S abs(T val) { return fabs(val); }

    template <typename RNG>
    inline static T rand(RNG &gen) { return (S)gen(); }

    inline static T add(T a, T b) { return a + b; }

    inline static T mul(T v, S f) { return v * f; }
};

template <>
struct traits<cuFloatComplex>
{
    // scalar type
    typedef float S;
    typedef cuFloatComplex T;

    static constexpr T zero = {0.f, 0.f};
    static constexpr cudaDataType cuda_data_type = CUDA_C_32F;
#if CUDART_VERSION >= 11000
    static constexpr cusolverPrecType_t cusolver_precision_type = CUSOLVER_C_32F;
#endif

    inline static S abs(T val) { return cuCabsf(val); }

    template <typename RNG>
    inline static T rand(RNG &gen)
    {
        return make_cuFloatComplex((S)gen(), (S)gen());
    }

    static inline T nan()
    {
        return make_cuFloatComplex(NAN, NAN);
    }

    inline static T add(T a, T b) { return cuCaddf(a, b); }
    inline static T add(T a, S b) { return cuCaddf(a, make_cuFloatComplex(b, 0.f)); }

    inline static T mul(T v, S f) { return make_cuFloatComplex(v.x * f, v.y * f); }
};

template <>
struct traits<cuDoubleComplex>
{
    // scalar type
    typedef double S;
    typedef cuDoubleComplex T;

    static constexpr T zero = {0., 0.};
    static constexpr cudaDataType cuda_data_type = CUDA_C_64F;
#if CUDART_VERSION >= 11000
    static constexpr cusolverPrecType_t cusolver_precision_type = CUSOLVER_C_64F;
#endif

    inline static S abs(T val) { return cuCabs(val); }

    template <typename RNG>
    inline static T rand(RNG &gen)
    {
        return make_cuDoubleComplex((S)gen(), (S)gen());
    }

    static inline T nan()
    {
        return make_cuDoubleComplex(NAN, NAN);
    }

    inline static T add(T a, T b) { return cuCadd(a, b); }
    inline static T add(T a, S b) { return cuCadd(a, make_cuDoubleComplex(b, 0.)); }

    inline static T mul(T v, S f) { return make_cuDoubleComplex(v.x * f, v.y * f); }
};

template <typename T>
void print_matrix(const int &m, const int &n, const T *A, const int &lda);

template <>
void print_matrix(const int &m, const int &n, const float *A, const int &lda)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            std::printf("%0.2f ", A[j * lda + i]);
        }
        std::printf("\n");
    }
}

template <>
void print_matrix(const int &m, const int &n, const double *A, const int &lda)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            std::printf("%0.2f ", A[j * lda + i]);
        }
        std::printf("\n");
    }
}

template <>
void print_matrix(const int &m, const int &n, const cuComplex *A, const int &lda)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            std::printf("%0.2f + %0.2fj ", A[j * lda + i].x, A[j * lda + i].y);
        }
        std::printf("\n");
    }
}

template <>
void print_matrix(const int &m, const int &n, const cuDoubleComplex *A, const int &lda)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            std::printf("%0.2f + %0.2fj ", A[j * lda + i].x, A[j * lda + i].y);
        }
        std::printf("\n");
    }
}

#endif // CUSOLVER_GPU_UTILS_H_