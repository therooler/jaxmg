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

// C++
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <cstdio>
#include <cstddef>
#include <library_types.h>
// Cusolver
#include <third_party/gpus/cuda/include/cuComplex.h>
#include <third_party/gpus/cuda/include/cusolverDn.h>

template <typename T_ELEM>
static void memcpyCyclicShard(int num_devices, gpuStream_t stream, const int *deviceIdA, /* <int> dimension num_devices */
                              int M,                                                     /* number of rows in local A, B */
                              int N_batch,                                               /* number of columns in local A, B */
                              /* input */
                              const T_ELEM *h_B, /* device array, h_B is M-by-N_batch with leading dimension ldb  */
                              int ldb,
                              /* output */
                              int N_A,                 /* number of columns of global A */
                              int T_A,                 /* number of columns per column tile */
                              int LLD_A,               /* leading dimension of local A */
                              T_ELEM *array_d_A_packed /* device pointer array of dimension num_devices */
)
{
    int currentDev = 0; /* record current device id */

    /*  Quick return if possible */
    if ((0 >= M) || (0 >= N_batch))
    {
        return;
    }

    /* consistent checking */
    if (ldb < M)
    {
        throw std::runtime_error("Consistency Error.");
    }

    CUDA_CHECK(cudaGetDevice(&currentDev));
    CUDA_CHECK(cudaDeviceSynchronize());

    const int num_blks = ((N_batch * num_devices + T_A - 1) / T_A) / num_devices;

    int nz_blks = 0;
    int global_blk_id = currentDev;
    int T_A_clip = 0;
    for (int JA_blk_id = 0; JA_blk_id < num_blks; JA_blk_id++)
    {
        T_ELEM *d_A = array_d_A_packed + static_cast<size_t>(LLD_A) * T_A * nz_blks;
        const T_ELEM *h_A = h_B + static_cast<size_t>(LLD_A) * T_A * nz_blks;
        T_A_clip = std::min((global_blk_id + 1) * T_A, N_A) - global_blk_id * T_A;
        if (T_A_clip <= 0)
        {
            break;
        }
        CUDA_CHECK(cudaMemcpy2D(d_A, /* dst */
                                     static_cast<size_t>(LLD_A) * sizeof(T_ELEM),
                                     h_A, /* src */
                                     static_cast<size_t>(ldb) * sizeof(T_ELEM),
                                     static_cast<size_t>(M) * sizeof(T_ELEM),
                                     static_cast<size_t>(T_A_clip),
                                     cudaMemcpyDeviceToDevice));
        nz_blks++;
        global_blk_id += num_devices;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename T_ELEM>
static void memcpyShard(int num_devices,
                        int M,       /* number of rows in local A, B */
                        int N_batch, /* number of columns in local A, B */
                        /* input */
                        const T_ELEM *h_B, /* device array, h_B is M-by-N_batch with leading dimension ldb  */
                        int ldb,
                        /* output */
                        int N_A,                 /* number of columns of global A */
                        int LLD_A,               /* leading dimension of local A */
                        T_ELEM *array_d_A_packed /* device pointer array of dimension num_devices */
)
{

    /*  Quick return if possible */
    if ((0 >= M) || (0 >= N_batch))
    {
        return;
    }

    /* consistent checking */
    if (ldb < M)
    {
        throw std::runtime_error("Consistency Error.");
    }
    T_ELEM *d_A = array_d_A_packed;
    const T_ELEM *h_A = h_B;
    CUDA_CHECK(cudaMemcpy2D(d_A, /* dst */
                            static_cast<size_t>(LLD_A) * sizeof(T_ELEM),
                            h_A, /* src */
                            static_cast<size_t>(ldb) * sizeof(T_ELEM),
                            static_cast<size_t>(M) * sizeof(T_ELEM),
                            static_cast<size_t>(N_batch),
                            cudaMemcpyDeviceToDevice));

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