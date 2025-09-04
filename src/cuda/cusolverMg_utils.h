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

#pragma once

#include <algorithm>
#include <cassert>
#include <stdexcept>

#include "cusolver_utils.h"
#include "jax_utils.h"

#include "xla/ffi/api/ffi.h"

// #ifndef IDX2F
// #define IDX2F(i, j, lda) ((((j) - 1) * (static_cast<size_t>(lda))) + ((i) - 1))
// #endif /* IDX2F */

// #ifndef IDX1F
// #define IDX1F(i) ((i) - 1)
// #endif /* IDX1F */


// namespace ffi = ::xla::ffi;

// template <typename T_ELEM>
// int64_t getWorkspaceBytesT_A(int num_devices, 
//                int N_A,                               /* number of columns of global A */
//                int T_A,                               /* number of columns per column tile */
//                int LLD_A                             /* leading dimension of local A */
//                )
// {
//     const int A_num_blks = (N_A + T_A - 1) / T_A;
//     const int max_A_num_blks_per_device = (A_num_blks + num_devices - 1) / num_devices;
//     int64_t nbytes = sizeof(T_ELEM) * LLD_A * T_A * max_A_num_blks_per_device;
//     return nbytes;
// }

// template <typename T_ELEM>
// static void
// mat_pack2unpack(int num_devices, int N_A,   /* number of columns of global A */
//                 int T_A,                    /* number of columns per column tile */
//                 int LLD_A,                  /* leading dimension of local A */
//                 T_ELEM **array_d_A_packed,  /* host pointer array of dimension num_devices */
//                                             /* output */
//                 T_ELEM **array_d_A_unpacked /* host pointer array of dimension num_blks */
// )
// {
//     const int num_blks = (N_A + T_A - 1) / T_A;

//     for (int p_a = 0; p_a < num_devices; p_a++)
//     {
//         T_ELEM *d_A = array_d_A_packed[p_a];
//         int nz_blks = 0;
//         for (int JA_blk_id = p_a; JA_blk_id < num_blks; JA_blk_id += num_devices)
//         {
//             array_d_A_unpacked[JA_blk_id] = d_A + static_cast<size_t>(LLD_A) * T_A * nz_blks;
//             nz_blks++;
//         }
//     }
// }

/*
 *  A(IA:IA+M-1, JA:JA+N-1) := B(1:M, 1:N)
Global Matrix A Column View (by tile):

Example for N = 4, T_A = 2, 2 GPUs
Columns:     0     1   |   2     3   |   4     5   |   6     7
             └─Tile 0─┘ └─Tile 1─┘   └─Tile 2─┘   └─Tile 3─┘
                 └─── assigned to GPU 0 ──┘
                            └── assigned to GPU 1 ──────┘

Global Matrix A Column View (by tile):
Example for N = 4, T_A = 8, 2 GPUs
Columns:     0     1     2     3   -    -    -    -               Null
             └───────────── Tile 0 ────────────────┘
             └───────── assigned to GPU 0 ─────────┘     └─ assigned to GPU 1 ─┘
*/
// template <typename T_ELEM>
// static void
// mat_packcyclic(int num_devices, int N_A,   /* number of columns of global A */
//                int T_A,                    /* number of columns per column tile */
//                int LLD_A,                  /* leading dimension of local A */
//                T_ELEM **array_d_A_packed,  /* host pointer array of dimension num_devices */
//                                            /* output */
//                T_ELEM **array_d_A_unpacked /* host pointer array of dimension num_blks */
// )
// {
//     const int num_blks = (N_A + T_A - 1) / T_A;
//     std::printf("num_blks: %d\n", num_blks);
//     for (int p_a = 0; p_a < num_devices; p_a++)
//     {
//         T_ELEM *d_A = array_d_A_packed[p_a];
//         std::printf("For GPU:%d\n", p_a);
//         int nz_blks = 0;
//         for (int JA_blk_id = p_a; JA_blk_id < num_blks; JA_blk_id += num_devices)
//         {
//             array_d_A_unpacked[JA_blk_id] = d_A + static_cast<size_t>(LLD_A) * T_A * nz_blks;
//             std::vector<T_ELEM> B(LLD_A * T_A, 0);
//             std::printf("\tAssigning block %d\n", JA_blk_id);
//             std::cout << "\tda %d" << array_d_A_unpacked[JA_blk_id] << std::endl;
//             gpuMemcpy(B.data(), array_d_A_unpacked[JA_blk_id], static_cast<size_t>(LLD_A) * T_A, gpuMemcpyDeviceToHost);
//             // for (int i = 0; i < LLD_A * T_A; i++)
//             // {
//             //     std::cout << B[i] << std::endl;
//             // }
//             nz_blks++;
//         }
//     }
// }

// template <typename T_ELEM>
// static void memcpyCyclic(int num_devices, const int *deviceIdA, /* <int> dimension num_devices */
//                          int M,                                 /* number of rows in local A, B */
//                          int N,                                 /* number of columns in local A, B */
//                          /* input */
//                          const T_ELEM *h_B,                     /* device array, h_B is M-by-N with leading dimension ldb  */
//                          int ldb,
//                          /* output */
//                          int N_A,                  /* number of columns of global A */
//                          int T_A,                  /* number of columns per column tile */
//                          int LLD_A,                /* leading dimension of local A */
//                          T_ELEM **array_d_A_packed /* host pointer array of dimension num_devices */
// )
// {
//     int currentDev = 0; /* record current device id */

//     /*  Quick return if possible */
//     if ((0 >= M) || (0 >= N))
//     {
//         return;
//     }

//     /* consistent checking */
//     if (ldb < M)
//     {
//         throw std::runtime_error("Consistency Error.");
//     }

//     CUDA_CHECK(cudaGetDevice(&currentDev));
//     CUDA_CHECK(cudaDeviceSynchronize());

//     const int num_blks = (N_A + T_A - 1) / T_A;
//     std::printf("num_blks: %d\n", num_blks);
//     std::vector<T_ELEM *> array_d_A_unpacked(num_blks);

//     mat_packcyclic<T_ELEM>(num_devices,
//                            N_A,                      /* number of columns of global A */
//                            T_A,                      /* number of columns per column tile */
//                            LLD_A,                    /* leading dimension of local A */
//                            array_d_A_packed,         /* host pointer array of size num_devices */
//                            /* output */
//                            array_d_A_unpacked.data() /* host pointer array of size num_blks */
//     );

//     /* region of interest is A(IA:IA+N-1, JA:JA+N-1) */

//     const int JA_start_blk_id = 0;
//     const int JA_end_blk_id = (N - 1) / T_A;

//     // for (int p_a = 0; p_a < num_devices; p_a++)
//     // {
//     std::printf("p_a: %d\n", currentDev);
//     std::printf("JA_start_blk_id: %d\n", JA_start_blk_id);
//     std::printf("JA_end_blk_id: %d\n", JA_end_blk_id);
//     std::printf("ldb: %d\n", ldb);
//     /* region of interest: JA_start_blk_id:1:JA_end_blk_id */
//     for (int p_a = 0; p_a < num_devices; p_a++)
//     {
//         std::printf("p_a: %d\n", p_a);
//         std::printf("JA_start_blk_id: %d\n", JA_start_blk_id);
//         std::printf("JA_end_blk_id: %d\n", JA_end_blk_id);
//         for (int JA_blk_id = p_a; JA_blk_id <= JA_end_blk_id; JA_blk_id += num_devices)
//         {
//             std::printf("\tJA_blk_id: %d\n", JA_blk_id);
//             if (JA_blk_id < JA_start_blk_id)
//             {
//                 continue;
//             }
//             /*
//              * process column block of A
//              *       A(A_start_row:M_A, A_start_col : (A_start_col + IT_A-1) )
//              */
//             const int IBX_A = (1 + JA_blk_id * T_A); /* base-1 */
//             const int A_start_col = IBX_A;           /* base-1 */
//             const int A_start_row = 1;               /* base-1 */

//             const int bdd = std::min(N, (IBX_A + T_A - 1));
//             const int IT_A = std::min(T_A, (bdd - A_start_col + 1));

//             const int loc_A_start_row = A_start_row;               /* base-1 */
//             const int loc_A_start_col = (A_start_col - IBX_A) + 1; /* base-1 */
//             std::printf("\t\tIBX_A: %d, A_start_row: %d, A_start_col: %d\n", IBX_A, A_start_row, A_start_col);
//             std::printf("\t\tIT_A: %d, loc_A_start_row: %d, loc_A_start_col: %d\n", IT_A, loc_A_start_row, loc_A_start_col);
//             std::printf("\t\tMoving d_A by: %d\n", IDX2F(loc_A_start_row, loc_A_start_col, LLD_A));
//             std::printf("\t\tMoving h_A by: %d\n", IDX2F(A_start_row, A_start_col, ldb));
//             T_ELEM *d_A =
//                 array_d_A_packed[JA_blk_id] + IDX2F(loc_A_start_row, loc_A_start_col, LLD_A);
//             const T_ELEM *h_A = h_B + IDX2F(A_start_row, A_start_col, ldb);
//             std::vector<T_ELEM> data_block(static_cast<size_t>(LLD_A), 0);
//             CUDA_CHECK(cudaMemcpy2D(d_A, /* dst */
//                                     static_cast<size_t>(LLD_A) * sizeof(T_ELEM),
//                                     h_A, /* src */
//                                     static_cast<size_t>(ldb) * sizeof(T_ELEM),
//                                     static_cast<size_t>(M) * sizeof(T_ELEM),
//                                     static_cast<size_t>(IT_A),
//                                     cudaMemcpyDeviceToDevice));
//             CUDA_CHECK(cudaMemcpy(data_block.data(), d_A, static_cast<size_t>(M) * sizeof(T_ELEM) * static_cast<size_t>(IT_A), gpuMemcpyDeviceToHost));
//             for (int i = 0; i < static_cast<size_t>(LLD_A); i++)
//             {
//                 std::cout << "b:" << data_block[i] << std::endl;
//             }
//             // for (int i = 0; i < static_cast<size_t>(LLD_A); i++)
//             // {
//             //     std::cout << data_block[i] << std::endl;
//             // }
//         }

//     } /* for each tile per device */
//     // }
//     CUDA_CHECK(cudaDeviceSynchronize());
// }

template <typename T_ELEM>
static void memcpyCyclicShard(int num_devices, const int *deviceIdA, /* <int> dimension num_devices */
                              int M,                                 /* number of rows in local A, B */
                              int N_batch,                           /* number of columns in local A, B */
                              /* input */
                              const T_ELEM *h_B,                     /* device array, h_B is M-by-N_batch with leading dimension ldb  */
                              int ldb,
                              /* output */
                              int N_A,                  /* number of columns of global A */
                              int T_A,                  /* number of columns per column tile */
                              int LLD_A,                /* leading dimension of local A */
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
    
    // std::printf("memcopyCyclic num_blks: %d\n", num_blks);
    int nz_blks = 0;
    int global_blk_id = currentDev;
    int T_A_clip=0;
    for (int JA_blk_id = 0; JA_blk_id < num_blks; JA_blk_id++)
    {
        // std::printf("JA_blk_id: %d\n", JA_blk_id);
        T_ELEM *d_A = array_d_A_packed + static_cast<size_t>(LLD_A) * T_A * nz_blks;
        const T_ELEM *h_A = h_B + static_cast<size_t>(LLD_A) * T_A * nz_blks;
        T_A_clip = min((global_blk_id +1) * T_A, N_A) - global_blk_id * T_A;
        // std::printf("nz_blks: %d\n", nz_blks);
        // std::printf("N_A: %d, T_A: %d\n", N_A, T_A);
        // std::printf("\tglobal_blk_id: %d, T_A_clip: %d\n", global_blk_id, T_A_clip);
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
    
        // std::printf(" nbytes to copy: %d\n", static_cast<size_t>(M) * sizeof(T_ELEM) * static_cast<size_t>(T_A_clip));
        // std::vector<T_ELEM> data_block(static_cast<size_t>(LLD_A) * static_cast<size_t>(T_A_clip), 0);
  
        // CUDA_CHECK(cudaMemcpy(data_block.data(), d_A, static_cast<size_t>(M) * sizeof(T_ELEM) * static_cast<size_t>(T_A_clip), gpuMemcpyDeviceToHost));
        // for (int i = 0; i < static_cast<size_t>(LLD_A)* static_cast<size_t>(T_A_clip); i++)
        // {
        //     std::cout << "A:" << data_block[i] << std::endl;
        // }
        // for (int i = 0; i < static_cast<size_t>(LLD_A) * static_cast<size_t>(T_A); i++)
        // {
        //     std::cout << data_block[i] << std::endl;
        // }
        nz_blks++;
        global_blk_id += num_devices;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}