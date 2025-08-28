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

#include <algorithm>
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <string>
#include "layout.h"

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " [a|b]" << std::endl;
        return 1;
    }
    using T_ELEM = double;

    std::string mode = argv[1];
    const int num_devices = 2;
    const int T_A = 1;  // tile width
    const int T_B = 10;  // tile width
    const int M = 4;    // rows
    const int N = 4;    // cols
    const int NRHS = 2; // cols
    const int lda = N;  // leading dim of h_B
    const int IA = 1;
    const int JA = 1;

    // Set CUDA devices
    std::vector<int> deviceIdA(num_devices);

    for (int i = 0; i < num_devices; i++)
    {
        deviceIdA[i] = i;
    }

    if (mode == "a")
    {
        std::cout << "Running a" << std::endl;
        createMat<T_ELEM>(num_devices, deviceIdA.data(),N, T_A, lda);
        memcpyH2D<T_ELEM>(
            num_devices,
            deviceIdA.data(),
            M,
            N,
            lda,
            N,
            T_A,
            lda,
            IA,
            JA);
    }
    else {
        std::cout << "Running b" << std::endl;
        createMat<T_ELEM>(num_devices, deviceIdA.data(), NRHS, T_A, lda);

        memcpyH2D<T_ELEM>(
            num_devices,
            deviceIdA.data(),
            M,
            NRHS,
            lda,
            1,
            T_B,
            lda,
            IA,
            JA);
    }

    return 0;
}