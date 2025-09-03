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

 #include <cmath>
 #include <functional>
 #include <iostream>
 #include <random>
 #include <stdexcept>
 #include <string>
 
 #include <cuComplex.h>
 #include <cuda_runtime_api.h>
 #include <cublas_api.h>
 #include <cusolverDn.h>
 #include <library_types.h>
 

//  inline absl::Status CudaToStatus(cudaError_t err, const char* file, int line) {
//     if (err == cudaSuccess) return absl::OkStatus();
//     return absl::InternalError(
//         absl::StrFormat("CUDA error %d (%s) at %s:%d",
//                         static_cast<int>(err),
//                         cudaGetErrorString(err),
//                         file, line));
//   }
  
//   inline absl::Status CusolverToStatus(cusolverStatus_t err, const char* file, int line) {
//     if (err == CUSOLVER_STATUS_SUCCESS) return absl::OkStatus();
//     return absl::InternalError(
//         absl::StrFormat("cuSolver error %d at %s:%d",
//                         static_cast<int>(err), file, line));
//   }
  
//   #define CUDA_CHECK(expr) CudaToStatus((expr), __FILE__, __LINE__)
//   #define CUSOLVER_CHECK(expr) CusolverToStatus((expr), __FILE__, __LINE__)

 // CUDA API error checking
 #define CUDA_CHECK(err)                                                                            \
     do {                                                                                           \
         cudaError_t err_ = (err);                                                                  \
         if (err_ != cudaSuccess) {                                                                 \
             printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                          \
             throw std::runtime_error("CUDA error");                                                \
         }                                                                                          \
     } while (0)
 
 // cusolver API error checking
 #define CUSOLVER_CHECK(err)                                                                        \
     do {                                                                                           \
         cusolverStatus_t err_ = (err);                                                             \
         if (err_ != CUSOLVER_STATUS_SUCCESS) {                                                     \
             printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__);                      \
             throw std::runtime_error("cusolver error");                                            \
         }                                                                                          \
     } while (0)
 
 // cublas API error checking
 #define CUBLAS_CHECK(err)                                                                          \
     do {                                                                                           \
         cublasStatus_t err_ = (err);                                                               \
         if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
             printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                        \
             throw std::runtime_error("cublas error");                                              \
         }                                                                                          \
     } while (0)
 
 // cublas API error checking
 #define CUSPARSE_CHECK(err)                                                                        \
     do {                                                                                           \
         cusparseStatus_t err_ = (err);                                                             \
         if (err_ != CUSPARSE_STATUS_SUCCESS) {                                                     \
             printf("cusparse error %d at %s:%d\n", err_, __FILE__, __LINE__);                      \
             throw std::runtime_error("cusparse error");                                            \
         }                                                                                          \
     } while (0)
 
 // memory alignment
 #define ALIGN_TO(A, B) (((A + B - 1) / B) * B)
 
 // device memory pitch alignment
 static const size_t device_alignment = 32;
 
 // type traits
 template <typename T> struct traits;
 
 template <> struct traits<float> {
     // scalar type
     typedef float T;
     typedef T S;
 
     static constexpr T zero = 0.f;
     static constexpr cudaDataType cuda_data_type = CUDA_R_32F;
 #if CUDART_VERSION >= 11000
     static constexpr cusolverPrecType_t cusolver_precision_type = CUSOLVER_R_32F;
 #endif
 
     inline static S abs(T val) { return fabs(val); }
 
     template <typename RNG> inline static T rand(RNG &gen) { return (S)gen(); }
 
     inline static T add(T a, T b) { return a + b; }
 
     inline static T mul(T v, S f) { return v * f; }
 };
 
 template <> struct traits<double> {
     // scalar type
     typedef double T;
     typedef T S;
 
     static constexpr T zero = 0.;
     static constexpr cudaDataType cuda_data_type = CUDA_R_64F;
 #if CUDART_VERSION >= 11000
     static constexpr cusolverPrecType_t cusolver_precision_type = CUSOLVER_R_64F;
 #endif
 
     inline static S abs(T val) { return fabs(val); }
 
     template <typename RNG> inline static T rand(RNG &gen) { return (S)gen(); }
 
     inline static T add(T a, T b) { return a + b; }
 
     inline static T mul(T v, S f) { return v * f; }
 };
 
 template <> struct traits<cuFloatComplex> {
     // scalar type
     typedef float S;
     typedef cuFloatComplex T;
 
     static constexpr T zero = {0.f, 0.f};
     static constexpr cudaDataType cuda_data_type = CUDA_C_32F;
 #if CUDART_VERSION >= 11000
     static constexpr cusolverPrecType_t cusolver_precision_type = CUSOLVER_C_32F;
 #endif
 
     inline static S abs(T val) { return cuCabsf(val); }
 
     template <typename RNG> inline static T rand(RNG &gen) {
         return make_cuFloatComplex((S)gen(), (S)gen());
     }
 
     inline static T add(T a, T b) { return cuCaddf(a, b); }
     inline static T add(T a, S b) { return cuCaddf(a, make_cuFloatComplex(b, 0.f)); }
 
     inline static T mul(T v, S f) { return make_cuFloatComplex(v.x * f, v.y * f); }
 };
 
 template <> struct traits<cuDoubleComplex> {
     // scalar type
     typedef double S;
     typedef cuDoubleComplex T;
 
     static constexpr T zero = {0., 0.};
     static constexpr cudaDataType cuda_data_type = CUDA_C_64F;
 #if CUDART_VERSION >= 11000
     static constexpr cusolverPrecType_t cusolver_precision_type = CUSOLVER_C_64F;
 #endif
 
     inline static S abs(T val) { return cuCabs(val); }
 
     template <typename RNG> inline static T rand(RNG &gen) {
         return make_cuDoubleComplex((S)gen(), (S)gen());
     }
 
     inline static T add(T a, T b) { return cuCadd(a, b); }
     inline static T add(T a, S b) { return cuCadd(a, make_cuDoubleComplex(b, 0.)); }
 
     inline static T mul(T v, S f) { return make_cuDoubleComplex(v.x * f, v.y * f); }
 };
 
 template <typename T> void print_matrix(const int &m, const int &n, const T *A, const int &lda);
 
 template <> void print_matrix(const int &m, const int &n, const float *A, const int &lda) {
     for (int i = 0; i < m; i++) {
         for (int j = 0; j < n; j++) {
             std::printf("%0.2f ", A[j * lda + i]);
         }
         std::printf("\n");
     }
 }
 
 template <> void print_matrix(const int &m, const int &n, const double *A, const int &lda) {
     for (int i = 0; i < m; i++) {
         for (int j = 0; j < n; j++) {
             std::printf("%0.2f ", A[j * lda + i]);
         }
         std::printf("\n");
     }
 }
 
 template <> void print_matrix(const int &m, const int &n, const cuComplex *A, const int &lda) {
     for (int i = 0; i < m; i++) {
         for (int j = 0; j < n; j++) {
             std::printf("%0.2f + %0.2fj ", A[j * lda + i].x, A[j * lda + i].y);
         }
         std::printf("\n");
     }
 }
 
 template <>
 void print_matrix(const int &m, const int &n, const cuDoubleComplex *A, const int &lda) {
     for (int i = 0; i < m; i++) {
         for (int j = 0; j < n; j++) {
             std::printf("%0.2f + %0.2fj ", A[j * lda + i].x, A[j * lda + i].y);
         }
         std::printf("\n");
     }
 }
 
 // Returns cudaDataType value as defined in library_types.h for the string containing type name
 cudaDataType get_cuda_library_type(std::string type_string) {
     if (type_string.compare("CUDA_R_16F") == 0)
         return CUDA_R_16F;
     else if (type_string.compare("CUDA_C_16F") == 0)
         return CUDA_C_16F;
     else if (type_string.compare("CUDA_R_32F") == 0)
         return CUDA_R_32F;
     else if (type_string.compare("CUDA_C_32F") == 0)
         return CUDA_C_32F;
     else if (type_string.compare("CUDA_R_64F") == 0)
         return CUDA_R_64F;
     else if (type_string.compare("CUDA_C_64F") == 0)
         return CUDA_C_64F;
     else if (type_string.compare("CUDA_R_8I") == 0)
         return CUDA_R_8I;
     else if (type_string.compare("CUDA_C_8I") == 0)
         return CUDA_C_8I;
     else if (type_string.compare("CUDA_R_8U") == 0)
         return CUDA_R_8U;
     else if (type_string.compare("CUDA_C_8U") == 0)
         return CUDA_C_8U;
     else if (type_string.compare("CUDA_R_32I") == 0)
         return CUDA_R_32I;
     else if (type_string.compare("CUDA_C_32I") == 0)
         return CUDA_C_32I;
     else if (type_string.compare("CUDA_R_32U") == 0)
         return CUDA_R_32U;
     else if (type_string.compare("CUDA_C_32U") == 0)
         return CUDA_C_32U;
     else
         throw std::runtime_error("Unknown CUDA datatype");
 }