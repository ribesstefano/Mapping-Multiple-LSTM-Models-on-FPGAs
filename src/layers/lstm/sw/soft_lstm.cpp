/******************************************************************************
 *  Copyright (c) 2019 Stefano Ribes.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 *  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 *  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *****************************************************************************/
/******************************************************************************
 *
 *
 * @file lstm_software.cpp
 * 
 * @author     Stefano Ribes
 *
 * Library of templated HLS functions for BNN deployment. 
 * This file lists a set of functions to access memory mapped values into 
 * streams 
 *
 *****************************************************************************/
#include "layers/lstm/sw/soft_lstm.h"
#include "math_utils/blas_utils.h"
#include "math_utils/activation_functions.h"

#ifdef SDS_DESIGN
#include <stdlib.h>
#include "sds_lib.h"
#endif

#include <algorithm>
#include <iostream>
#include <ctime>
#include <stdint.h>
#ifndef __SYNTHESIS__
#include <chrono>
#endif

template <typename T>
void MatrixMatrixMul(const int M, const int N, const int K, const T *A,
                     const T *B, T *C) {
  T sum = 0;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {     
      sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

template <typename T, int Tm = 1, int Tn = 1, bool use_tile = false>
void MatrixVectorMul(const int M, const int N, const T* mat, 
  const T* vec, T* result) { 
  if (!use_tile) {
    T sum = 0;
    for (int i = 0; i < M; i++) {
      sum = 0;
      for (int j = 0; j < N; j++) {
        sum += mat[i * N + j] * vec[j];
      }
      result[i] = sum;
    }
  } else {
    for (int i = 0; i < M; ++i) {
      result[i] = 0; // needed for accumulation
    }
    for (int i = 0; i < M; i += Tm) {
      for (int j = 0; j < N; j += Tn) {
        for (int x = i; x < std::min(i + Tm, M); x++) {
          for (int y = j; y < std::min(j + Tn, N); y++) {
            result[x] += mat[x * N + y] * vec[y];
          }
        }
      }
    }
  }
}

#ifdef __cplusplus
extern "C"
#endif
void Lstm(const bool use_blas,
          const float *x,
          const int num_samples,
          const int num_timesteps,
          const int input_size,
          const int hidden_size,
          const float *cur_i,
          const float *cur_f,
          const float *cur_c,
          const float *cur_o,
          const float *rec_i,
          const float *rec_f,
          const float *rec_c,
          const float *rec_o,
          const float *bias_i,
          const float *bias_f,
          const float *bias_c,
          const float *bias_o,
          float *out) {
  // Current y
  float *cur_i_y = new float[num_timesteps * hidden_size];
  float *cur_f_y = new float[num_timesteps * hidden_size];
  float *cur_c_y = new float[num_timesteps * hidden_size];
  float *cur_o_y = new float[num_timesteps * hidden_size];
  // Recurrent y
  float *rec_i_y = new float[hidden_size];
  float *rec_f_y = new float[hidden_size];
  float *rec_c_y = new float[hidden_size];
  float *rec_o_y = new float[hidden_size];

  // Output y
  float *i_cur_bias = new float[hidden_size];
  float *f_cur_bias = new float[hidden_size];
  float *c_cur_bias = new float[hidden_size];
  float *o_cur_bias = new float[hidden_size];

  float *i_sum = new float[hidden_size];
  float *f_sum = new float[hidden_size];
  float *c_sum = new float[hidden_size];
  float *o_sum = new float[hidden_size];

  float *i_gate = new float[hidden_size];
  float *f_gate = new float[hidden_size];
  float *o_gate = new float[hidden_size];
  float *c_sum_tanh = new float[hidden_size];
  float *c_tanh = new float[hidden_size];
  float *c_lhs = new float[hidden_size];
  float *c_rhs = new float[hidden_size];
  float *c = new float[hidden_size];

  float *rec_i_T = new float[hidden_size * hidden_size];
  float *rec_f_T = new float[hidden_size * hidden_size];
  float *rec_c_T = new float[hidden_size * hidden_size];
  float *rec_o_T = new float[hidden_size * hidden_size];

  svd_transpose(hidden_size, hidden_size, rec_i, rec_i_T);
  svd_transpose(hidden_size, hidden_size, rec_f, rec_f_T);
  svd_transpose(hidden_size, hidden_size, rec_c, rec_c_T);
  svd_transpose(hidden_size, hidden_size, rec_o, rec_o_T);

#ifndef __SYNTHESIS__
  auto begin = std::chrono::high_resolution_clock::now();
#endif

#ifdef USE_BLAS
  if (use_blas) {
    for (int i = 0; i < num_samples; ++i) {
      // =========================================================================
      // Current LSTM gates
      // =========================================================================
      // Current x @ w: (num_timesteps, input_size) @ (input_size, hidden_size)
      svd_cpu_gemm(CblasNoTrans, CblasNoTrans, num_timesteps, hidden_size, input_size, (float)1., &x[i * input_size * num_timesteps], cur_i, (float)0., cur_i_y);
      svd_cpu_gemm(CblasNoTrans, CblasNoTrans, num_timesteps, hidden_size, input_size, (float)1., &x[i * input_size * num_timesteps], cur_f, (float)0., cur_f_y);
      svd_cpu_gemm(CblasNoTrans, CblasNoTrans, num_timesteps, hidden_size, input_size, (float)1., &x[i * input_size * num_timesteps], cur_c, (float)0., cur_c_y);
      svd_cpu_gemm(CblasNoTrans, CblasNoTrans, num_timesteps, hidden_size, input_size, (float)1., &x[i * input_size * num_timesteps], cur_o, (float)0., cur_o_y);
  
      svd_set(hidden_size, (float)0., c);
      svd_set(hidden_size, (float)0., &out[i * hidden_size]);

      for (int j = 0; j < num_timesteps; ++j) {
        // =======================================================================
        // Recurrent LSTM gates
        // =======================================================================
        // Recurrent w.T @ h: (hidden_size, hidden_size).T @ (hidden_size, 1)
        svd_cpu_gemv(CblasNoTrans, hidden_size, hidden_size, (float)1., rec_i_T, &out[i * hidden_size], (float)0., rec_i_y);
        svd_cpu_gemv(CblasNoTrans, hidden_size, hidden_size, (float)1., rec_f_T, &out[i * hidden_size], (float)0., rec_f_y);
        svd_cpu_gemv(CblasNoTrans, hidden_size, hidden_size, (float)1., rec_c_T, &out[i * hidden_size], (float)0., rec_c_y);
        svd_cpu_gemv(CblasNoTrans, hidden_size, hidden_size, (float)1., rec_o_T, &out[i * hidden_size], (float)0., rec_o_y);

        // =======================================================================
        // Non Linearities
        // =======================================================================
        svd_add(hidden_size, &cur_i_y[j * hidden_size], bias_i, i_cur_bias);
        svd_add(hidden_size, &cur_f_y[j * hidden_size], bias_f, f_cur_bias);
        svd_add(hidden_size, &cur_c_y[j * hidden_size], bias_c, c_cur_bias);
        svd_add(hidden_size, &cur_o_y[j * hidden_size], bias_o, o_cur_bias);

        svd_add(hidden_size, i_cur_bias, rec_i_y, i_sum);
        svd_add(hidden_size, f_cur_bias, rec_f_y, f_sum);
        svd_add(hidden_size, c_cur_bias, rec_c_y, c_sum);
        svd_add(hidden_size, o_cur_bias, rec_o_y, o_sum);

        svd_hard_sigmoid(hidden_size, i_sum, i_gate);
        svd_hard_sigmoid(hidden_size, f_sum, f_gate);
        svd_hard_sigmoid(hidden_size, o_sum, o_gate);
        svd_tanh(hidden_size, c_sum, c_sum_tanh);
        svd_mul(hidden_size, c_sum_tanh, i_gate, c_lhs);
        svd_mul(hidden_size, c, f_gate, c_rhs);

        svd_add(hidden_size, c_lhs, c_rhs, c);
        svd_tanh(hidden_size, c, c_tanh);
        svd_mul(hidden_size, c_tanh, o_gate, &out[i * hidden_size]);
      }
    }
  } else {
#endif
    for (int i = 0; i < num_samples; ++i) {
      // =========================================================================
      // Current LSTM gates
      // =========================================================================
      // Current x @ w: (num_timesteps, input_size) @ (input_size, hidden_size)
      MatrixMatrixMul(num_timesteps, hidden_size, input_size, &x[i * input_size * num_timesteps], cur_i, cur_i_y);
      MatrixMatrixMul(num_timesteps, hidden_size, input_size, &x[i * input_size * num_timesteps], cur_f, cur_f_y);
      MatrixMatrixMul(num_timesteps, hidden_size, input_size, &x[i * input_size * num_timesteps], cur_c, cur_c_y);
      MatrixMatrixMul(num_timesteps, hidden_size, input_size, &x[i * input_size * num_timesteps], cur_o, cur_o_y);

      svd_set(hidden_size, (float)0., c);
      svd_set(hidden_size, (float)0., &out[i * hidden_size]);

      for (int j = 0; j < num_timesteps; ++j) {
        // =======================================================================
        // Recurrent LSTM gates
        // =======================================================================
        // Recurrent w.T @ h: (hidden_size, hidden_size).T @ (hidden_size, 1)
        MatrixVectorMul(hidden_size, hidden_size, rec_i_T, &out[i * hidden_size], rec_i_y);
        MatrixVectorMul(hidden_size, hidden_size, rec_f_T, &out[i * hidden_size], rec_f_y);
        MatrixVectorMul(hidden_size, hidden_size, rec_c_T, &out[i * hidden_size], rec_c_y);
        MatrixVectorMul(hidden_size, hidden_size, rec_o_T, &out[i * hidden_size], rec_o_y);
        // =======================================================================
        // Non Linearities
        // =======================================================================
        svd_add(hidden_size, &cur_i_y[j * hidden_size], bias_i, i_cur_bias);
        svd_add(hidden_size, &cur_f_y[j * hidden_size], bias_f, f_cur_bias);
        svd_add(hidden_size, &cur_c_y[j * hidden_size], bias_c, c_cur_bias);
        svd_add(hidden_size, &cur_o_y[j * hidden_size], bias_o, o_cur_bias);

        svd_add(hidden_size, i_cur_bias, rec_i_y, i_sum);
        svd_add(hidden_size, f_cur_bias, rec_f_y, f_sum);
        svd_add(hidden_size, c_cur_bias, rec_c_y, c_sum);
        svd_add(hidden_size, o_cur_bias, rec_o_y, o_sum);

        svd_hard_sigmoid(hidden_size, i_sum, i_gate);
        svd_hard_sigmoid(hidden_size, f_sum, f_gate);
        svd_hard_sigmoid(hidden_size, o_sum, o_gate);
        svd_tanh(hidden_size, c_sum, c_sum_tanh);
        svd_mul(hidden_size, c_sum_tanh, i_gate, c_lhs);
        svd_mul(hidden_size, c, f_gate, c_rhs);

        svd_add(hidden_size, c_lhs, c_rhs, c);
        svd_tanh(hidden_size, c, c_tanh);
        svd_mul(hidden_size, c_tanh, o_gate, &out[i * hidden_size]);
      }
    }
#ifdef USE_BLAS
  }
#endif

#ifndef __SYNTHESIS__
  auto end = std::chrono::high_resolution_clock::now();
  auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - 
    begin).count();
  auto duration_us = duration_ns / 1000.0;
  auto duration_ms = duration_us / 1000.0;
  auto duration_s = duration_ms / 1000.0;
#ifdef USE_BLAS
  if (use_blas) {
    std::cout << "[BLAS version] ";
  } else {
    std::cout << "[no BLAS] ";
  }
#endif
  std::cout << "Batched LSTM: Execution time total: " << duration_ms
            << " ms (" << duration_s << " s)."<< "\n";
#ifdef USE_BLAS
  if (use_blas) {
    std::cout << "[BLAS version] ";
  } else {
    std::cout << "[no BLAS] ";
  }
#endif
  std::cout << "Batched LSTM: Execution time per sample: " << duration_ms / num_samples
            << " ms (" << duration_s / num_samples << " s)."<< "\n";
#endif
  delete[] cur_i_y;
  delete[] cur_f_y;
  delete[] cur_c_y;
  delete[] cur_o_y;
  delete[] rec_i_y;
  delete[] rec_f_y;
  delete[] rec_c_y;
  delete[] rec_o_y;
  delete[] i_cur_bias;
  delete[] f_cur_bias;
  delete[] c_cur_bias;
  delete[] o_cur_bias;
  delete[] i_sum;
  delete[] f_sum;
  delete[] c_sum;
  delete[] o_sum;
  delete[] i_gate;
  delete[] f_gate;
  delete[] o_gate;
  delete[] c_sum_tanh;
  delete[] c_tanh;
  delete[] c_lhs;
  delete[] c_rhs;
  delete[] c;
  delete[] rec_i_T;
  delete[] rec_f_T;
  delete[] rec_c_T;
  delete[] rec_o_T;
}

#ifdef __cplusplus
extern "C"
#endif
void LstmUnbatched(const bool use_blas,
                   const float *x,
                   const int num_samples,
                   const int num_timesteps,
                   const int input_size,
                   const int hidden_size,
                   const float *cur_i,
                   const float *cur_f,
                   const float *cur_c,
                   const float *cur_o,
                   const float *rec_i,
                   const float *rec_f,
                   const float *rec_c,
                   const float *rec_o,
                   const float *bias_i,
                   const float *bias_f,
                   const float *bias_c,
                   const float *bias_o,
                   float *out) {
  // Current y
  float *cur_i_y = new float[hidden_size];
  float *cur_f_y = new float[hidden_size];
  float *cur_c_y = new float[hidden_size];
  float *cur_o_y = new float[hidden_size];
  // Recurrent y
  float *rec_i_y = new float[hidden_size];
  float *rec_f_y = new float[hidden_size];
  float *rec_c_y = new float[hidden_size];
  float *rec_o_y = new float[hidden_size];

  // Output y
  float *i_cur_bias = new float[hidden_size];
  float *f_cur_bias = new float[hidden_size];
  float *c_cur_bias = new float[hidden_size];
  float *o_cur_bias = new float[hidden_size];

  float *i_sum = new float[hidden_size];
  float *f_sum = new float[hidden_size];
  float *c_sum = new float[hidden_size];
  float *o_sum = new float[hidden_size];

  float *i_gate = new float[hidden_size];
  float *f_gate = new float[hidden_size];
  float *o_gate = new float[hidden_size];
  float *c_sum_tanh = new float[hidden_size];
  float *c_tanh = new float[hidden_size];
  float *c_lhs = new float[hidden_size];
  float *c_rhs = new float[hidden_size];
  float *c = new float[hidden_size];

  float *x_T = new float[num_timesteps * input_size];

  float *rec_i_T = new float[hidden_size * hidden_size];
  float *rec_f_T = new float[hidden_size * hidden_size];
  float *rec_c_T = new float[hidden_size * hidden_size];
  float *rec_o_T = new float[hidden_size * hidden_size];

  svd_transpose(hidden_size, hidden_size, rec_i, rec_i_T);
  svd_transpose(hidden_size, hidden_size, rec_f, rec_f_T);
  svd_transpose(hidden_size, hidden_size, rec_c, rec_c_T);
  svd_transpose(hidden_size, hidden_size, rec_o, rec_o_T);

  float *cur_i_T = new float[input_size * hidden_size];
  float *cur_f_T = new float[input_size * hidden_size];
  float *cur_c_T = new float[input_size * hidden_size];
  float *cur_o_T = new float[input_size * hidden_size];

  svd_transpose(input_size, hidden_size, cur_i, cur_i_T);
  svd_transpose(input_size, hidden_size, cur_f, cur_f_T);
  svd_transpose(input_size, hidden_size, cur_c, cur_c_T);
  svd_transpose(input_size, hidden_size, cur_o, cur_o_T);

#ifndef __SYNTHESIS__
  auto begin = std::chrono::high_resolution_clock::now();
#endif

  for (int i = 0; i < num_samples; ++i) {
    svd_set(hidden_size, (float)0., c);
    svd_set(hidden_size, (float)0., &out[i * hidden_size]);
    for (int j = 0; j < num_timesteps; ++j) {
      // =======================================================================
      // Current LSTM gates
      // =======================================================================
#ifdef USE_BLAS
      if (use_blas) {
        // Current w @ x: (hidden_size, input_size) @ (input_size, 1)
        svd_cpu_gemv(CblasNoTrans, hidden_size, input_size, (float)1., cur_i_T, &x[i * input_size * num_timesteps + j * input_size], (float)0., cur_i_y);
        svd_cpu_gemv(CblasNoTrans, hidden_size, input_size, (float)1., cur_f_T, &x[i * input_size * num_timesteps + j * input_size], (float)0., cur_f_y);
        svd_cpu_gemv(CblasNoTrans, hidden_size, input_size, (float)1., cur_c_T, &x[i * input_size * num_timesteps + j * input_size], (float)0., cur_c_y);
        svd_cpu_gemv(CblasNoTrans, hidden_size, input_size, (float)1., cur_o_T, &x[i * input_size * num_timesteps + j * input_size], (float)0., cur_o_y);
      } else {
#endif
        MatrixVectorMul(hidden_size, input_size, cur_i_T, &x[i * input_size * num_timesteps + j * input_size], cur_i_y);
        MatrixVectorMul(hidden_size, input_size, cur_f_T, &x[i * input_size * num_timesteps + j * input_size], cur_f_y);
        MatrixVectorMul(hidden_size, input_size, cur_c_T, &x[i * input_size * num_timesteps + j * input_size], cur_c_y);
        MatrixVectorMul(hidden_size, input_size, cur_o_T, &x[i * input_size * num_timesteps + j * input_size], cur_o_y);
#ifdef USE_BLAS
      }
#endif
      // =======================================================================
      // Recurrent LSTM gates
      // =======================================================================
#ifdef USE_BLAS
      if (use_blas) {
        // Recurrent w.T @ h: (hidden_size, hidden_size).T @ (hidden_size, 1)
        svd_cpu_gemv(CblasNoTrans, hidden_size, hidden_size, (float)1., rec_i_T, &out[i * hidden_size], (float)0., rec_i_y);
        svd_cpu_gemv(CblasNoTrans, hidden_size, hidden_size, (float)1., rec_f_T, &out[i * hidden_size], (float)0., rec_f_y);
        svd_cpu_gemv(CblasNoTrans, hidden_size, hidden_size, (float)1., rec_c_T, &out[i * hidden_size], (float)0., rec_c_y);
        svd_cpu_gemv(CblasNoTrans, hidden_size, hidden_size, (float)1., rec_o_T, &out[i * hidden_size], (float)0., rec_o_y);
      } else {
#endif
        // Recurrent w.T @ h: (hidden_size, hidden_size).T @ (hidden_size, 1)
        MatrixVectorMul(hidden_size, hidden_size, rec_i_T, &out[i * hidden_size], rec_i_y);
        MatrixVectorMul(hidden_size, hidden_size, rec_f_T, &out[i * hidden_size], rec_f_y);
        MatrixVectorMul(hidden_size, hidden_size, rec_c_T, &out[i * hidden_size], rec_c_y);
        MatrixVectorMul(hidden_size, hidden_size, rec_o_T, &out[i * hidden_size], rec_o_y);
#ifdef USE_BLAS
      }
#endif
      // =======================================================================
      // Non linearities
      // =======================================================================
      svd_add(hidden_size, cur_i_y, bias_i, i_cur_bias);
      svd_add(hidden_size, cur_f_y, bias_f, f_cur_bias);
      svd_add(hidden_size, cur_c_y, bias_c, c_cur_bias);
      svd_add(hidden_size, cur_o_y, bias_o, o_cur_bias);

      svd_add(hidden_size, i_cur_bias, rec_i_y, i_sum);
      svd_add(hidden_size, f_cur_bias, rec_f_y, f_sum);
      svd_add(hidden_size, c_cur_bias, rec_c_y, c_sum);
      svd_add(hidden_size, o_cur_bias, rec_o_y, o_sum);

      svd_hard_sigmoid(hidden_size, i_sum, i_gate);
      svd_hard_sigmoid(hidden_size, f_sum, f_gate);
      svd_hard_sigmoid(hidden_size, o_sum, o_gate);
      svd_tanh(hidden_size, c_sum, c_sum_tanh);
      svd_mul(hidden_size, c_sum_tanh, i_gate, c_lhs);
      svd_mul(hidden_size, c, f_gate, c_rhs);

      svd_add(hidden_size, c_lhs, c_rhs, c);
      svd_tanh(hidden_size, c, c_tanh);
      svd_mul(hidden_size, c_tanh, o_gate, &out[i * hidden_size]);
    }
  }
#ifndef __SYNTHESIS__
  auto end = std::chrono::high_resolution_clock::now();
  auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - 
    begin).count();
  auto duration_us = duration_ns / 1000.0;
  auto duration_ms = duration_us / 1000.0;
  auto duration_s = duration_ms / 1000.0;
#ifdef USE_BLAS
  if (use_blas) {
    std::cout << "[BLAS version] ";
  } else {
    std::cout << "[no BLAS] ";
  }
#endif
  std::cout << "Unbatched LSTM: Execution time total: " << duration_ms
            << " ms (" << duration_s << " s)."<< "\n";
#ifdef USE_BLAS
  if (use_blas) {
    std::cout << "[BLAS version] ";
  } else {
    std::cout << "[no BLAS] ";
  }
#endif
  std::cout << "Unbatched LSTM: Execution time per sample: " << duration_ms / num_samples
            << " ms (" << duration_s / num_samples << " s)."<< "\n";
#endif
  delete[] cur_i_y;
  delete[] cur_f_y;
  delete[] cur_c_y;
  delete[] cur_o_y;
  delete[] rec_i_y;
  delete[] rec_f_y;
  delete[] rec_c_y;
  delete[] rec_o_y;
  delete[] i_cur_bias;
  delete[] f_cur_bias;
  delete[] c_cur_bias;
  delete[] o_cur_bias;
  delete[] i_sum;
  delete[] f_sum;
  delete[] c_sum;
  delete[] o_sum;
  delete[] i_gate;
  delete[] f_gate;
  delete[] o_gate;
  delete[] c_sum_tanh;
  delete[] c_tanh;
  delete[] c_lhs;
  delete[] c_rhs;
  delete[] c;
  delete[] rec_i_T;
  delete[] rec_f_T;
  delete[] rec_c_T;
  delete[] rec_o_T;
}

#ifdef __cplusplus
extern "C"
#endif
void LstmBatched(const bool use_blas,
                 const float *x,
                 const int batch_size,
                 const int num_samples,
                 const int num_timesteps,
                 const int input_size,
                 const int hidden_size,
                 const float *cur_i,
                 const float *cur_f,
                 const float *cur_c,
                 const float *cur_o,
                 const float *rec_i,
                 const float *rec_f,
                 const float *rec_c,
                 const float *rec_o,
                 const float *bias_i,
                 const float *bias_f,
                 const float *bias_c,
                 const float *bias_o,
                 float *out) {

}