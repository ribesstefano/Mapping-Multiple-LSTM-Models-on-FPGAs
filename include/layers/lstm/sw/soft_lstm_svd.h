#ifndef LAYERS_LSTM_SW_SOFT_LSTM_SVD_H_
#define LAYERS_LSTM_SW_SOFT_LSTM_SVD_H_

#include "math_utils/blas_utils.h"
#include "math_utils/activation_functions.h"

#ifdef SDS_DESIGN
#include <stdlib.h>
#include "sds_lib.h"
#endif // end SDS_DESIGN

#ifndef __SYNTHESIS__
#include <chrono>
#endif

#ifdef EIGEN_DESIGN
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/StdVector>
#endif

#include <iomanip>
#include <iostream>
#include <vector>

// #define MULTITHREAD_DESIGN
#if defined(MULTITHREAD_DESIGN) && !defined(SDS_DESIGN) && !defined(__SYNTHESIS__)
#include <thread>
#endif

#include "hls_math.h"

#ifdef AP_INT_MAX_W
#undef AP_INT_MAX_W
#define AP_INT_MAX_W 4096
#endif
#include "ap_int.h"

#define FIX8_INT_BIT 3
#define FIX16_INT_BIT 7

namespace svd {

typedef half HalfD;
typedef ap_fixed<8, FIX8_INT_BIT, AP_RND_ZERO, AP_SAT_SYM> Fix8D;
typedef ap_fixed<16, FIX16_INT_BIT, AP_RND_ZERO, AP_SAT_SYM> Fix16D;
typedef ap_fixed<Fix8D::width * 2, Fix8D::iwidth * 2, AP_RND_ZERO, AP_SAT_SYM> Accum8D;
typedef ap_fixed<Fix16D::width * 2, Fix16D::iwidth * 2, AP_RND_ZERO, AP_SAT_SYM> Accum16D;
typedef half AccumHalfD;

/*
 * @todo       Using Eigen library is an attempt to using sparse matrixes
 *             computation (also on the FPGA ARM core).
 */
#ifdef EIGEN_DESIGN
typedef Eigen::SparseMatrix<float, Eigen::RowMajor> SpMatD; // declares a row-major sparse matrix type of float
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatD; // declares a row-major dense matrix type of float
// typedef Eigen::Matrix<float, Eigen::Dynamic, 1, Eigen::RowMajor> VecD; // declares a row-major dense matrix type of float
typedef Eigen::Triplet<float> TripletD;
#endif

/**
 * @brief      Used for performance comparisons against hardware designs.
 *
 *             The current part is computed using gemm, effectively batching the
 *             inputs.
 *
 * @param[in]  verbose        The verbose
 * @param[in]  use_blas       The use blas
 * @param[in]  x              The input vector
 * @param[in]  num_samples    The number samples
 * @param[in]  num_timesteps  The number timesteps
 * @param[in]  n_steps        The n steps
 * @param[in]  input_size     The input size
 * @param[in]  hidden_size    The hidden size
 * @param[in]  cur_i_u        The current i-gate u-component
 * @param[in]  cur_i_s        The current i-gate s-component
 * @param[in]  cur_i_v        The current i-gate v-component
 * @param[in]  cur_f_u        The current f-gate u-component
 * @param[in]  cur_f_s        The current f-gate s-component
 * @param[in]  cur_f_v        The current f-gate v-component
 * @param[in]  cur_c_u        The current c-gate u-component
 * @param[in]  cur_c_s        The current c-gate s-component
 * @param[in]  cur_c_v        The current c-gate v-component
 * @param[in]  cur_o_u        The current o-gate u-component
 * @param[in]  cur_o_s        The current o-gate s-component
 * @param[in]  cur_o_v        The current o-gate v-component
 * @param[in]  rec_i_u        The recurrent i-gate u-component
 * @param[in]  rec_i_s        The recurrent i-gate s-component
 * @param[in]  rec_i_v        The recurrent i-gate v-component
 * @param[in]  rec_f_u        The recurrent f-gate u-component
 * @param[in]  rec_f_s        The recurrent f-gate s-component
 * @param[in]  rec_f_v        The recurrent f-gate v-component
 * @param[in]  rec_c_u        The recurrent c-gate u-component
 * @param[in]  rec_c_s        The recurrent c-gate s-component
 * @param[in]  rec_c_v        The recurrent c-gate v-component
 * @param[in]  rec_o_u        The recurrent o-gate u-component
 * @param[in]  rec_o_s        The recurrent o-gate s-component
 * @param[in]  rec_o_v        The recurrent o-gate v-component
 * @param[in]  bias_i         The bias i
 * @param[in]  bias_f         The bias f
 * @param[in]  bias_c         The bias c
 * @param[in]  bias_o         The bias o
 * @param      out            The out
 */
#ifdef __cplusplus
extern "C"
#endif
void SvdModelSoftwareBatched(const int verbose,
                             const bool use_blas,
                             const float *x,
                             const int num_samples,
                             const int num_timesteps,
                             const int n_steps,
                             const int input_size,
                             const int hidden_size,
                             const float *cur_i_u,
                             const float *cur_i_s,
                             const float *cur_i_v,
                             const float *cur_f_u,
                             const float *cur_f_s,
                             const float *cur_f_v,
                             const float *cur_c_u,
                             const float *cur_c_s,
                             const float *cur_c_v,
                             const float *cur_o_u,
                             const float *cur_o_s,
                             const float *cur_o_v,
                             const float *rec_i_u,
                             const float *rec_i_s,
                             const float *rec_i_v,
                             const float *rec_f_u,
                             const float *rec_f_s,
                             const float *rec_f_v,
                             const float *rec_c_u,
                             const float *rec_c_s,
                             const float *rec_c_v,
                             const float *rec_o_u,
                             const float *rec_o_s,
                             const float *rec_o_v,
                             const float *bias_i,
                             const float *bias_f,
                             const float *bias_c,
                             const float *bias_o,
                             float *out);

/**
 * @brief      Used for performance comparisons against hardware designs.
 *
 *             The current and recurrent parts are computed using gemv in a timestep loop.
 *
 * @param[in]  verbose        The verbose
 * @param[in]  use_blas       The use blas
 * @param[in]  x              The input vector
 * @param[in]  num_samples    The number samples
 * @param[in]  num_timesteps  The number timesteps
 * @param[in]  n_steps        The n steps
 * @param[in]  input_size     The input size
 * @param[in]  hidden_size    The hidden size
 * @param[in]  cur_i_u        The current i-gate u-component
 * @param[in]  cur_i_s        The current i-gate s-component
 * @param[in]  cur_i_v        The current i-gate v-component
 * @param[in]  cur_f_u        The current f-gate u-component
 * @param[in]  cur_f_s        The current f-gate s-component
 * @param[in]  cur_f_v        The current f-gate v-component
 * @param[in]  cur_c_u        The current c-gate u-component
 * @param[in]  cur_c_s        The current c-gate s-component
 * @param[in]  cur_c_v        The current c-gate v-component
 * @param[in]  cur_o_u        The current o-gate u-component
 * @param[in]  cur_o_s        The current o-gate s-component
 * @param[in]  cur_o_v        The current o-gate v-component
 * @param[in]  rec_i_u        The recurrent i-gate u-component
 * @param[in]  rec_i_s        The recurrent i-gate s-component
 * @param[in]  rec_i_v        The recurrent i-gate v-component
 * @param[in]  rec_f_u        The recurrent f-gate u-component
 * @param[in]  rec_f_s        The recurrent f-gate s-component
 * @param[in]  rec_f_v        The recurrent f-gate v-component
 * @param[in]  rec_c_u        The recurrent c-gate u-component
 * @param[in]  rec_c_s        The recurrent c-gate s-component
 * @param[in]  rec_c_v        The recurrent c-gate v-component
 * @param[in]  rec_o_u        The recurrent o-gate u-component
 * @param[in]  rec_o_s        The recurrent o-gate s-component
 * @param[in]  rec_o_v        The recurrent o-gate v-component
 * @param[in]  bias_i         The bias i
 * @param[in]  bias_f         The bias f
 * @param[in]  bias_c         The bias c
 * @param[in]  bias_o         The bias o
 * @param      out            The out
 */
#ifdef __cplusplus
extern "C"
#endif
void SvdModelSoftwareUnbatched(const int verbose, 
                               const bool use_blas,
                               const float *x,
                               const int num_samples,
                               const int num_timesteps,
                               const int n_steps,
                               const int input_size,
                               const int hidden_size,
                               const float *cur_i_u,
                               const float *cur_i_s,
                               const float *cur_i_v,
                               const float *cur_f_u,
                               const float *cur_f_s,
                               const float *cur_f_v,
                               const float *cur_c_u,
                               const float *cur_c_s,
                               const float *cur_c_v,
                               const float *cur_o_u,
                               const float *cur_o_s,
                               const float *cur_o_v,
                               const float *rec_i_u,
                               const float *rec_i_s,
                               const float *rec_i_v,
                               const float *rec_f_u,
                               const float *rec_f_s,
                               const float *rec_f_v,
                               const float *rec_c_u,
                               const float *rec_c_s,
                               const float *rec_c_v,
                               const float *rec_o_u,
                               const float *rec_o_s,
                               const float *rec_o_v,
                               const float *bias_i,
                               const float *bias_f,
                               const float *bias_c,
                               const float *bias_o,
                               float *out);

template <typename DataType, typename AccumType>
void hls_gemv(const int m, const int n, const DataType *a, const DataType* b,
              DataType *c) { 
  AccumType sum = 0;
// #pragma omp parallel for shared(c) schedule(static, 8)
  for (int i = 0; i < m; i++) {
    sum = 0;
    for (int j = 0; j < n; j++) {
      sum += a[i * n + j] * b[j];
    }
    c[i] = sum;
  }
}

template <typename DataType, typename AccumType>
void hls_gemm(const int M, const int N, const int K, const DataType *a,
              const DataType *b, DataType *c) {
  AccumType sum = 0;
// #pragma omp parallel for shared(c) schedule(static, 8)
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += a[i * K + k] * b[k * N + j];
      }
      c[i * N + j] = sum;
    }
  }
}

template <typename Dtype>
void hls_set(const int N, const Dtype alpha, Dtype *Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

template <typename Dtype, typename Atype>
void hls_add(const int n, const Dtype *a, const Dtype *b, Atype *y) {
  for (int i = 0; i < n; ++i) {
    y[i] = a[i] + b[i];
  }
}

template <typename Dtype, typename Atype>
void hls_mul(const int n, const Dtype *a, const Dtype *b, Atype *y) {
  for (int i = 0; i < n; ++i) {
    y[i] = a[i] * b[i];
  }
}

template <typename Dtype>
void hls_transpose(const int n, const int m, const Dtype *x, Dtype *y) {
  for(int i = 0; i < n; ++i) {
    for(int j = 0; j < m; ++j) {
      y[j * n + i] = x[i * m + j];
    }
  }
}

template <typename Dtype>
void hls_sigmoid(const int n, const Dtype *a, Dtype *y) {
  for (int i = 0; i < n; ++i) {
    y[i] = 1 / (1 + exp(-a[i]));
  }
}

template <typename Dtype>
void hls_hard_sigmoid(const int n, const Dtype *a, Dtype *y) {
  for (int i = 0; i < n; ++i) {
    if (a[i] < Dtype(-2.5)) {
      y[i] = 0;
    } else if (a[i] > Dtype(2.5)) {
      y[i] = 1;
    } else {
      y[i] = Dtype(0.2) * a[i] + Dtype(0.5);
    }
  }
}

template <typename Dtype, int TableSize>
void hls_init_tanh_table(Dtype tanh_table[TableSize]) {
  for (int i = 0; i < TableSize; i++) {
    // First, convert from table index to X-value (signed 8-bit, range -4 to +4)
    float in_val = 2 * 4.0 * (i - float(TableSize) / 2.0) / float(TableSize);
    // Next, compute lookup table function
    Dtype real_val = tanh(in_val);
    tanh_table[i] = real_val;
  }
}

template <typename Dtype, typename Dtable, int TableSize>
Dtype hls_tanh_lookup(const Dtype data_in, const Dtable table[TableSize]) {
  // int data_round = data_in * table_size / 8;
  // int index = data_round + 4 * table_size / 8;
  int data_round = data_in * (TableSize >> 3);
  int index = data_round + ((TableSize << 2) >> 3);
  if (index < 0) {
    index = 0;
  }
  if (index > TableSize - 1) {
    index = TableSize - 1;
  }
  return (Dtype) table[index];
}

template <typename Dtype, int TableSize>
void hls_tanh(const int n, const Dtype *a, const Dtype *tanh_table, Dtype *y) {
  for (int i = 0; i < n; ++i) {
    y[i] = hls_tanh_lookup<Dtype, Dtype, TableSize>(a[i], tanh_table);
  }
}

template <typename DtypeIn, typename DtypeOut>
void hls_copy_cast(const int n, const DtypeIn *a, DtypeOut *y) {
  for (int i = 0; i < n; ++i) {
    y[i] = DtypeOut(a[i]);
  }
}

template <typename T, typename A, int TableSize>
void SvdModelLstmTemplatedLatencyCC(const int verbose,
                                     const T *x,
                                     const int num_samples,
                                     const int num_timesteps,
                                     const int n_steps,
                                     const int input_size,
                                     const int hidden_size,
                                     const T *cur_i_u,
                                     const T *cur_i_s,
                                     const T *cur_i_v,
                                     const T *cur_f_u,
                                     const T *cur_f_s,
                                     const T *cur_f_v,
                                     const T *cur_c_u,
                                     const T *cur_c_s,
                                     const T *cur_c_v,
                                     const T *cur_o_u,
                                     const T *cur_o_s,
                                     const T *cur_o_v,
                                     const T *rec_i_u,
                                     const T *rec_i_s,
                                     const T *rec_i_v,
                                     const T *rec_f_u,
                                     const T *rec_f_s,
                                     const T *rec_f_v,
                                     const T *rec_c_u,
                                     const T *rec_c_s,
                                     const T *rec_c_v,
                                     const T *rec_o_u,
                                     const T *rec_o_s,
                                     const T *rec_o_v,
                                     const T *bias_i,
                                     const T *bias_f,
                                     const T *bias_c,
                                     const T *bias_o,
                                     T *out) {
  // ===========================================================================
  // This C++ function is implementing the following python snippet:
  // (Note the extensive use of matrix multiplications)
  // 
  //   m = 6 # input size
  //   n = 8 # output size, i.e. LSTM hidden size (in Keras)
  //   b = 4 # batch size (or timesteps)
  //   k = 2 # num iterations, i.e. n_steps
  //   x = np.random.randn(m, b)
  //   u = np.random.randn(m, k)
  //   v = np.random.randn(n, k)
  //   s = np.random.randn(k)
  //   s2 = np.random.randn(k)
  //   
  //   s_mat = np.repeat(s, m).reshape(k, m).T
  //   us = u * s_mat
  //   
  //   y_not_batched = np.zeros((b, n))
  //   for i in range(b):
  //       ux = x.T[i] @ us
  //       y_not_batched[i] = ux @ v.T
  // ===========================================================================
  // ===========================================================================
  // Define: x.T @ US Matrixes
  // ===========================================================================
  // Current x
  T *cur_i_ux = new T[n_steps];
  T *cur_f_ux = new T[n_steps];
  T *cur_c_ux = new T[n_steps];
  T *cur_o_ux = new T[n_steps];
  // Recurrent x
  T *rec_i_uh = new T[n_steps];
  T *rec_f_uh = new T[n_steps];
  T *rec_c_uh = new T[n_steps];
  T *rec_o_uh = new T[n_steps];

  // Current y
  T *cur_i_y = new T[hidden_size];
  T *cur_f_y = new T[hidden_size];
  T *cur_c_y = new T[hidden_size];
  T *cur_o_y = new T[hidden_size];
  // Recurrent y
  T *rec_i_y = new T[hidden_size];
  T *rec_f_y = new T[hidden_size];
  T *rec_c_y = new T[hidden_size];
  T *rec_o_y = new T[hidden_size];

  // Output y + bias
  T *i_cur_bias = new T[hidden_size];
  T *f_cur_bias = new T[hidden_size];
  T *c_cur_bias = new T[hidden_size];
  T *o_cur_bias = new T[hidden_size];

  T *i_sum = new T[hidden_size];
  T *f_sum = new T[hidden_size];
  T *c_sum = new T[hidden_size];
  T *o_sum = new T[hidden_size];

  T *i_gate = new T[hidden_size];
  T *f_gate = new T[hidden_size];
  T *o_gate = new T[hidden_size];
  T *c_sum_tanh = new T[hidden_size];
  T *c_tanh = new T[hidden_size];
  T *c_lhs = new T[hidden_size];
  T *c_rhs = new T[hidden_size];
  T *c = new T[hidden_size];

  // Current v Transposed
  T *cur_i_v_T = new T[hidden_size * n_steps];
  T *cur_f_v_T = new T[hidden_size * n_steps];
  T *cur_c_v_T = new T[hidden_size * n_steps];
  T *cur_o_v_T = new T[hidden_size * n_steps];
  // Recurrent v Transposed
  T *rec_i_v_T = new T[hidden_size * n_steps];
  T *rec_f_v_T = new T[hidden_size * n_steps];
  T *rec_c_v_T = new T[hidden_size * n_steps];
  T *rec_o_v_T = new T[hidden_size * n_steps];
  // ===========================================================================
  // Compute:
  // - S * U Matrix
  // ===========================================================================
  // Current us1
  T *cur_i_u_T = new T[n_steps * input_size];
  T *cur_f_u_T = new T[n_steps * input_size];
  T *cur_c_u_T = new T[n_steps * input_size];
  T *cur_o_u_T = new T[n_steps * input_size];
  // Recurrent us
  T *rec_i_u_T = new T[n_steps * hidden_size];
  T *rec_f_u_T = new T[n_steps * hidden_size];
  T *rec_c_u_T = new T[n_steps * hidden_size];
  T *rec_o_u_T = new T[n_steps * hidden_size];
  // Current us1
  T *cur_i_us = new T[n_steps * input_size];
  T *cur_f_us = new T[n_steps * input_size];
  T *cur_c_us = new T[n_steps * input_size];
  T *cur_o_us = new T[n_steps * input_size];
  // Recurrent us
  T *rec_i_us = new T[n_steps * hidden_size];
  T *rec_f_us = new T[n_steps * hidden_size];
  T *rec_c_us = new T[n_steps * hidden_size];
  T *rec_o_us = new T[n_steps * hidden_size];

  // ===========================================================================
  // TanH lookup table
  // ===========================================================================
  T tanh_table[TableSize];
  svd::hls_init_tanh_table<T, TableSize>(tanh_table);

  // ===========================================================================
  // NOTE: We need to 'transpose' u in order to generate the us matrix. This is
  // because the multiplication svd_mul() operates row-wise.
  // ===========================================================================
  // BEFORE TRANSPOSE: s.shape = (n_steps)
  // BEFORE TRANSPOSE: u.shape = (n_steps, input_size)
  // BEFORE TRANSPOSE: us.shape = (n_steps, input_size)
  svd::hls_transpose(n_steps, input_size, cur_i_u, cur_i_u_T);
  svd::hls_transpose(n_steps, input_size, cur_f_u, cur_f_u_T);
  svd::hls_transpose(n_steps, input_size, cur_c_u, cur_c_u_T);
  svd::hls_transpose(n_steps, input_size, cur_o_u, cur_o_u_T);
  svd::hls_transpose(n_steps, hidden_size, rec_i_u, rec_i_u_T);
  svd::hls_transpose(n_steps, hidden_size, rec_f_u, rec_f_u_T);
  svd::hls_transpose(n_steps, hidden_size, rec_c_u, rec_c_u_T);
  svd::hls_transpose(n_steps, hidden_size, rec_o_u, rec_o_u_T);
  for (int i = 0; i < input_size; ++i) {
    svd::hls_mul(n_steps, &cur_i_u_T[i * n_steps], cur_i_s, &cur_i_us[i * n_steps]);
    svd::hls_mul(n_steps, &cur_f_u_T[i * n_steps], cur_f_s, &cur_f_us[i * n_steps]);
    svd::hls_mul(n_steps, &cur_c_u_T[i * n_steps], cur_c_s, &cur_c_us[i * n_steps]);
    svd::hls_mul(n_steps, &cur_o_u_T[i * n_steps], cur_o_s, &cur_o_us[i * n_steps]);
  }
  for (int i = 0; i < hidden_size; ++i) {
    svd::hls_mul(n_steps, &rec_i_u_T[i * n_steps], rec_i_s, &rec_i_us[i * n_steps]);
    svd::hls_mul(n_steps, &rec_f_u_T[i * n_steps], rec_f_s, &rec_f_us[i * n_steps]);
    svd::hls_mul(n_steps, &rec_c_u_T[i * n_steps], rec_c_s, &rec_c_us[i * n_steps]);
    svd::hls_mul(n_steps, &rec_o_u_T[i * n_steps], rec_o_s, &rec_o_us[i * n_steps]);
  }
  // ===========================================================================
  // Transpose back current v and current u vectors.
  // ===========================================================================
  // From (input_size, n_steps) to (n_steps, input_size)
  svd::hls_transpose(input_size, n_steps, cur_i_us, cur_i_u_T);
  svd::hls_transpose(input_size, n_steps, cur_f_us, cur_f_u_T);
  svd::hls_transpose(input_size, n_steps, cur_c_us, cur_c_u_T);
  svd::hls_transpose(input_size, n_steps, cur_o_us, cur_o_u_T);
  // From (n_steps, hidden_size) to (hidden_size, n_steps)
  svd::hls_transpose(n_steps, hidden_size, cur_i_v, cur_i_v_T);
  svd::hls_transpose(n_steps, hidden_size, cur_f_v, cur_f_v_T);
  svd::hls_transpose(n_steps, hidden_size, cur_c_v, cur_c_v_T);
  svd::hls_transpose(n_steps, hidden_size, cur_o_v, cur_o_v_T); 
  // ===========================================================================
  // Transpose back recurrent v and recurrent u vectors.
  // ===========================================================================
  // From (hidden_size, n_steps) to (n_steps, hidden_size)
  svd::hls_transpose(hidden_size, n_steps, rec_i_us, rec_i_u_T);
  svd::hls_transpose(hidden_size, n_steps, rec_f_us, rec_f_u_T);
  svd::hls_transpose(hidden_size, n_steps, rec_c_us, rec_c_u_T);
  svd::hls_transpose(hidden_size, n_steps, rec_o_us, rec_o_u_T);
  // From (n_steps, hidden_size) to (hidden_size, n_steps)
  svd::hls_transpose(n_steps, hidden_size, rec_i_v, rec_i_v_T);
  svd::hls_transpose(n_steps, hidden_size, rec_f_v, rec_f_v_T);
  svd::hls_transpose(n_steps, hidden_size, rec_c_v, rec_c_v_T);
  svd::hls_transpose(n_steps, hidden_size, rec_o_v, rec_o_v_T);

  const int kSampleSize = num_timesteps * input_size;

#ifdef SDS_DESIGN
  perf_counter sw_ctr;
#else
#ifndef __SYNTHESIS__
  std::chrono::duration<double> total_time(0);
#endif
#endif
// =============================================================================
// @todo (11/02/2019): openmp NOT WORKING: threads still share resources,
// creating data races and producing incorrect results.
// =============================================================================
// #pragma omp parallel for schedule(static, 8)
// =============================================================================
#ifdef SDS_DESIGN
  sw_ctr.start();
#else
#ifndef __SYNTHESIS__
  auto begin_timestep = std::chrono::high_resolution_clock::now();
#endif
#endif
  for (int i = 0; i < num_samples; ++i) {
    hls_set(hidden_size, (T)0., c);
    hls_set(hidden_size, (T)0., &out[i * hidden_size]);

    for (int j = 0; j < num_timesteps; ++j) {

#if defined(MULTITHREAD_DESIGN) && !defined(SDS_DESIGN) && !defined(__SYNTHESIS__)
      std::thread cur_i_ux_thread(svd::hls_gemv<T, A>, n_steps, input_size, cur_i_u_T, &x[i * kSampleSize + j * input_size], cur_i_ux);
      std::thread cur_f_ux_thread(svd::hls_gemv<T, A>, n_steps, input_size, cur_f_u_T, &x[i * kSampleSize + j * input_size], cur_f_ux);
      std::thread cur_c_ux_thread(svd::hls_gemv<T, A>, n_steps, input_size, cur_c_u_T, &x[i * kSampleSize + j * input_size], cur_c_ux);
      std::thread cur_o_ux_thread(svd::hls_gemv<T, A>, n_steps, input_size, cur_o_u_T, &x[i * kSampleSize + j * input_size], cur_o_ux);

      std::thread rec_i_uh_thread(svd::hls_gemv<T, A>, n_steps, hidden_size, rec_i_u_T, &out[i * hidden_size], rec_i_uh);
      std::thread rec_f_uh_thread(svd::hls_gemv<T, A>, n_steps, hidden_size, rec_f_u_T, &out[i * hidden_size], rec_f_uh);
      std::thread rec_c_uh_thread(svd::hls_gemv<T, A>, n_steps, hidden_size, rec_c_u_T, &out[i * hidden_size], rec_c_uh);
      std::thread rec_o_uh_thread(svd::hls_gemv<T, A>, n_steps, hidden_size, rec_o_u_T, &out[i * hidden_size], rec_o_uh);

      cur_i_ux_thread.join();
      cur_f_ux_thread.join();
      cur_c_ux_thread.join();
      cur_o_ux_thread.join();

      rec_i_uh_thread.join();
      rec_f_uh_thread.join();
      rec_c_uh_thread.join();
      rec_o_uh_thread.join();

      std::thread cur_i_y_thread(svd::hls_gemv<T, A>, hidden_size, n_steps, cur_i_v_T, cur_i_ux, cur_i_y);
      std::thread cur_f_y_thread(svd::hls_gemv<T, A>, hidden_size, n_steps, cur_f_v_T, cur_f_ux, cur_f_y);
      std::thread cur_c_y_thread(svd::hls_gemv<T, A>, hidden_size, n_steps, cur_c_v_T, cur_c_ux, cur_c_y);
      std::thread cur_o_y_thread(svd::hls_gemv<T, A>, hidden_size, n_steps, cur_o_v_T, cur_o_ux, cur_o_y);

      std::thread rec_i_y_thread(svd::hls_gemv<T, A>, hidden_size, n_steps, rec_i_v_T, rec_i_uh, rec_i_y);
      std::thread rec_f_y_thread(svd::hls_gemv<T, A>, hidden_size, n_steps, rec_f_v_T, rec_f_uh, rec_f_y);
      std::thread rec_c_y_thread(svd::hls_gemv<T, A>, hidden_size, n_steps, rec_c_v_T, rec_c_uh, rec_c_y);
      std::thread rec_o_y_thread(svd::hls_gemv<T, A>, hidden_size, n_steps, rec_o_v_T, rec_o_uh, rec_o_y);

      cur_i_y_thread.join();
      cur_f_y_thread.join();
      cur_c_y_thread.join();
      cur_o_y_thread.join();

      rec_i_y_thread.join();
      rec_f_y_thread.join();
      rec_c_y_thread.join();
      rec_o_y_thread.join();
#else
      // =======================================================================
      // Current LSTM gates
      // =======================================================================
      // NOTE: in this unbatched version, current and recurrent gates execution
      // is simmetrical, i.e. same transposed matrices logic.
      // =======================================================================
      // us.T @ x
      svd::hls_gemv<T, A>(n_steps, input_size, cur_i_u_T, &x[i * kSampleSize + j * input_size], cur_i_ux);
      svd::hls_gemv<T, A>(n_steps, input_size, cur_f_u_T, &x[i * kSampleSize + j * input_size], cur_f_ux);
      svd::hls_gemv<T, A>(n_steps, input_size, cur_c_u_T, &x[i * kSampleSize + j * input_size], cur_c_ux);
      svd::hls_gemv<T, A>(n_steps, input_size, cur_o_u_T, &x[i * kSampleSize + j * input_size], cur_o_ux);

      // v.T @ xus
      svd::hls_gemv<T, A>(hidden_size, n_steps, cur_i_v_T, cur_i_ux, cur_i_y);
      svd::hls_gemv<T, A>(hidden_size, n_steps, cur_f_v_T, cur_f_ux, cur_f_y);
      svd::hls_gemv<T, A>(hidden_size, n_steps, cur_c_v_T, cur_c_ux, cur_c_y);
      svd::hls_gemv<T, A>(hidden_size, n_steps, cur_o_v_T, cur_o_ux, cur_o_y);
      // =======================================================================
      // Recurrent LSTM gates
      // =======================================================================
      // us.T @ h
      svd::hls_gemv<T, A>(n_steps, hidden_size, rec_i_u_T, &out[i * hidden_size], rec_i_uh);
      svd::hls_gemv<T, A>(n_steps, hidden_size, rec_f_u_T, &out[i * hidden_size], rec_f_uh);
      svd::hls_gemv<T, A>(n_steps, hidden_size, rec_c_u_T, &out[i * hidden_size], rec_c_uh);
      svd::hls_gemv<T, A>(n_steps, hidden_size, rec_o_u_T, &out[i * hidden_size], rec_o_uh);
      
      // v.T @ hus
      svd::hls_gemv<T, A>(hidden_size, n_steps, rec_i_v_T, rec_i_uh, rec_i_y);
      svd::hls_gemv<T, A>(hidden_size, n_steps, rec_f_v_T, rec_f_uh, rec_f_y);
      svd::hls_gemv<T, A>(hidden_size, n_steps, rec_c_v_T, rec_c_uh, rec_c_y);
      svd::hls_gemv<T, A>(hidden_size, n_steps, rec_o_v_T, rec_o_uh, rec_o_y);
#endif
      // =======================================================================
      // Non linearities
      // =======================================================================
      svd::hls_add(hidden_size, cur_i_y, bias_i, i_cur_bias);
      svd::hls_add(hidden_size, cur_f_y, bias_f, f_cur_bias);
      svd::hls_add(hidden_size, cur_c_y, bias_c, c_cur_bias);
      svd::hls_add(hidden_size, cur_o_y, bias_o, o_cur_bias);

      svd::hls_add(hidden_size, i_cur_bias, rec_i_y, i_sum);
      svd::hls_add(hidden_size, f_cur_bias, rec_f_y, f_sum);
      svd::hls_add(hidden_size, c_cur_bias, rec_c_y, c_sum);
      svd::hls_add(hidden_size, o_cur_bias, rec_o_y, o_sum);

      svd::hls_hard_sigmoid(hidden_size, i_sum, i_gate);
      svd::hls_hard_sigmoid(hidden_size, f_sum, f_gate);
      svd::hls_hard_sigmoid(hidden_size, o_sum, o_gate);
      svd::hls_tanh<T, TableSize>(hidden_size, c_sum, tanh_table, c_sum_tanh);
      svd::hls_mul(hidden_size, c_sum_tanh, i_gate, c_lhs);
      svd::hls_mul(hidden_size, c, f_gate, c_rhs);

      svd::hls_add(hidden_size, c_lhs, c_rhs, c);
      svd::hls_tanh<T, TableSize>(hidden_size, c, tanh_table, c_tanh);
      svd::hls_mul(hidden_size, c_tanh, o_gate, &out[i * hidden_size]);
    }
  }
#ifdef SDS_DESIGN
  sw_ctr.stop();
#else
#ifndef __SYNTHESIS__
  auto end_timestep = std::chrono::high_resolution_clock::now();
  total_time += end_timestep - begin_timestep;
#endif
#endif
  // ===========================================================================
  // NOTE: We are NOT taking into account the time it takes to both setup the u @ s
  // matrices and perform their transpositions because these operations can be
  // done "offline", i.e. can be stored in that form already, performance-wise.
  // ===========================================================================
  if (verbose == 1) {
#ifdef SDS_DESIGN
    auto sw_cycles = sw_ctr.avg_cpu_cycles();
    auto sw_freq = sds_clock_frequency();
    std::cout << "Frequency: " << sw_freq << " ticks/second\n";
    std::cout << "Unbatched SVD: Total CPU cycles: " << sw_cycles << "\n";
    std::cout << "Unbatched SVD: Average CPU cycles per sample: "
              << sw_cycles / num_samples << "\n";
    std::cout << "Unbatched SVD: Average CPU cycles per timesteps: "
              << sw_cycles / num_samples / num_timesteps << "\n";
#else
#ifndef __SYNTHESIS__
    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(total_time).count();
    auto duration_us = duration_ns / 1000.0;
    auto duration_ms = duration_us / 1000.0;
    auto duration_s = duration_ms / 1000.0;
    std::cout << "Unbatched SVD: Total time: " << duration_ms
              << " ms (" << duration_s << " s)."<< "\n";
    std::cout << "Unbatched SVD: Average time per sample: " << duration_ms / num_samples
              << " ms (" << duration_s / num_samples << " s)."<< "\n";
    std::cout << "Unbatched SVD: Average time per timesteps: " << duration_ms / num_samples / num_timesteps
              << " ms (" << duration_s / num_samples / num_timesteps << " s)."<< "\n";
#endif
#endif
  }
  delete[] cur_i_ux;
  delete[] cur_f_ux;
  delete[] cur_c_ux;
  delete[] cur_o_ux;
  delete[] rec_i_uh;
  delete[] rec_f_uh;
  delete[] rec_c_uh;
  delete[] rec_o_uh;
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
  delete[] rec_i_v_T;
  delete[] rec_f_v_T;
  delete[] rec_c_v_T;
  delete[] rec_o_v_T;
  delete[] cur_i_u_T;
  delete[] cur_f_u_T;
  delete[] cur_c_u_T;
  delete[] cur_o_u_T;
  delete[] rec_i_u_T;
  delete[] rec_f_u_T;
  delete[] rec_c_u_T;
  delete[] rec_o_u_T;
  delete[] cur_i_us;
  delete[] cur_f_us;
  delete[] cur_c_us;
  delete[] cur_o_us;
  delete[] rec_i_us;
  delete[] rec_f_us;
  delete[] rec_c_us;
  delete[] rec_o_us;
}

#ifdef __cplusplus
extern "C"
#endif
void SvdModelLstmFix8(const int verbose,
                       const Fix8D *x,
                       const int num_samples,
                       const int num_timesteps,
                       const int n_steps,
                       const int input_size,
                       const int hidden_size,
                       const Fix8D *cur_i_u,
                       const Fix8D *cur_i_s,
                       const Fix8D *cur_i_v,
                       const Fix8D *cur_f_u,
                       const Fix8D *cur_f_s,
                       const Fix8D *cur_f_v,
                       const Fix8D *cur_c_u,
                       const Fix8D *cur_c_s,
                       const Fix8D *cur_c_v,
                       const Fix8D *cur_o_u,
                       const Fix8D *cur_o_s,
                       const Fix8D *cur_o_v,
                       const Fix8D *rec_i_u,
                       const Fix8D *rec_i_s,
                       const Fix8D *rec_i_v,
                       const Fix8D *rec_f_u,
                       const Fix8D *rec_f_s,
                       const Fix8D *rec_f_v,
                       const Fix8D *rec_c_u,
                       const Fix8D *rec_c_s,
                       const Fix8D *rec_c_v,
                       const Fix8D *rec_o_u,
                       const Fix8D *rec_o_s,
                       const Fix8D *rec_o_v,
                       const Fix8D *bias_i,
                       const Fix8D *bias_f,
                       const Fix8D *bias_c,
                       const Fix8D *bias_o,
                       Fix8D *out);

#ifdef __cplusplus
extern "C"
#endif
void SvdModelLstmFix16(const int verbose,
                        const Fix16D *x,
                        const int num_samples,
                        const int num_timesteps,
                        const int n_steps,
                        const int input_size,
                        const int hidden_size,
                        const Fix16D *cur_i_u,
                        const Fix16D *cur_i_s,
                        const Fix16D *cur_i_v,
                        const Fix16D *cur_f_u,
                        const Fix16D *cur_f_s,
                        const Fix16D *cur_f_v,
                        const Fix16D *cur_c_u,
                        const Fix16D *cur_c_s,
                        const Fix16D *cur_c_v,
                        const Fix16D *cur_o_u,
                        const Fix16D *cur_o_s,
                        const Fix16D *cur_o_v,
                        const Fix16D *rec_i_u,
                        const Fix16D *rec_i_s,
                        const Fix16D *rec_i_v,
                        const Fix16D *rec_f_u,
                        const Fix16D *rec_f_s,
                        const Fix16D *rec_f_v,
                        const Fix16D *rec_c_u,
                        const Fix16D *rec_c_s,
                        const Fix16D *rec_c_v,
                        const Fix16D *rec_o_u,
                        const Fix16D *rec_o_s,
                        const Fix16D *rec_o_v,
                        const Fix16D *bias_i,
                        const Fix16D *bias_f,
                        const Fix16D *bias_c,
                        const Fix16D *bias_o,
                        Fix16D *out);

#ifdef __cplusplus
extern "C"
#endif
void SvdModelLstmSoftware(const int verbose,
                           const bool use_blas,
                           const int type, // 0:float, 1:fix8, 2:fix16
                           const float *x,
                           const int num_samples,
                           const int num_timesteps,
                           const int n_steps,
                           const int input_size,
                           const int hidden_size,
                           const float *cur_i_u,
                           const float *cur_i_s,
                           const float *cur_i_v,
                           const float *cur_f_u,
                           const float *cur_f_s,
                           const float *cur_f_v,
                           const float *cur_c_u,
                           const float *cur_c_s,
                           const float *cur_c_v,
                           const float *cur_o_u,
                           const float *cur_o_s,
                           const float *cur_o_v,
                           const float *rec_i_u,
                           const float *rec_i_s,
                           const float *rec_i_v,
                           const float *rec_f_u,
                           const float *rec_f_s,
                           const float *rec_f_v,
                           const float *rec_c_u,
                           const float *rec_c_s,
                           const float *rec_c_v,
                           const float *rec_o_u,
                           const float *rec_o_s,
                           const float *rec_o_v,
                           const float *bias_i,
                           const float *bias_f,
                           const float *bias_c,
                           const float *bias_o,
                           float *out);

#ifdef EIGEN_DESIGN
void FillSparseMatrix(const int m, const int n, const float *dense_mat, SpMatD &sparse_mat);
#endif

#ifdef __cplusplus
extern "C"
#endif
void SvdModelEigenBatched(const int verbose,
                          const bool use_blas,
                          const float *x,
                          const int num_samples,
                          const int num_timesteps,
                          const int n_steps,
                          const int input_size,
                          const int hidden_size,
                          const float *cur_i_u,
                          const float *cur_i_s,
                          const float *cur_i_v,
                          const float *cur_f_u,
                          const float *cur_f_s,
                          const float *cur_f_v,
                          const float *cur_c_u,
                          const float *cur_c_s,
                          const float *cur_c_v,
                          const float *cur_o_u,
                          const float *cur_o_s,
                          const float *cur_o_v,
                          const float *rec_i_u,
                          const float *rec_i_s,
                          const float *rec_i_v,
                          const float *rec_f_u,
                          const float *rec_f_s,
                          const float *rec_f_v,
                          const float *rec_c_u,
                          const float *rec_c_s,
                          const float *rec_c_v,
                          const float *rec_o_u,
                          const float *rec_o_s,
                          const float *rec_o_v,
                          const float *bias_i,
                          const float *bias_f,
                          const float *bias_c,
                          const float *bias_o,
                          float *out);

#ifdef __cplusplus
extern "C"
#endif
void SvdModelEigenUnbatched(const int verbose, 
                            const bool use_blas,
                            float *x,
                            const int num_samples,
                            const int num_timesteps,
                            const int n_steps,
                            const int input_size,
                            const int hidden_size,
                            const float *cur_i_u,
                            const float *cur_i_s,
                            const float *cur_i_v,
                            const float *cur_f_u,
                            const float *cur_f_s,
                            const float *cur_f_v,
                            const float *cur_c_u,
                            const float *cur_c_s,
                            const float *cur_c_v,
                            const float *cur_o_u,
                            const float *cur_o_s,
                            const float *cur_o_v,
                            const float *rec_i_u,
                            const float *rec_i_s,
                            const float *rec_i_v,
                            const float *rec_f_u,
                            const float *rec_f_s,
                            const float *rec_f_v,
                            const float *rec_c_u,
                            const float *rec_c_s,
                            const float *rec_c_v,
                            const float *rec_o_u,
                            const float *rec_o_s,
                            const float *rec_o_v,
                            const float *bias_i,
                            const float *bias_f,
                            const float *bias_c,
                            const float *bias_o,
                            float *out);


#ifdef __cplusplus
extern "C"
#endif
void SvdModelLstmSoftwareBatched(const int verbose,
                             const bool use_blas,
                             const float *x, // (num_samples, num_inputs, num_timesteps, input_size)
                             const int num_inputs,
                             const int num_samples,
                             const int num_timesteps,
                             const int n_steps,
                             const int input_size,
                             const int hidden_size,
                             const float *cur_i_u,
                             const float *cur_i_s,
                             const float *cur_i_v,
                             const float *cur_f_u,
                             const float *cur_f_s,
                             const float *cur_f_v,
                             const float *cur_c_u,
                             const float *cur_c_s,
                             const float *cur_c_v,
                             const float *cur_o_u,
                             const float *cur_o_s,
                             const float *cur_o_v,
                             const float *rec_i_u,
                             const float *rec_i_s,
                             const float *rec_i_v,
                             const float *rec_f_u,
                             const float *rec_f_s,
                             const float *rec_f_v,
                             const float *rec_c_u,
                             const float *rec_c_s,
                             const float *rec_c_v,
                             const float *rec_o_u,
                             const float *rec_o_s,
                             const float *rec_o_v,
                             const float *bias_i,
                             const float *bias_f,
                             const float *bias_c,
                             const float *bias_o,
                             float *out);


} // svd

#endif // end LAYERS_LSTM_SW_SOFT_LSTM_SVD_H_