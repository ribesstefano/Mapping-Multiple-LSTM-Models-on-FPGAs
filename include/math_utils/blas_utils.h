#ifndef MATH_BLAS_UTILS_H_
#define MATH_BLAS_UTILS_H_

#ifdef USE_BLAS
#include <cblas.h>
#include <f77blas.h>
#endif

#include <cmath>
#include <cstring>

#ifdef USE_BLAS
template <typename Dtype>
void svd_cpu_gemm(const CBLAS_TRANSPOSE TransA,
  const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
  const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
  Dtype* C);

template <typename Dtype>
void svd_cpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
  const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
  Dtype* y);

template <typename Dtype>
void svd_cpu_gemm_gates(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const Dtype* X, const Dtype* I, const Dtype* F, const Dtype* C,
    const Dtype* O, Dtype* Y_I, Dtype* Y_F, Dtype* Y_C, Dtype* Y_O);
#endif

template <typename Dtype>
void svd_set(const int N, const Dtype alpha, Dtype* Y);

template <typename Dtype>
void svd_copy(const int N, const Dtype *X, Dtype *Y);

template <typename Dtype>
void svd_scal(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void svd_add(const int n, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void svd_mul(const int n, const Dtype* a, const Dtype* b, Dtype* y);

/**
 * @brief      Transpose a matrix x into y.
 *
 * @param[in]  n      The initial number of rows (BEFORE transpose).
 * @param[in]  m      The initial number of columns (BEFORE transpose).
 * @param[in]  x      The input matrix to transpose. Shape: (n, m)
 * @param      y      The output matrix transposed. Shape: (m, n)
 *
 * @tparam     Dtype  The real data type.
 */
template <typename Dtype>
void svd_transpose(const int n, const int m, const Dtype* x, Dtype* y);

#endif // end MATH_BLAS_UTILS_H_