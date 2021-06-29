#include "math_utils/blas_utils.h"

#ifdef USE_BLAS
template<>
void svd_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template<>
void svd_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template <>
void svd_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void svd_cpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void svd_cpu_gemm_gates<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float* X, const float* I, const float* F, const float* C,
    const float* O, float* Y_I, float* Y_F, float* Y_C, float* Y_O) {
  svd_cpu_gemm(TransA, TransB, M, N, K, (float) 1., X, I, (float) 0., Y_I);
  svd_cpu_gemm(TransA, TransB, M, N, K, (float) 1., X, F, (float) 0., Y_F);
  svd_cpu_gemm(TransA, TransB, M, N, K, (float) 1., X, C, (float) 0., Y_C);
  svd_cpu_gemm(TransA, TransB, M, N, K, (float) 1., X, O, (float) 0., Y_O);
}

template <>
void svd_cpu_gemm_gates<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double* X, const double* I, const double* F, const double* C,
    const double* O, double* Y_I, double* Y_F, double* Y_C, double* Y_O) {
  svd_cpu_gemm(TransA, TransB, M, N, K, (double) 1., X, I, (double) 0., Y_I);
  svd_cpu_gemm(TransA, TransB, M, N, K, (double) 1., X, F, (double) 0., Y_F);
  svd_cpu_gemm(TransA, TransB, M, N, K, (double) 1., X, C, (double) 0., Y_C);
  svd_cpu_gemm(TransA, TransB, M, N, K, (double) 1., X, O, (double) 0., Y_O);
}
#else
#endif // end USE_BLAS

template <typename Dtype>
void svd_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

template void svd_set<int>(const int N, const int alpha, int* Y);
template void svd_set<float>(const int N, const float alpha, float* Y);
template void svd_set<double>(const int N, const double alpha, double* Y);

template <typename Dtype>
void svd_copy(const int N, const Dtype* X, Dtype* Y) {
  if (X != Y) {
    memcpy(Y, X, sizeof(Dtype) * N);
  } else {
    for (int i = 0; i < N; ++i) {
      Y[i] = X[i];
    }
  }
}

template void svd_copy<int>(const int N, const int* X, int* Y);
template void svd_copy<unsigned int>(const int N, const unsigned int* X,
    unsigned int* Y);
template void svd_copy<float>(const int N, const float* X, float* Y);
template void svd_copy<double>(const int N, const double* X, double* Y);

template <>
void svd_scal<float>(const int N, const float alpha, float *X) {
#ifdef USE_BLAS
  cblas_sscal(N, alpha, X, 1);
#else
  for (int i = 0; i < N; ++i) {
    X[i] *= alpha; 
  }
#endif
}

template <>
void svd_scal<double>(const int N, const double alpha, double *X) {
#ifdef USE_BLAS
  cblas_dscal(N, alpha, X, 1);
#else
  for (int i = 0; i < N; ++i) {
    X[i] *= alpha; 
  }
#endif
}

template <>
void svd_add<float>(const int n, const float* a, const float* b, float* y) {
// NOTE: Using OpenMP drastically slows down the execution.
// #pragma omp parallel for num_threads(4) private(i)
  for (int i = 0; i < n; ++i) {
    y[i] = a[i] + b[i];
  }
}

template <>
void svd_add<double>(const int n, const double* a, const double* b, double* y) {
// NOTE: Using OpenMP drastically slows down the execution.
// #pragma omp parallel for num_threads(4) private(i)
  for (int i = 0; i < n; ++i) {
    y[i] = a[i] + b[i];
  }
}

template <>
void svd_mul<float>(const int n, const float* a, const float* b, float* y) {
// NOTE: Using OpenMP drastically slows down the execution.
// #pragma omp parallel for num_threads(4) private(i)
  for (int i = 0; i < n; ++i) {
    y[i] = a[i] * b[i];
  }
}

template <>
void svd_mul<double>(const int n, const double* a, const double* b, double* y) {
  for (int i = 0; i < n; ++i) {
    y[i] = a[i] * b[i];
  }
}

template <>
void svd_transpose<float>(const int n, const int m, const float* x, float* y) {
  for(int i = 0; i < n; ++i) {
    for(int j = 0; j < m; ++j) {
      y[j * n + i] = x[i * m + j];
    }
  }
}

template <>
void svd_transpose<double>(const int n, const int m, const double* x, double* y) {
  for(int i = 0; i < n; ++i) {
    for(int j = 0; j < m; ++j) {
      y[j * n + i] = x[i * m + j];
    }
  }
}
