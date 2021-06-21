#include "math/activation_functions.h"

template <>
void svd_sigmoid<float>(const int n, const float* a, float* y) {
// NOTE: Using OpenMP drastically slows down the execution.
// #pragma omp parallel for num_threads(4) private(i)
  for (int i = 0; i < n; ++i) {
    y[i] = 1 / (1 + exp(-a[i]));
  }
}

template <>
void svd_sigmoid<double>(const int n, const double* a, double* y) {
// NOTE: Using OpenMP drastically slows down the execution.
// #pragma omp parallel for num_threads(4) private(i)
  for (int i = 0; i < n; ++i) {
    y[i] = 1 / (1 + exp(-a[i]));
  }
}

template <>
void svd_hard_sigmoid<float>(const int n, const float* a, float* y) {
// NOTE: Using OpenMP drastically slows down the execution.
// #pragma omp parallel for num_threads(4) private(i)
  for (int i = 0; i < n; ++i) {
    if (a[i] < -2.5) {
      y[i] = 0;
    } else if (a[i] > 2.5) {
      y[i] = 1;
    } else {
      y[i] = 0.2 * a[i] + 0.5;
    }
  }
}

template <>
void svd_hard_sigmoid<double>(const int n, const double* a, double* y) {
// NOTE: Using OpenMP drastically slows down the execution.
// #pragma omp parallel for num_threads(4) private(i)
  for (int i = 0; i < n; ++i) {
    if (a[i] < -2.5) {
      y[i] = 0;
    } else if (a[i] > 2.5) {
      y[i] = 1;
    } else {
      y[i] = 0.2 * a[i] + 0.5;
    }
  }
}

template <>
void svd_tanh<float>(const int n, const float* a, float* y) {
// NOTE: Using OpenMP drastically slows down the execution.
// #pragma omp parallel for num_threads(4)
  for (int i = 0; i < n; ++i) {
    y[i] = tanh(a[i]);
  }
}

template <>
void svd_tanh<double>(const int  n, const double* a, double* y) {
// NOTE: Using OpenMP drastically slows down the execution.
// #pragma omp parallel for num_threads(4)
  for (int i = 0; i < n; ++i) {
    y[i] = tanh(a[i]);
  }
}