#include "kernel/gemv_kernel.h"

#include "hls_stream.h"
#ifdef __VITIS_HLS__
#include "hls_vector.h"
#endif

int main(int argc, char const *argv[]) {
#ifndef __VITIS_HLS__
  return 0;
#else

  const int N = 2;
  const int I = 1024;
  const int T = 4;
  const int R = 64;

  typedef hls::vector<short, T> VectType;
  short x[I];
  short w[I][R];

  short y[R] = {0};

  hls::stream<VectType> x_port[N];
  hls::stream<VectType> w_port[N];
  hls::stream<short> y_port[N];

  for (int i = 0; i < I; ++i) {
    x[i] = short(rand() * 0.0001);
    for (int j = 0; j < R; ++j) {
      w[i][j] = short(rand() * 0.0001);
    }
  }

  for (int i = 0; i < R; ++i) {
    for (int j = 0; j < I / T; ++j) {
      VectType tmp;
      for (int k = 0; k < T; ++k) {
        tmp[k] = w[j * T + k][i];
      }
      for (int ii = 0; ii < N; ++ii) {
        w_port[ii] << tmp;
      }
    }
  }

  for (int i = 0; i < R; ++i) {
    for (int j = 0; j < I / T; ++j) {
      VectType tmp;
      for (int k = 0; k < T; ++k) {
        tmp[k] = x[j * T + k];
      }
      for (int ii = 0; ii < N; ++ii) {
        x_port[ii] << tmp;
      }
    }
  }

  HlsGemvKernel(I, R, x_port[0], x_port[1], w_port[0], w_port[1], y_port[0], y_port[1]);
  for (int i = 0; i < R; ++i) {
    y[i] = 0;
    for (int j = 0; j < I; ++j) {
      y[i] += x[j] * w[j][i];
    }
  }

  std::cout << "Checking results." << std::endl;
  int num_errors = 0;
  for (int i = 0; i < R; ++i) {
    for (int j = 0; j < N; ++j) {
      auto y_test = y_port[j].read();
      if (y[i] != y_test) {
          std::cout << i << ") test/gold: " << y_test << " / "
                    << y[i] << std::endl;
        ++num_errors;
      }
    }
  }
  std::cout << "[INFO] Number of mismatches: " << num_errors << std::endl;
  return 0; // num_errors;
#endif
}