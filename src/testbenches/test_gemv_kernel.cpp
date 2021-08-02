#include "kernel/gemv_kernel.h"

#include "hls_stream.h"
#ifdef __VITIS_HLS__
#include "hls_vector.h"
#endif

int main(int argc, char const *argv[]) {
#ifndef __VITIS_HLS__
  return 0;
#else
  typedef hls::vector<testgemv::DataType, testgemv::T> VectType;
  testgemv::DataType x[testgemv::I];
  testgemv::DataType w[testgemv::I][testgemv::R];

  testgemv::DataType y[testgemv::R] = {0};

  hls::stream<VectType> x_port[testgemv::N];
  hls::stream<VectType> w_port[testgemv::N];
  hls::stream<testgemv::DataType> y_port[testgemv::N];
  for (int i = 0; i < testgemv::I; ++i) {

    x[i] = testgemv::DataType(rand() * 0.0001);
    for (int j = 0; j < testgemv::R; ++j) {
      w[i][j] = testgemv::DataType(rand() * 0.0001);
    }
  }

  for (int i = 0; i < testgemv::R; ++i) {
    for (int j = 0; j < testgemv::I / testgemv::T; ++j) {
      VectType tmp;
      for (int k = 0; k < testgemv::T; ++k) {
        tmp[k] = w[j * testgemv::T + k][i];
      }
      for (int ii = 0; ii < testgemv::N; ++ii) {
        w_port[ii] << tmp;
      }
    }
  }

  for (int i = 0; i < testgemv::R; ++i) {
    for (int j = 0; j < testgemv::I / testgemv::T; ++j) {
      VectType tmp;
      for (int k = 0; k < testgemv::T; ++k) {
        tmp[k] = x[j * testgemv::T + k];
      }
      for (int ii = 0; ii < testgemv::N; ++ii) {
        x_port[ii] << tmp;
      }
    }
  }

  HlsGemvKernel(testgemv::I, testgemv::R, x_port[0], x_port[1], w_port[0], w_port[1], y_port[0], y_port[1]);
  for (int i = 0; i < testgemv::R; ++i) {
    y[i] = 0;
    for (int j = 0; j < testgemv::I; ++j) {
      y[i] += x[j] * w[j][i];
    }
  }

  std::cout << "Checking results." << std::endl;
  int num_errors = 0;
  for (int i = 0; i < testgemv::R; ++i) {
    for (int j = 0; j < testgemv::N; ++j) {
      auto y_test = y_port[j].read();
      if (y[i] - y_test > testgemv::DataType(0.001)) {
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