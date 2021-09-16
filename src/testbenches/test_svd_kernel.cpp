#include "testbenches/test_svd_kernel.h"
#include "dma/axis_lib.h"

#ifdef __VITIS_HLS__
#include "hls_vector.h"
#endif
#include "ap_int.h"
#include "hls_stream.h"
#include <iostream>
#include <cstdlib>

int main(int argc, char const *argv[]) {
#ifndef __VITIS_HLS__
  return 0;
#else
  std::cout << "[INFO] Starting HlsSvdKernel test." << std::endl;
  typedef typename svd::svd_params::ActivationD ActivationType;
  const int kG = svd::svd_params::G;
  int num_active_inputs = svd::svd_params::N;
  int input_size = 16;
  int output_size = 16;
  int max_R = 1;
  int num_tests = 2;
  auto get_arg = [&](const int i, const int max_val, int& arg) {
    if (argc >= i) {
      arg = atoi(argv[i -1]);
      arg = (arg > max_val) ? max_val : arg;
    }
  };
  get_arg(2, svd::svd_params::N, num_active_inputs);
  get_arg(3, 512, max_R);
  get_arg(4, svd::svd_params::I, input_size);
  get_arg(5, svd::svd_params::H, output_size);
  get_arg(6, 32, num_tests);
  int num_refinements[svd::svd_params::N];
  ActivationType* x = new ActivationType[num_active_inputs * input_size];
  ActivationType* u = new ActivationType[max_R * input_size * kG];
  ActivationType* s = new ActivationType[max_R * num_active_inputs * kG];
  ActivationType* v = new ActivationType[max_R * output_size * kG];
  ActivationType* y = new ActivationType[num_active_inputs * kG * output_size];
  hls::stream<typename svd::svd_params::VectTuAxiPacketType> x_port("x_port");
  hls::stream<typename svd::svd_params::VectTuAxiPacketType> u_port("u_port");
  hls::stream<typename svd::svd_params::VectG_AxiPacketType> s_port("s_port");
  hls::stream<typename svd::svd_params::VectTvAxiPacketType> v_port("v_port");
  hls::stream<typename svd::svd_params::VectGTvAxiPacketType> y_port("y_port");
  auto init_random = [&](const int size, ActivationType* x) {
    for (int i = 0; i < size; ++i) {
      if (std::is_same<short, ActivationType>::value) {
        x[i] = ActivationType(rand());
      } else {
        x[i] = ActivationType(rand() * 0.00001);
      }
    }
  };
  auto init_zero = [&](const int size, ActivationType* x) {
    for (int i = 0; i < size; ++i) {
      x[i] = ActivationType(0);
    }
  };
  for (int i = 0; i < svd::svd_params::N; ++i) {
    num_refinements[i] = max_R;
  }
  init_random(num_active_inputs * input_size, x);
  init_random(num_active_inputs * kG * output_size, y);
  init_random(max_R * input_size * kG, u);
  init_random(max_R * num_active_inputs * kG, s);
  init_random(max_R * output_size * kG, v);
  std::cout << "[INFO] Calling accelerator." << std::endl;
  for (int i = 0; i < num_tests; ++i) {
    svd::SetSvdKernelInputs<svd::svd_params>(num_active_inputs, input_size,
      output_size, num_refinements, x, u, s, v, x_port, u_port, s_port, v_port);
    HlsSvdKernel(num_active_inputs, input_size, output_size, num_refinements,
      x_port, u_port, s_port, v_port, y_port);
    svd::GetSvdKernelOutputs<svd::svd_params>(num_active_inputs, output_size,
      y_port, y);
  }
  delete[] x;
  delete[] u;
  delete[] s;
  delete[] v;
  delete[] y;
  std::cout << "[INFO] Exiting." << std::endl;
  return 0;
#endif // end __VITIS_HLS__
}