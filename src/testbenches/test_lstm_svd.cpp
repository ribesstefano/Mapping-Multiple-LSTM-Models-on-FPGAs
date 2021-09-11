#include "testbenches/test_lstm_svd.h"

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
  std::cout << "[INFO] Starting HlsDenseSvd test." << std::endl;
  typedef typename svd::lstm_params::ActivationD ActivationType;
  const int kG = svd::lstm_params::G;
  int num_active_inputs = svd::lstm_params::N;
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
  get_arg(2, svd::lstm_params::N, num_active_inputs);
  get_arg(3, 512, max_R);
  get_arg(4, svd::lstm_params::I, input_size);
  get_arg(5, svd::lstm_params::H, output_size);
  get_arg(6, 32, num_tests);
  int num_refinements[svd::lstm_params::N];
  ActivationType* x = new ActivationType[num_active_inputs * input_size];
  ActivationType* h_prev = new ActivationType[num_active_inputs * output_size];
  ActivationType* c_prev = new ActivationType[num_active_inputs * output_size];
  ActivationType* h_curr = new ActivationType[num_active_inputs * output_size];
  ActivationType* c_curr = new ActivationType[num_active_inputs * output_size];
  ActivationType* u_cur = new ActivationType[max_R * input_size * kG];
  ActivationType* s_cur = new ActivationType[max_R * num_active_inputs * kG];
  ActivationType* v_cur = new ActivationType[max_R * output_size * kG];
  ActivationType* u_rec = new ActivationType[max_R * output_size * kG];
  ActivationType* s_rec = new ActivationType[max_R * num_active_inputs * kG];
  ActivationType* v_rec = new ActivationType[max_R * output_size * kG];
  ActivationType* bias = new ActivationType[num_active_inputs * kG * output_size];
  auto init_random = [&](const int size, ActivationType* x) {
    for (int i = 0; i < size; ++i) {
      if (std::is_same<short, ActivationType>::value) {
        x[i] = ActivationType(rand());
      } else {
        x[i] = ActivationType(rand() * 0.00001);
      }
    }
  };
  for (int i = 0; i < svd::lstm_params::N; ++i) {
    num_refinements[i] = max_R;
  }
  init_random(num_active_inputs * input_size, x);
  init_random(max_R * input_size * kG, u_cur);
  init_random(max_R * num_active_inputs * kG, s_cur);
  init_random(max_R * output_size * kG, v_cur);
  init_random(max_R * output_size * kG, u_rec);
  init_random(max_R * num_active_inputs * kG, s_rec);
  init_random(max_R * output_size * kG, v_rec);
  init_random(num_active_inputs * kG * output_size, bias);
  std::cout << "[INFO] Calling accelerator." << std::endl;
  for (int i = 0; i < num_tests; ++i) {
    HlsWrapperLstmSvd(num_active_inputs, input_size, output_size,
      num_refinements, x, u_cur, s_cur, v_cur, h_prev, u_rec, s_rec, v_rec,
      bias, c_prev, h_curr, c_curr);
  }
  delete[] x;
  delete[] h_prev;
  delete[] c_prev;
  delete[] h_curr;
  delete[] c_curr;
  delete[] u_cur;
  delete[] s_cur;
  delete[] v_cur;
  delete[] u_rec;
  delete[] s_rec;
  delete[] v_rec;
  delete[] bias;
  std::cout << "[INFO] Exiting." << std::endl;
  return 0;
#endif // end __VITIS_HLS__
}