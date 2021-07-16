#include "testbenches/test_u_kernel.h"
#ifdef __VITIS_HLS__
#include "hls_vector.h"
#endif

#include <iostream>
#include <cstdlib>

int main(int argc, char const *argv[]) {
#ifdef COSIM_DESIGN
  srand(1);
#else
  srand(time(NULL));
#endif
  std::cout << "[INFO] Starting HlsKernelU test." << std::endl;
#ifndef __VITIS_HLS__
  return 0;
#else
  const int num_refinements = 8;
  typedef typename testu::params::ActivationD ActivationType;
  typedef hls::vector<ActivationType, testu::params::N> VectN_Type;
  typedef hls::vector<ActivationType, testu::params::G> VectG_Type;

  VectN_Type x_port[testu::params::I];
  VectG_Type u_port[num_refinements * testu::params::PrunedSizeU];
  VectN_Type xu_port[num_refinements * testu::params::G];
  VectN_Type xu_gold[num_refinements * testu::params::G];

  hls::vector<int, 4> testv = 4;
  testv *= 4;
  for (int i = 0; i < 4; ++i) {
    std::cout << testv[i] << std::endl;
  }

  for (int i = 0; i < testu::params::I; ++i) {
    for (int j = 0; j < testu::params::N; ++j) {
      x_port[i][j] = 0.0001 * rand();
      // std::cout << x_port[i][j] << std::endl;
    }
  }
  for (int i = 0; i < num_refinements * testu::params::PrunedSizeU; ++i) {
    for (int j = 0; j < testu::params::N; ++j) {
      u_port[i][j] = 0.0001 * rand();
      // std::cout << u_port[i][j] << std::endl;
    }
  }
  for (int i = 0; i < num_refinements * testu::params::G; ++i) {
    xu_port[i] = VectN_Type(0);
    xu_gold[i] = VectN_Type(0);
  }

  HlsKernelU(num_refinements, x_port, u_port, xu_port);

  assert(testu::params::I == testu::params::PrunedSizeU);
  for (int i = 0; i < num_refinements; ++i) {
    for (int g = 0; g < testu::params::G; ++g) {
      for (int k = 0; k < testu::params::N; ++k) {
        typename testu::params::AccumulationD accum = 0;
        for (int j = 0; j < testu::params::PrunedSizeU; ++j) {
          accum += x_port[j][k] * u_port[i * testu::params::PrunedSizeU + j][g];
        }
        xu_gold[i * testu::params::G + g][k] = typename testu::params::ActivationD(accum);
      }
    }
  }

  int num_errors = 0;
  for (int i = 0; i < num_refinements * testu::params::G; ++i) {
    for (int j = 0; j < testu::params::N; ++j) {
      std::cout << i << ") orig/gold: " << xu_port[i][j] << " / " << xu_gold[i][j] << std::endl;
      if (xu_port[i][j] != xu_gold[i][j]) {
        ++num_errors;
      }
    }
  }
  std::cout << "[INFO] Number of mismatches: " << num_errors << std::endl;
  return num_errors;
#endif
}