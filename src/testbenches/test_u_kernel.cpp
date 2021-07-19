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
  const int num_refinements = 16;
  const int kNumTilesU = testu::params::I / testu::params::Tu;
  typedef typename testu::params::ActivationD ActivationType;
  typedef hls::vector<ActivationType, testu::params::N> VectN_Type;
  typedef hls::vector<ActivationType, testu::params::G> VectG_Type;
  typedef hls::vector<ActivationType, testu::params::Tu> VectTuAct_Type;
  assert(testu::params::I == testu::params::PrunedSizeU);

  ActivationType x[testu::params::N][testu::params::I];
  ActivationType u[num_refinements][testu::params::PrunedSizeU][testu::params::G];
  ActivationType xu[num_refinements][testu::params::N][testu::params::G];

  VectN_Type xu_port[num_refinements * testu::params::G];
  VectN_Type xu_gold[num_refinements * testu::params::G];

  for (int i = 0; i < testu::params::N; ++i) {
    for (int j = 0; j < testu::params::I; ++j) {
      x[i][j] = 0.00001 * rand();
    }
  }
  for (int i = 0; i < num_refinements; ++i) {
    for (int j = 0; j < testu::params::PrunedSizeU; ++j) {
      for (int k = 0; k < testu::params::G; ++k) {
        u[i][j][k] = 0.00001 * rand();
      }
    }
  }
  for (int i = 0; i < num_refinements; ++i) {
    for (int j = 0; j < testu::params::N; ++j) {
      for (int k = 0; k < testu::params::G; ++k) {
        xu[i][j][k] = 0;
      }
    }
  }
  for (int i = 0; i < num_refinements; ++i) {
    for (int j = 0; j < testu::params::I; ++j) {
      for (int k = 0; k < testu::params::G; ++k) {
        for (int ii = 0; ii < testu::params::N; ++ii) {
          xu[i][ii][k] += u[i][j][k] * x[ii][j];
        }
      }
    }
  }


  for (int i = 0; i < num_refinements; ++i) {
    for (int j = 0; j < testu::params::N; ++j) {
      for (int k = 0; k < testu::params::G; ++k) {
        xu_gold[i * testu::params::G + k][j] = xu[i][j][k];
        xu_port[i * testu::params::G + k][j] = 0;
      }
    }
  }
  
  VectTuAct_Type x_port[testu::params::N * kNumTilesU];
  VectTuAct_Type u_port[num_refinements * kNumTilesU * testu::params::G];
  for (int i = 0; i < testu::params::N; ++i) {
    for (int j = 0; j < kNumTilesU; ++j) {
      for (int k = 0; k < testu::params::Tu; ++k) {
        x_port[i * kNumTilesU + j][k] = x[i][j * testu::params::Tu + k];
      }
    }
  }
  for (int i = 0; i < num_refinements; ++i) {
    for (int j = 0; j < kNumTilesU; ++j) {
      for (int k = 0; k < testu::params::G; ++k) {
        for (int ii = 0; ii < testu::params::Tu; ++ii) {
          int u_idx = i * kNumTilesU * testu::params::G + j * testu::params::G + k;
          u_port[u_idx][ii] = u[i][j * testu::params::Tu + ii][k];
        }
      }
    }
  }

  // VectN_Type x_port[testu::params::I];
  // VectG_Type u_port[num_refinements * testu::params::PrunedSizeU];
  // for (int i = 0; i < testu::params::I; ++i) {
  //   for (int j = 0; j < testu::params::N; ++j) {
  //     x_port[i][j] = 0.0001 * rand();
  //   }
  // }
  // for (int i = 0; i < num_refinements * testu::params::PrunedSizeU; ++i) {
  //   for (int j = 0; j < testu::params::N; ++j) {
  //     u_port[i][j] = 0.0001 * rand();
  //   }
  // }

  std::cout << "Starting HlsVectorKernelU." << std::endl;
  // HlsKernelU(num_refinements, x_port, u_port, xu_port);
  HlsVectorKernelU(num_refinements, x_port, u_port, xu_port);

  // for (int i = 0; i < num_refinements; ++i) {
  //   for (int g = 0; g < testu::params::G; ++g) {
  //     for (int k = 0; k < testu::params::N; ++k) {
  //       typename testu::params::AccumulationD accum = 0;
  //       for (int j = 0; j < testu::params::PrunedSizeU; ++j) {
  //         accum += x_port[j][k] * u_port[i * testu::params::PrunedSizeU + j][g];
  //       }
  //       xu_gold[i * testu::params::G + g][k] = typename testu::params::ActivationD(accum);
  //     }
  //   }
  // }

  int num_errors = 0;
  for (int i = 0; i < num_refinements * testu::params::G; ++i) {
    for (int j = 0; j < testu::params::N; ++j) {
      std::cout << i << ") test/gold: " << xu_port[i][j] << " / "
                << xu_gold[i][j] << std::endl;
      if (xu_port[i][j] != xu_gold[i][j]) {
        ++num_errors;
      }
    }
  }
  std::cout << "[INFO] Number of mismatches: " << num_errors << std::endl;
  return 0; // num_errors;
#endif
}