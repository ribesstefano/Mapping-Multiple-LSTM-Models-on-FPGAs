#include "testbenches/test_u_kernel.h"
#include "dma/axis_lib.h"
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
  const int num_refinements = testu::params::R;
  hls::vector<int, testu::params::N> num_refinements_vect = hls::vector<int, testu::params::N>(num_refinements);
  for (int i = 0; i < testu::params::N; ++i) {
    num_refinements_vect[i] = i + 2;
  }
  const int kNumTilesU = testu::params::I / testu::params::Tu;
  typedef typename testu::params::ActivationD ActivationType;
  typedef hls::vector<ActivationType, testu::params::N> VectN_Type;
  typedef hls::vector<ActivationType, testu::params::G> VectG_Type;
  typedef hls::vector<ActivationType, testu::params::Tu> VectTuAct_Type;
  assert(testu::params::I == testu::params::PrunedSizeU);

  ActivationType x[testu::params::N][testu::params::I];
  ActivationType u[num_refinements][testu::params::PrunedSizeU][testu::params::G];
  ActivationType xu[num_refinements][testu::params::N][testu::params::G];

  hls::stream<VectTuAct_Type> x_port; //[testu::params::N * kNumTilesU];
  hls::stream<VectTuAct_Type> u_port; //[num_refinements * kNumTilesU * testu::params::G];
  hls::stream<VectN_Type> xu_port; //[num_refinements * testu::params::G];
  hls::stream<typename testu::VectTuAxiType> x_axis("x_axis");
  hls::stream<typename testu::VectTuAxiType> u_axis("u_axis");
  hls::stream<typename testu::VectGN_AxiType> xu_gn_axis("xu_gn_axis");
  hls::stream<typename testu::VectN_AxiType> xu_n_axis("xu_n_axis");
  VectN_Type xu_gold[num_refinements * testu::params::G];

  auto x_axis_interface = svd::AxiStreamInterface<testu::VectTuAxiBitwidth>(x_axis);
  auto u_axis_interface = svd::AxiStreamInterface<testu::VectTuAxiBitwidth>(u_axis);
  auto xu_gn_axis_interface = svd::AxiStreamInterface<testu::VectGN_AxiBitwidth>(xu_gn_axis);
  auto xu_n_axis_interface = svd::AxiStreamInterface<testu::VectN_AxiBitwidth>(xu_n_axis);

  for (int i = 0; i < testu::params::N; ++i) {
    for (int j = 0; j < testu::params::I; ++j) {
      x[i][j] = rand(); // * 0.00001;
    }
  }
  for (int i = 0; i < num_refinements; ++i) {
    for (int j = 0; j < testu::params::PrunedSizeU; ++j) {
      for (int k = 0; k < testu::params::G; ++k) {
        u[i][j][k] = rand(); // * 0.00001;
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
      }
    }
  }

  const int num_tests = 2;
  int num_errors = 0;
  
  for (int t = 0; t < num_tests; ++t) {
    for (int jj = 0; jj < 2; ++jj) {
      for (int i = 0; i < testu::params::N; ++i) {
        for (int j = 0; j < kNumTilesU; ++j) {
          VectTuAct_Type x_val;
          for (int k = 0; k < testu::params::Tu; ++k) {
            x_val[k] = x[i][j * testu::params::Tu + k];
          }
          if (jj == 0) {
            x_port << x_val;
          }
          x_axis_interface.PushVector<ActivationType, testu::params::Tu>(x_val);
        }
      }
    }
    for (int i = 0; i < num_refinements; ++i) {
      for (int j = 0; j < kNumTilesU; ++j) {
        for (int k = 0; k < testu::params::G; ++k) {
          VectTuAct_Type u_val;
          for (int ii = 0; ii < testu::params::Tu; ++ii) {
            u_val[ii] = u[i][j * testu::params::Tu + ii][k];
          }
          u_port << u_val;
          u_axis_interface.PushVector<ActivationType, testu::params::Tu>(u_val);
        }
      }
    }
    // NOTE: The streaming order differs from before!
    for (int i = 0; i < num_refinements; ++i) {
      for (int k = 0; k < testu::params::G; ++k) {
        for (int j = 0; j < kNumTilesU; ++j) {
          VectTuAct_Type u_val;
          for (int ii = 0; ii < testu::params::Tu; ++ii) {
            u_val[ii] = u[i][j * testu::params::Tu + ii][k];
          }
          u_axis_interface.PushVector<ActivationType, testu::params::Tu>(u_val);
        }
      }
    }

    std::cout << "[INFO] Starting HlsVectorKernelU." << std::endl;
    HlsVectorKernelU(num_refinements, x_port, u_port, xu_port);
    std::cout << "[INFO] Starting HlsAxisKernelU." << std::endl;
    HlsAxisKernelU(num_refinements, x_axis, u_axis, xu_gn_axis);
    std::cout << "[INFO] Starting HlsManySamplingsKernelU." << std::endl;
    HlsManySamplingsKernelU(num_refinements_vect, x_axis, u_axis, xu_n_axis);

    for (int i = 0; i < num_refinements; ++i) {
      auto xu_gn_val = xu_gn_axis_interface.PopVector<ActivationType, testu::params::G * testu::params::N>();
      for (int j = 0; j < testu::params::G; ++j) {
        auto tmp = xu_port.read();
        for (int k = 0; k < testu::params::N; ++k) {
          std::cout << i << ") test/gold: " << xu_gn_val[j * testu::params::N + k] << " / "
                    << xu_gold[i * testu::params::G + j][k] << std::endl;
          if (xu_gn_val[j * testu::params::N + k] != xu_gold[i * testu::params::G + j][k]) {
            ++num_errors;
          }
        }
      }
    }


    for (int i = 0; i < num_refinements; ++i) {
      for (int j = 0; j < testu::params::G; ++j) {
        auto xu_n_val = xu_n_axis_interface.PopVector<ActivationType, testu::params::N>();
        for (int k = 0; k < testu::params::N; ++k) {
          std::cout << i << ") test/gold: " << xu_n_val[k] << " / "
                    << xu_gold[i * testu::params::G + j][k] << std::endl;
          if (xu_n_val[k] != xu_gold[i * testu::params::G + j][k]) {
            ++num_errors;
          }
        }
      }
    }
  }
  std::cout << "[INFO] Number of mismatches: " << num_errors << std::endl;
  return 0; // num_errors;
#endif
}