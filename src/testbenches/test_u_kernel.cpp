#include "testbenches/test_u_kernel.h"
#include "dma/axis_lib.h"

#ifdef __VITIS_HLS__
#include "hls_vector.h"
#endif
#include "ap_int.h"
#include "hls_stream.h"
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
  for (int i = testu::params::N; i >= 0; --i) {
    int R_tmp = testu::params::R - 2 * (testu::params::N - i - 1);
    num_refinements_vect[i] = R_tmp > 0 ? R_tmp : 1;
  }
  const int kNumActiveInputs = testu::params::N - 2;
  const int kInputSize_tmp = testu::params::I / 1;
  const int kInputSize = (kInputSize_tmp > testu::params::I) ? testu::params::I : kInputSize_tmp;
  const int kNumTilesU = kInputSize / testu::params::Tu;
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
  hls::stream<typename testu::params::VectTuAxiPacketType> x_axis("x_axis");
  hls::stream<typename testu::params::VectTuAxiPacketType> u_axis("u_axis");
  hls::stream<typename testu::params::VectGN_AxiPacketType> xu_gn_axis("xu_gn_axis");
  hls::stream<typename testu::params::VectN_AxiPacketType> xu_n_axis("xu_n_axis");
  hls::stream<typename testu::params::VectG_AxiPacketType> xu_g_axis("xu_g_axis");
  VectN_Type xu_gold[num_refinements * testu::params::G];

  auto x_axis_interface = svd::AxiStreamPort<testu::params::VectTuAxiWidth>(x_axis);
  auto u_axis_interface = svd::AxiStreamPort<testu::params::VectTuAxiWidth>(u_axis);
  auto xu_gn_axis_interface = svd::AxiStreamPort<testu::params::VectGN_AxiWidth>(xu_gn_axis);
  auto xu_n_axis_interface = svd::AxiStreamPort<testu::params::VectN_AxiWidth>(xu_n_axis);
  auto xu_g_axis_interface = svd::AxiStreamPort<testu::params::VectG_AxiWidth>(xu_g_axis);

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
    for (int j = 0; j < kInputSize; ++j) {
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

// #define TEST_OLD_KERNEL_U
#ifdef TEST_OLD_KERNEL_U
    for (int i = 0; i < testu::params::N; ++i) {
      for (int j = 0; j < kNumTilesU; ++j) {
        VectTuAct_Type x_val;
        for (int k = 0; k < testu::params::Tu; ++k) {
          x_val[k] = x[i][j * testu::params::Tu + k];
        }
        x_port << x_val;
        x_axis_interface.PushVector<ActivationType, testu::params::Tu>(x_val);
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

    std::cout << "[INFO] Starting HlsVectorKernelU." << std::endl;
    HlsVectorKernelU(num_refinements, x_port, u_port, xu_port);
    std::cout << "[INFO] Starting HlsAxisKernelU." << std::endl;
    HlsAxisKernelU(num_refinements, x_axis, u_axis, xu_gn_axis);

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
    std::cout << "[INFO] Number of mismatches: " << num_errors << std::endl;
#endif
    // NOTE: The streaming order differs from before! kNumTilesU is swapped with
    // testu::params::N.
    for (int j = 0; j < kNumTilesU; ++j) {
      for (int i = 0; i < kNumActiveInputs; ++i) {
        VectTuAct_Type x_val;
        for (int k = 0; k < testu::params::Tu; ++k) {
          x_val[k] = x[i][j * testu::params::Tu + k];
        }
        x_axis_interface.PushVector<ActivationType, testu::params::Tu>(x_val);
      }
    }
    // NOTE: The streaming order differs from before! kNumTilesU is swapped with
    // testu::params::G.
    for (int i = 0; i < num_refinements_vect[kNumActiveInputs - 1]; ++i) {
      for (int j = 0; j < kNumTilesU; ++j) {
        for (int k = 0; k < testu::params::G; ++k) {
          VectTuAct_Type u_val;
          for (int ii = 0; ii < testu::params::Tu; ++ii) {
            u_val[ii] = u[i][j * testu::params::Tu + ii][k];
          }
          u_axis_interface.PushVector<ActivationType, testu::params::Tu>(u_val);
        }
      }
    }
    std::cout << "[INFO] Starting HlsKernelU." << std::endl;
    
    int refinements_tmp[testu::params::N];
    for (int i = 0; i < testu::params::N; ++i) {
      refinements_tmp[i] = num_refinements_vect[i];
    }
    HlsKernelU(kNumActiveInputs, kInputSize, refinements_tmp, false, x_axis, u_axis, xu_g_axis);

    testu::params::VectG_Type xu_g_val;
    int total_cnt = 0;
    int last_at = -1;
    for (int i = 0; i < num_refinements_vect[kNumActiveInputs - 1]; ++i) { // R_max
      for (int j = 0; j < kNumActiveInputs; ++j) {
        if (i < num_refinements_vect[j]) {
          bool is_last = xu_g_axis_interface.isLastPopVector<ActivationType, testu::params::G>(xu_g_val);
          if (is_last) {
            last_at = total_cnt;
            std::cout << "[INFO] Last index arrived at iteration: " << last_at << std::endl;
          }
          ++total_cnt;
          // std::cout << "\t[INFO] Reading xu[R." << i << "][N." << j << "]" << std::endl;
          for (int k = 0; k < testu::params::G; ++k) {
            // VectN_Type xu_gold[num_refinements * testu::params::G];
            std::cout << i << ") test/gold: " << xu_g_val[k] << " / "
                      << xu[i][j][k] << std::endl;
            if (xu_g_val[k] != xu[i][j][k]) {
              ++num_errors;
            }
          }
        }
      }
    }
    std::cout << "[INFO] Last index arrived at iteration: " << last_at << std::endl;
    std::cout << "[INFO] Total iterations: " << total_cnt << std::endl;
    std::cout << "[INFO] Number of mismatches: " << num_errors << std::endl;

    while(!xu_n_axis.empty()) {
      auto xu_n_val = xu_n_axis_interface.PopVector<ActivationType, testu::params::N>();
    }
  }
  std::cout << "[INFO] Number of mismatches: " << num_errors << std::endl;
  return 0; // num_errors;
#endif
}