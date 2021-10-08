#include "testbenches/test_u_kernel_pruned.h"
#include "layers/lstm/lstm_data_handler.h"
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
  std::cout << "[INFO] Starting HlsKernelU_Pruned test." << std::endl;
#ifndef __VITIS_HLS__
  return 0;
#else
  const int max_num_refinements = testu::params::R;
  hls::vector<int, testu::params::N> num_refinements_vect = hls::vector<int, testu::params::N>(max_num_refinements);
  for (int i = testu::params::N - 1; i >= 0; --i) {
    int R_tmp = testu::params::R - 2 * (testu::params::N - i - 1);
    num_refinements_vect[i] = R_tmp > 0 ? R_tmp : 1;
  }
  const int kNumActiveInputs = 1; // testu::params::N;
  const int kInputSize_tmp = testu::params::I / 1;
  const int kInputSize = (kInputSize_tmp > testu::params::I) ? testu::params::I : kInputSize_tmp;
  const int kNumTilesU = kInputSize / testu::params::Tu;

  const int kN = testu::params::N;
  const int kR = testu::params::R;
  const int kI = testu::params::I;
  const int kH = testu::params::H;
  const int kNTu = testu::params::MaxNumTu;
  const int kZTu = testu::params::ZTu;
  const int kNTv = testu::params::MaxNumTv;
  const int kZTv = testu::params::ZTv;

  typedef typename testu::params::ActivationD ActivationType;
  typedef hls::vector<ActivationType, testu::params::N> VectN_Type;
  typedef hls::vector<ActivationType, testu::params::G> VectG_Type;
  typedef hls::vector<ActivationType, testu::params::Tu> VectTuAct_Type;
  assert(testu::params::I == testu::params::PrunedSizeU);

  ActivationType x[testu::params::N][testu::params::I];
  ActivationType u[max_num_refinements][testu::params::PrunedSizeU][testu::params::G];
  ActivationType xu[max_num_refinements][testu::params::N][testu::params::G];

  hls::stream<typename testu::params::VectGZTuAxiPacketType> unz_idx_axis("unz_idx_axis");
  hls::stream<typename testu::params::VectTuAxiPacketType> x_axis("x_axis");
  hls::stream<typename testu::params::VectTuAxiPacketType> u_axis("u_axis");
  hls::stream<typename testu::params::VectG_AxiPacketType> xu_axis("xu_axis");

  VectN_Type xu_gold[max_num_refinements * testu::params::G];

  auto unz_idx_interface = svd::AxiStreamPort<testu::params::NumGTuBitsAligned>(unz_idx_axis);
  auto x_interface = svd::AxiStreamPort<testu::params::VectTuAxiWidth>(x_axis);
  auto u_interface = svd::AxiStreamPort<testu::params::VectTuAxiWidth>(u_axis);
  auto xu_interface = svd::AxiStreamPort<testu::params::VectG_AxiWidth>(xu_axis);

  std::cout << "kN: " << kN << std::endl;
  std::cout << "kR: " << kR << std::endl;
  std::cout << "kI: " << kI << std::endl;
  std::cout << "kH: " << kH << std::endl;
  std::cout << "kNTu: " << kNTu << std::endl;
  std::cout << "kZTu: " << kZTu << std::endl;
  std::cout << "kNTv: " << kNTv << std::endl;
  std::cout << "kZTv: " << kZTv << std::endl;

  std::cout << "Setting AcceleratorBlob." << std::endl;
  auto storage = svd::AcceleratorBlob<float, testu::params::ActivationD, 1, 1>(
    kN, kR, kInputSize, kInputSize, kH, kNumTilesU, kZTu, kNTv, kZTv);

  // const int kPrunedSize = storage.get_cur_gates("i")->get_pruned_total_size();
  auto i_gate = storage.get_cur_gates("i")->get_u();
  auto f_gate = storage.get_cur_gates("f")->get_u();
  auto c_gate = storage.get_cur_gates("c")->get_u();
  auto o_gate = storage.get_cur_gates("o")->get_u();

  int* nz_i_idx = i_gate->get_nz_idx();
  int* nz_f_idx = f_gate->get_nz_idx();
  int* nz_c_idx = c_gate->get_nz_idx();
  int* nz_o_idx = o_gate->get_nz_idx();

  std::cout << "i_gate->get_nz_idx(0, 0): ";
  int tmp = i_gate->get_nz_idx(0, 0);
  std::cout << tmp << std::endl;


  std::cout << "x setup." << std::endl;
  for (int i = 0; i < testu::params::N; ++i) {
    for (int j = 0; j < testu::params::I; ++j) {
      if (std::is_same<short, ActivationType>::value) {
        x[i][j] = ActivationType(rand());
      } else {
        x[i][j] = ActivationType(rand() * 0.00001);
      }
    }
  }
  std::cout << "xu setup." << std::endl;
  for (int i = 0; i < max_num_refinements; ++i) {
    for (int j = 0; j < testu::params::N; ++j) {
      for (int k = 0; k < testu::params::G; ++k) {
        xu[i][j][k] = 0;
      }
    }
  }

  std::cout << "[INFO] Generating gold results." << std::endl;
  auto i_weight = i_gate->fix_data();
  auto f_weight = f_gate->fix_data();
  auto c_weight = c_gate->fix_data();
  auto o_weight = o_gate->fix_data();
  for (int i = 0; i < max_num_refinements; ++i) {
    for (int j = 0; j < kInputSize; ++j) {
      for (int ii = 0; ii < testu::params::N; ++ii) {
        xu[i][ii][0] += i_weight[i * kInputSize + j] * storage.get_fix_x(ii)[j];
        xu[i][ii][1] += f_weight[i * kInputSize + j] * storage.get_fix_x(ii)[j];
        xu[i][ii][2] += c_weight[i * kInputSize + j] * storage.get_fix_x(ii)[j];
        xu[i][ii][3] += o_weight[i * kInputSize + j] * storage.get_fix_x(ii)[j];
      }
    }
  }
  std::cout << "[INFO] Generating gold results." << std::endl;
  for (int i = 0; i < max_num_refinements; ++i) {
    for (int j = 0; j < testu::params::N; ++j) {
      for (int k = 0; k < testu::params::G; ++k) {
        // xu_gold[i * testu::params::G + k][j] = xu[i][j][k];
      }
    }
  }

#if 0
  const int num_tests = 2;
  int num_errors = 0;
  
  std::cout << "[INFO] Starting tests." << std::endl;
  for (int t = 0; t < num_tests; ++t) {
    // NOTE: The streaming order differs from before! kNumTilesU is swapped with
    // testu::params::N.
      for (int j = 0; j < kNumTilesU; ++j) {
    for (int i = 0; i < kNumActiveInputs; ++i) {
        VectTuAct_Type x_val;
        for (int k = 0; k < testu::params::Tu; ++k) {
          x_val[k] = x[i][j * testu::params::Tu + k];
        }
        x_interface.PushVector<ActivationType, testu::params::Tu>(x_val);
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
          u_interface.PushVector<ActivationType, testu::params::Tu>(u_val);
        }
      }
    }
    std::cout << "[INFO] Starting HlsKernelU." << std::endl;
    
    int refinements_tmp[testu::params::N];
    for (int i = 0; i < testu::params::N; ++i) {
      refinements_tmp[i] = num_refinements_vect[i];
    }
    // HlsKernelU(kNumActiveInputs, kInputSize, refinements_tmp, false, x_axis, u_axis, xu_axis);
    const int ztu = 0; // kZTu;
    HlsKernelU_Pruned(kNumActiveInputs, kInputSize, refinements_tmp, ztu, unz_idx_axis, x_axis, u_axis, xu_axis);

    testu::params::VectG_Type xu_g_val;
    int total_cnt = 0;
    int last_at = -1;
    for (int i = 0; i < num_refinements_vect[kNumActiveInputs - 1]; ++i) { // R_max
      for (int j = 0; j < kNumActiveInputs; ++j) {
        if (i < num_refinements_vect[j]) {
          bool is_last = xu_interface.isLastPopVector<ActivationType, testu::params::G>(xu_g_val);
          if (is_last) {
            last_at = total_cnt;
            std::cout << "[INFO] Last index arrived at iteration: " << last_at << std::endl;
          }
          ++total_cnt;
          // std::cout << "\t[INFO] Reading xu[R." << i << "][N." << j << "]" << std::endl;
          for (int k = 0; k < testu::params::G; ++k) {
            // VectN_Type xu_gold[max_num_refinements * testu::params::G];
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
  }
  std::cout << "[INFO] Number of mismatches: " << num_errors << std::endl;
  return 0; // num_errors;

#endif
  std::cout << "Exiting." << std::endl;


#endif
}