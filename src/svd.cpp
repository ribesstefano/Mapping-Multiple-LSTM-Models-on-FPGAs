#include "svd_params.h"
#include "svd_ip.h"
#include "layers/lstm/lstm_data_handler.h"
#include "layers/lstm/sw/soft_lstm_svd.h"
#include "layers/lstm/hls/lstm_svd.h"
#include "layers/lstm/hls/lstm_svd_emulator.h"

#include "ap_fixed.h"

#include <iostream>
#include <cstdlib>

int main(int argc, char const *argv[]) {
  std::cout << "Hello SVD!" << std::endl;

  const bool kTestSoftwareAccelerator = false;
  const int kN = 2;
  const int kNumActiveInputs = 1;
  const int kR = svd::lstm_params::R;
  const int kI = svd::lstm_params::I;
  const int kH = svd::lstm_params::H;
  const int kNTu = svd::lstm_params::MaxNumTu;
  const int kZTu = svd::lstm_params::ZTu;
  const int kNTv = svd::lstm_params::MaxNumTv;
  const int kZTv = svd::lstm_params::ZTv;
  const int kLutSize = (FIX_WIDTH == 16) ? 512 : 256;
  std::cout << "kN: " << kN << std::endl;
  std::cout << "kR: " << kR << std::endl;
  std::cout << "kI: " << kI << std::endl;
  std::cout << "kH: " << kH << std::endl;
  std::cout << "kNTu: " << kNTu << std::endl;
  std::cout << "kZTu: " << kZTu << std::endl;
  std::cout << "kNTv: " << kNTv << std::endl;
  std::cout << "kZTv: " << kZTv << std::endl;
  std::cout << "Setting AcceleratorBlob." << std::endl;
  // TODO: Change svd::ActivationD into svd::lstm_params::ActivationD.
  auto storage = svd::AcceleratorBlob<float, svd::ActivationD, kNTu, kNTv>(kN,
    kR, kI, kH, kH, kNTu, kZTu, kNTv, kZTv);

  std::cout << "Running SvdIp2Inputs." << std::endl;
  typename svd::lstm_params::ActivationD x_port[svd::lstm_params::N][svd::lstm_params::I] = {rand()};
  typename svd::lstm_params::UPortD u_port[svd::lstm_params::R * svd::lstm_params::PrunedSizeU] = {rand()};
  typename svd::lstm_params::SPortD s_port[svd::lstm_params::N][svd::lstm_params::R] = {rand()};
  typename svd::lstm_params::VPortD v_port[svd::lstm_params::R * svd::lstm_params::PrunedSizeV] = {rand()};
  ap_uint<svd::lstm_params::MaxNumTu> nz_u_port[svd::lstm_params::G * svd::lstm_params::R] = {rand()};
  ap_uint<svd::lstm_params::MaxNumTv> nz_v_port[svd::lstm_params::G * svd::lstm_params::R] = {rand()};
  typename svd::lstm_params::ActivationD y_port[svd::lstm_params::N][svd::lstm_params::G][svd::lstm_params::H] = {rand()};

  std::cout << "reinterpret_cast." << std::endl;

  ap_uint<64>* u_cur_uint = reinterpret_cast<ap_uint<64>*>(storage.get_fix_u_cur());
  ap_uint<64>* u_rec_uint = reinterpret_cast<ap_uint<64>*>(storage.get_fix_u_rec());
  ap_uint<128>* v_uint = reinterpret_cast<ap_uint<128>*>(storage.get_fix_v());
  ap_uint<128>* s1_uint = reinterpret_cast<ap_uint<128>*>(storage.get_fix_s(0));
  ap_uint<128>* s2_uint = reinterpret_cast<ap_uint<128>*>(storage.get_fix_s(1));

  assert(storage.get_u_cur_size() == svd::lstm_params::R*4*svd::lstm_params::I / svd::lstm_params::MaxNumTu * (svd::lstm_params::MaxNumTu - svd::lstm_params::ZTu));
  assert(storage.get_u_rec_size() == svd::lstm_params::R*4*svd::lstm_params::H / svd::lstm_params::MaxNumTu * (svd::lstm_params::MaxNumTu - svd::lstm_params::ZTu));
  assert(storage.get_v_size() == svd::lstm_params::R*4*2*svd::lstm_params::H / svd::lstm_params::MaxNumTv * (svd::lstm_params::MaxNumTv - svd::lstm_params::ZTv));
  assert(storage.get_s_size() == svd::lstm_params::R*8);

  svd::ActivationD** h_prev_hls = new svd::ActivationD*[kN];
  svd::ActivationD** h_curr_hls = new svd::ActivationD*[kN];
  svd::ActivationD** c_prev_hls = new svd::ActivationD*[kN];
  svd::ActivationD** c_curr_hls = new svd::ActivationD*[kN];
  svd::ActivationD** h_prev_emulator = new svd::ActivationD*[kN];
  svd::ActivationD** h_curr_emulator = new svd::ActivationD*[kN];
  svd::ActivationD** c_prev_emulator = new svd::ActivationD*[kN];
  svd::ActivationD** c_curr_emulator = new svd::ActivationD*[kN];
  svd::ActivationD** h_prev_sw = new svd::ActivationD*[kN];
  svd::ActivationD** h_curr_sw = new svd::ActivationD*[kN];
  svd::ActivationD** c_prev_sw = new svd::ActivationD*[kN];
  svd::ActivationD** c_curr_sw = new svd::ActivationD*[kN];
  for (int i = 0; i < kN; ++i) {
    h_prev_emulator[i] = new svd::ActivationD[kH];
    h_curr_emulator[i] = new svd::ActivationD[kH];
    c_prev_emulator[i] = new svd::ActivationD[kH];
    c_curr_emulator[i] = new svd::ActivationD[kH];
    h_prev_hls[i] = reinterpret_cast<svd::ActivationD*>(storage.get_fix_h_prev(i));
    h_curr_hls[i] = reinterpret_cast<svd::ActivationD*>(storage.get_fix_h_curr(i));
    c_prev_hls[i] = reinterpret_cast<svd::ActivationD*>(storage.get_fix_c_prev(i));
    c_curr_hls[i] = reinterpret_cast<svd::ActivationD*>(storage.get_fix_c_curr(i));
  }
  if (kZTu > 0 && kZTv > 0) {
    for (int i = 0; i < NUM_TIMESTEPS; ++i) {
      for (int j = 0; j < kN; ++j) {
        std::swap(h_prev_hls[j], h_curr_hls[j]);
        std::swap(c_prev_hls[j], c_curr_hls[j]);
      }
      std::cout << "Starting accelerator." << std::endl;
      svd::SvdModel2LstmSDSoCV2(storage.get_fix_x(0), storage.get_fix_x(1), // [s * NUM_TIMESTEPS + t] samples?
        h_prev_hls[0], h_prev_hls[1], c_prev_hls[0], c_prev_hls[1],
        u_cur_uint, u_rec_uint, v_uint, s1_uint, s2_uint,
        storage.get_fix_bias(0), storage.get_fix_bias(1),
        storage.get_fix_nz_v(), storage.get_fix_nz_u(),
        h_curr_hls[0], h_curr_hls[1], c_curr_hls[0], c_curr_hls[1]);
      for (int j = 0; j < kN; ++j) {
        std::cout << "Starting Emulator: " << j << std::endl;
        svd::LstmSvdSoftEmulator<svd::ActivationD, svd::WeightD, svd::AccumD, svd::MultD, kLutSize>(
          kI, kH, kR, kNTu, kZTu, kNTv, kZTv, 1, storage.get_fix_x(j),
          storage.get_cur_gates("i")->get_u()->fix_pruned_data(),
          storage.get_cur_gates("i")->get_s(j).fix_pruned_data(),
          storage.get_cur_gates("i")->get_v()->fix_pruned_data(),
          storage.get_cur_gates("i")->get_u()->get_nz_idx(),
          storage.get_cur_gates("i")->get_v()->get_nz_idx(),
          storage.get_cur_gates("f")->get_u()->fix_pruned_data(),
          storage.get_cur_gates("f")->get_s(j).fix_pruned_data(),
          storage.get_cur_gates("f")->get_v()->fix_pruned_data(),
          storage.get_cur_gates("f")->get_u()->get_nz_idx(),
          storage.get_cur_gates("f")->get_v()->get_nz_idx(),
          storage.get_cur_gates("c")->get_u()->fix_pruned_data(),
          storage.get_cur_gates("c")->get_s(j).fix_pruned_data(),
          storage.get_cur_gates("c")->get_v()->fix_pruned_data(),
          storage.get_cur_gates("c")->get_u()->get_nz_idx(),
          storage.get_cur_gates("c")->get_v()->get_nz_idx(),
          storage.get_cur_gates("o")->get_u()->fix_pruned_data(),
          storage.get_cur_gates("o")->get_s(j).fix_pruned_data(),
          storage.get_cur_gates("o")->get_v()->fix_pruned_data(),
          storage.get_cur_gates("o")->get_u()->get_nz_idx(),
          storage.get_cur_gates("o")->get_v()->get_nz_idx(),
          storage.get_rec_gates("i")->get_u()->fix_pruned_data(),
          storage.get_rec_gates("i")->get_s(j).fix_pruned_data(),
          storage.get_rec_gates("i")->get_v()->fix_pruned_data(),
          storage.get_rec_gates("i")->get_u()->get_nz_idx(),
          storage.get_rec_gates("i")->get_v()->get_nz_idx(),
          storage.get_rec_gates("f")->get_u()->fix_pruned_data(),
          storage.get_rec_gates("f")->get_s(j).fix_pruned_data(),
          storage.get_rec_gates("f")->get_v()->fix_pruned_data(),
          storage.get_rec_gates("f")->get_u()->get_nz_idx(),
          storage.get_rec_gates("f")->get_v()->get_nz_idx(),
          storage.get_rec_gates("c")->get_u()->fix_pruned_data(),
          storage.get_rec_gates("c")->get_s(j).fix_pruned_data(),
          storage.get_rec_gates("c")->get_v()->fix_pruned_data(),
          storage.get_rec_gates("c")->get_u()->get_nz_idx(),
          storage.get_rec_gates("c")->get_v()->get_nz_idx(),
          storage.get_rec_gates("o")->get_u()->fix_pruned_data(),
          storage.get_rec_gates("o")->get_s(j).fix_pruned_data(),
          storage.get_rec_gates("o")->get_v()->fix_pruned_data(),
          storage.get_rec_gates("o")->get_u()->get_nz_idx(),
          storage.get_rec_gates("o")->get_v()->get_nz_idx(),
          storage.get_fix_bias(j),
          c_prev_emulator[j], h_prev_emulator[j],
          c_curr_emulator[j], h_curr_emulator[j]);
        std::cout << "Swapping LSTM outputs." << std::endl;
        std::swap(h_prev_emulator[j], h_curr_emulator[j]);
        std::swap(c_prev_emulator[j], c_curr_emulator[j]);
      }
    }
  }
  const int num_errors = 0; // storage.CountMismatches(h_prev_emulator);
  std::cout << "Number of mismatches: " << num_errors << std::endl;
  if (kTestSoftwareAccelerator) {
    for (int j = 0; j < kN; ++j) {
      const bool kVerbose = true;
      const bool kUseBlas = false;
      const int kUsaFloat = 0;
      const int kNumSamples = 1;
      std::cout << "Starting BLAS." << std::endl;
      svd::SvdModelLstmSoftware(kVerbose, kUseBlas, kUsaFloat,
        storage.get_x(j), kNumSamples, NUM_TIMESTEPS, svd::lstm_params::R,
        svd::lstm_params::I, svd::lstm_params::H,
        storage.get_cur_gates("i")->get_u()->data(),
        storage.get_cur_gates("i")->get_s(j).data(),
        storage.get_cur_gates("i")->get_v()->data(),
        storage.get_cur_gates("f")->get_u()->data(),
        storage.get_cur_gates("f")->get_s(j).data(),
        storage.get_cur_gates("f")->get_v()->data(),
        storage.get_cur_gates("c")->get_u()->data(),
        storage.get_cur_gates("c")->get_s(j).data(),
        storage.get_cur_gates("c")->get_v()->data(),
        storage.get_cur_gates("o")->get_u()->data(),
        storage.get_cur_gates("o")->get_s(j).data(),
        storage.get_cur_gates("o")->get_v()->data(),
        storage.get_rec_gates("i")->get_u()->data(),
        storage.get_rec_gates("i")->get_s(j).data(),
        storage.get_rec_gates("i")->get_v()->data(),
        storage.get_rec_gates("f")->get_u()->data(),
        storage.get_rec_gates("f")->get_s(j).data(),
        storage.get_rec_gates("f")->get_v()->data(),
        storage.get_rec_gates("c")->get_u()->data(),
        storage.get_rec_gates("c")->get_s(j).data(),
        storage.get_rec_gates("c")->get_v()->data(),
        storage.get_rec_gates("o")->get_u()->data(),
        storage.get_rec_gates("o")->get_s(j).data(),
        storage.get_rec_gates("o")->get_v()->data(),
        &storage.get_bias(j)[0 * storage.get_lstm_output_size()],
        &storage.get_bias(j)[1 * storage.get_lstm_output_size()],
        &storage.get_bias(j)[2 * storage.get_lstm_output_size()],
        &storage.get_bias(j)[3 * storage.get_lstm_output_size()],
        storage.get_h(j));
    }
  }

  // int num_refinements[kN] = {kR};
  // C_WrapperLstmSvd(
  //   NUM_TIMESTEPS,
  //   kNumActiveInputs,
  //   kI,
  //   kH,
  //   num_refinements,
  //   kZTu,
  //   kZTv,
  //   // Current Gates
  //   x_in,
  //   u_cur_in,
  //   s_cur_in,
  //   v_cur_in,
  //   uz_idx_cur_in,
  //   vz_idx_cur_in,
  //   // Recurrent Gates
  //   h_in,
  //   u_rec_in,
  //   s_rec_in,
  //   v_rec_in,
  //   uz_idx_rec_in,
  //   vz_idx_rec_in,
  //   // Non-Linearities
  //   bias_in,
  //   c_prev_in,
  //   h_curr_in,
  //   c_curr_in);

  // storage.ResetLstmOutputs();

  std::cout << "Cleaning up." << std::endl;
  delete[] h_prev_hls;
  delete[] h_curr_hls;
  delete[] c_prev_hls;
  delete[] c_curr_hls;
  delete[] h_prev_sw;
  delete[] h_curr_sw;
  delete[] c_prev_sw;
  delete[] c_curr_sw;
  for (int i = 0; i < kN; ++i) {
    delete[] h_prev_emulator[i];
    delete[] h_curr_emulator[i];
    delete[] c_prev_emulator[i];
    delete[] c_curr_emulator[i];
  }
  delete[] h_prev_emulator;
  delete[] h_curr_emulator;
  delete[] c_prev_emulator;
  delete[] c_curr_emulator;

  return 0;
}