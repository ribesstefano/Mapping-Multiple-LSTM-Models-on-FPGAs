#include "svd_params.h"
#include "svd_ip.h"
#include "lstm/lstm_data_handler.h"
#include "lstm/sw/soft_lstm_svd.h"
#include "lstm/hls/lstm_svd.h"
#include "lstm/hls/lstm_svd_emulator.h"

#include "ap_fixed.h"

#include <iostream>
#include <cstdlib>

int main(int argc, char const *argv[]) {
  std::cout << "Hello SVD!" << std::endl;

  const bool kTestSoftwareAccelerator = false;
  const int kNumInputs = 2;
  const int kRefinementSteps = NUM_ITERATIONS;
  const int kLstmInputSize = INPUT_SIZE;
  const int kLstmOutputSize = HIDDEN_SIZE;
  const int kUCurSize = kLstmInputSize;
  const int kURecSize = kLstmOutputSize;
  const int kVSize = kLstmOutputSize;
  const int kNumTilesU = NUM_TILES_U;
  const int kNumZeroTilesU = NUM_ZERO_TILES_U;
  const int kNumTilesV = NUM_TILES_V;
  const int kNumZeroTilesV = NUM_ZERO_TILES_V;
  const int kLutSize = (FIX_WIDTH == 16) ? 512 : 256;

  std::cout << "Setting AcceleratorBlob." << std::endl;
  typedef svd::AcceleratorBlob<float, svd::ActivationD, kNumTilesU, kNumTilesV> AcceleratorStorage;
  AcceleratorStorage storage = AcceleratorStorage(kNumInputs, kRefinementSteps, kUCurSize,
    kURecSize, kVSize, kNumTilesU, kNumZeroTilesU, kNumTilesV, kNumZeroTilesV);

  std::cout << "Running SvdIp2Inputs." << std::endl;
  typename svd::svd_params::ActivationD x_port[svd::svd_params::N][svd::svd_params::I] = {rand()};
  typename svd::svd_params::UPortD u_port[svd::svd_params::R * svd::svd_params::PrunedSizeU] = {rand()};
  typename svd::svd_params::SPortD s_port[svd::svd_params::N][svd::svd_params::R] = {rand()};
  typename svd::svd_params::VPortD v_port[svd::svd_params::R * svd::svd_params::PrunedSizeV] = {rand()};
  ap_uint<svd::svd_params::Tu> nz_u_port[svd::svd_params::G * svd::svd_params::R] = {rand()};
  ap_uint<svd::svd_params::Tv> nz_v_port[svd::svd_params::G * svd::svd_params::R] = {rand()};
  typename svd::svd_params::ActivationD y_port[svd::svd_params::N][svd::svd_params::G][svd::svd_params::H] = {rand()};
  // SvdIp2Inputs(x_port, u_port, s_port, v_port, nz_u_port, nz_v_port, y_port);

  std::cout << "reinterpret_cast." << std::endl;

  ap_uint<64>* u_cur_uint = reinterpret_cast<ap_uint<64>*>(storage.get_fix_u_cur());
  ap_uint<64>* u_rec_uint = reinterpret_cast<ap_uint<64>*>(storage.get_fix_u_rec());
  ap_uint<128>* v_uint = reinterpret_cast<ap_uint<128>*>(storage.get_fix_v());
  ap_uint<128>* s1_uint = reinterpret_cast<ap_uint<128>*>(storage.get_fix_s(0));
  ap_uint<128>* s2_uint = reinterpret_cast<ap_uint<128>*>(storage.get_fix_s(1));

  assert(storage.get_u_cur_size() == NUM_ITERATIONS*4*INPUT_SIZE / NUM_TILES_U * (NUM_TILES_U - NUM_ZERO_TILES_U));
  assert(storage.get_u_rec_size() == NUM_ITERATIONS*4*HIDDEN_SIZE / NUM_TILES_U * (NUM_TILES_U - NUM_ZERO_TILES_U));
  assert(storage.get_v_size() == NUM_ITERATIONS*4*2*HIDDEN_SIZE / NUM_TILES_V * (NUM_TILES_V - NUM_ZERO_TILES_V));
  assert(storage.get_s_size() == NUM_ITERATIONS*8);

  svd::ActivationD** h_prev_hls = new svd::ActivationD*[kNumInputs];
  svd::ActivationD** h_curr_hls = new svd::ActivationD*[kNumInputs];
  svd::ActivationD** c_prev_hls = new svd::ActivationD*[kNumInputs];
  svd::ActivationD** c_curr_hls = new svd::ActivationD*[kNumInputs];
  svd::ActivationD** h_prev_emulator = new svd::ActivationD*[kNumInputs];
  svd::ActivationD** h_curr_emulator = new svd::ActivationD*[kNumInputs];
  svd::ActivationD** c_prev_emulator = new svd::ActivationD*[kNumInputs];
  svd::ActivationD** c_curr_emulator = new svd::ActivationD*[kNumInputs];
  svd::ActivationD** h_prev_sw = new svd::ActivationD*[kNumInputs];
  svd::ActivationD** h_curr_sw = new svd::ActivationD*[kNumInputs];
  svd::ActivationD** c_prev_sw = new svd::ActivationD*[kNumInputs];
  svd::ActivationD** c_curr_sw = new svd::ActivationD*[kNumInputs];
  for (int i = 0; i < kNumInputs; ++i) {
    h_prev_emulator[i] = new svd::ActivationD[kLstmOutputSize];
    h_curr_emulator[i] = new svd::ActivationD[kLstmOutputSize];
    c_prev_emulator[i] = new svd::ActivationD[kLstmOutputSize];
    c_curr_emulator[i] = new svd::ActivationD[kLstmOutputSize];
    h_prev_hls[i] = reinterpret_cast<svd::ActivationD*>(storage.get_fix_h_prev(i));
    h_curr_hls[i] = reinterpret_cast<svd::ActivationD*>(storage.get_fix_h_curr(i));
    c_prev_hls[i] = reinterpret_cast<svd::ActivationD*>(storage.get_fix_c_prev(i));
    c_curr_hls[i] = reinterpret_cast<svd::ActivationD*>(storage.get_fix_c_curr(i));
  }
  for (int i = 0; i < NUM_TIMESTEPS; ++i) {
    for (int j = 0; j < kNumInputs; ++j) {
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
    for (int j = 0; j < kNumInputs; ++j) {
      std::cout << "Starting Emulator: " << j << std::endl;
      svd::LstmSvdSoftEmulator<svd::ActivationD, svd::WeightD, svd::AccumD, svd::MultD, kLutSize>(
        kLstmInputSize, kLstmOutputSize, kRefinementSteps,
        kNumTilesU, kNumZeroTilesU,
        kNumTilesV, kNumZeroTilesV, 1, storage.get_fix_x(j),
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
  const int num_errors = storage.CountMismatches(h_prev_emulator);
  std::cout << "Number of mismatches: " << num_errors << std::endl;
  if (kTestSoftwareAccelerator) {
    for (int j = 0; j < kNumInputs; ++j) {
      const bool kVerbose = true;
      const bool kUseBlas = false;
      const int kUsaFloat = 0;
      const int kNumSamples = 1;
      std::cout << "Starting BLAS." << std::endl;
      svd::SvdModelLstmSoftware(kVerbose, kUseBlas, kUsaFloat,
        storage.get_x(j), kNumSamples, NUM_TIMESTEPS, NUM_ITERATIONS,
        INPUT_SIZE, HIDDEN_SIZE,
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
  storage.ResetLstmOutputs();
  std::cout << "Cleaning up." << std::endl;
  delete[] h_prev_hls;
  delete[] h_curr_hls;
  delete[] c_prev_hls;
  delete[] c_curr_hls;
  delete[] h_prev_sw;
  delete[] h_curr_sw;
  delete[] c_prev_sw;
  delete[] c_curr_sw;
  for (int i = 0; i < kNumInputs; ++i) {
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