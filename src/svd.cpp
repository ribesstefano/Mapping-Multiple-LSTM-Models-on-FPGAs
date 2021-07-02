#include "svd_params.h"
#include "svd_ip.h"
#include "lstm/hls/lstm_svd.h"
#include "lstm/sw/soft_lstm_svd.h"
#include "lstm/lstm_data_handler.h"

#include "ap_fixed.h"

#include <iostream>
#include <cstdlib>

int main(int argc, char const *argv[]) {
  std::cout << "Hello SVD!" << std::endl;

  typename svd_params::ActivationD x_port[svd_params::N][svd_params::I] = {rand()};
  typename svd_params::UPortD u_port[svd_params::PrunedSizeU] = {rand()};
  typename svd_params::SPortD s_port[svd_params::N][svd_params::R] = {rand()};
  typename svd_params::VPortD v_port[svd_params::PrunedSizeV] = {rand()};
  ap_uint<svd_params::Tu> nz_u_port[svd_params::N] = {rand()};
  ap_uint<svd_params::Tv> nz_v_port[svd_params::N] = {rand()};
  typename svd_params::ActivationD y_port[svd_params::N][svd_params::G][svd_params::H] = {rand()};

  std::cout << "Running SvdIp2Inputs." << std::endl;
  SvdIp2Inputs(x_port, u_port, s_port, v_port, nz_u_port, nz_v_port, y_port);

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

  std::cout << "Setting AcceleratorBlob." << std::endl;
  typedef lstm::AcceleratorBlob<float, svd::ActivationD, kNumTilesU, kNumTilesV> AccelDataType;
  AccelDataType storage = AccelDataType(kNumInputs, kRefinementSteps, kUCurSize,
    kURecSize, kVSize, kNumTilesU, kNumZeroTilesU, kNumTilesV, kNumZeroTilesV);

  std::cout << "reinterpret_cast." << std::endl;

  ap_uint<64>* u_cur_uint = reinterpret_cast<ap_uint<64>*>(storage.get_fix_u_cur());
  ap_uint<64>* u_rec_uint = reinterpret_cast<ap_uint<64>*>(storage.get_fix_u_rec());
  ap_uint<128>* v_uint = reinterpret_cast<ap_uint<128>*>(storage.get_fix_v());
  ap_uint<128>* s1_uint = reinterpret_cast<ap_uint<128>*>(storage.get_fix_s(0));
  ap_uint<128>* s2_uint = reinterpret_cast<ap_uint<128>*>(storage.get_fix_s(1));

  std::cout << "Starting accelerator." << std::endl;

  assert(storage.get_u_cur_size() == NUM_ITERATIONS*4*INPUT_SIZE / NUM_TILES_U * (NUM_TILES_U - NUM_ZERO_TILES_U));
  assert(storage.get_u_rec_size() == NUM_ITERATIONS*4*HIDDEN_SIZE / NUM_TILES_U * (NUM_TILES_U - NUM_ZERO_TILES_U));
  assert(storage.get_v_size() == NUM_ITERATIONS*4*2*HIDDEN_SIZE / NUM_TILES_V * (NUM_TILES_V - NUM_ZERO_TILES_V));
  assert(storage.get_s_size() == NUM_ITERATIONS*8);

  svd::ActivationD* h1_prev;
  svd::ActivationD* h2_prev;
  svd::ActivationD* h1_curr;
  svd::ActivationD* h2_curr;
  svd::ActivationD* c1_prev;
  svd::ActivationD* c2_prev;
  svd::ActivationD* c1_curr;
  svd::ActivationD* c2_curr;

  for (int t = 0; t < NUM_TIMESTEPS; ++t) {
    if (t % 2 == 0) {
      h1_prev = reinterpret_cast<svd::ActivationD*>(storage.get_fix_h_curr(0));
      h2_prev = reinterpret_cast<svd::ActivationD*>(storage.get_fix_h_curr(1));
      h1_curr = reinterpret_cast<svd::ActivationD*>(storage.get_fix_h_prev(0));
      h2_curr = reinterpret_cast<svd::ActivationD*>(storage.get_fix_h_prev(1));
      c1_prev = reinterpret_cast<svd::ActivationD*>(storage.get_fix_c_curr(0));
      c2_prev = reinterpret_cast<svd::ActivationD*>(storage.get_fix_c_curr(1));
      c1_curr = reinterpret_cast<svd::ActivationD*>(storage.get_fix_c_prev(0));
      c2_curr = reinterpret_cast<svd::ActivationD*>(storage.get_fix_c_prev(1));
    } else {
      h1_prev = reinterpret_cast<svd::ActivationD*>(storage.get_fix_h_curr(0));
      h2_prev = reinterpret_cast<svd::ActivationD*>(storage.get_fix_h_curr(1));
      h1_curr = reinterpret_cast<svd::ActivationD*>(storage.get_fix_h_prev(0));
      h2_curr = reinterpret_cast<svd::ActivationD*>(storage.get_fix_h_prev(1));
      c1_prev = reinterpret_cast<svd::ActivationD*>(storage.get_fix_c_curr(0));
      c2_prev = reinterpret_cast<svd::ActivationD*>(storage.get_fix_c_curr(1));
      c1_curr = reinterpret_cast<svd::ActivationD*>(storage.get_fix_c_prev(0));
      c2_curr = reinterpret_cast<svd::ActivationD*>(storage.get_fix_c_prev(1));
    }
    SvdModel2LstmSDSoCV2(storage.get_fix_x(0), storage.get_fix_x(1), // [s * NUM_TIMESTEPS + t] samples?
      h1_curr, h2_curr, c1_curr, c2_curr,
      u_cur_uint, u_rec_uint, v_uint, s1_uint, s2_uint,
      storage.get_fix_bias(0), storage.get_fix_bias(1),
      storage.get_fix_nz_v(), storage.get_fix_nz_u(),
      h1_prev, h2_prev, c1_prev, c2_prev);
  }


  std::cout << "Starting software Model." << std::endl;

  const bool kVerbose = true;
  const bool kUseBlas = false;
  const int kUsaFloat = 0;
  const int kNumSamples = 1;
  SvdModel2LstmSoftware(kVerbose, kUseBlas, kUsaFloat,
                          storage.get_x(0),
                          kNumSamples,
                          NUM_TIMESTEPS,
                          NUM_ITERATIONS,
                          INPUT_SIZE,
                          HIDDEN_SIZE,
                          storage.get_cur_gates()["i"]->get_u()->data(),
                          storage.get_cur_gates()["i"]->get_s(0).data(),
                          storage.get_cur_gates()["i"]->get_v()->data(),
                          storage.get_cur_gates()["f"]->get_u()->data(),
                          storage.get_cur_gates()["f"]->get_s(0).data(),
                          storage.get_cur_gates()["f"]->get_v()->data(),
                          storage.get_cur_gates()["c"]->get_u()->data(),
                          storage.get_cur_gates()["c"]->get_s(0).data(),
                          storage.get_cur_gates()["c"]->get_v()->data(),
                          storage.get_cur_gates()["o"]->get_u()->data(),
                          storage.get_cur_gates()["o"]->get_s(0).data(),
                          storage.get_cur_gates()["o"]->get_v()->data(),
                          storage.get_rec_gates()["i"]->get_u()->data(),
                          storage.get_rec_gates()["i"]->get_s(0).data(),
                          storage.get_rec_gates()["i"]->get_v()->data(),
                          storage.get_rec_gates()["f"]->get_u()->data(),
                          storage.get_rec_gates()["f"]->get_s(0).data(),
                          storage.get_rec_gates()["f"]->get_v()->data(),
                          storage.get_rec_gates()["c"]->get_u()->data(),
                          storage.get_rec_gates()["c"]->get_s(0).data(),
                          storage.get_rec_gates()["c"]->get_v()->data(),
                          storage.get_rec_gates()["o"]->get_u()->data(),
                          storage.get_rec_gates()["o"]->get_s(0).data(),
                          storage.get_rec_gates()["o"]->get_v()->data(),
                          &storage.get_bias(0)[0 * HIDDEN_SIZE],
                          &storage.get_bias(0)[1 * HIDDEN_SIZE],
                          &storage.get_bias(0)[2 * HIDDEN_SIZE],
                          &storage.get_bias(0)[3 * HIDDEN_SIZE],
                          storage.get_h(0));

  storage.ResetLstmOutputs();
  std::cout << "Cleaning up." << std::endl;

  return 0;
}