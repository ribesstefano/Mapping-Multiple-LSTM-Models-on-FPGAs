#include "svd_params.h"
#include "svd_ip.h"
#include "lstm/hls/lstm_svd.h"
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

  std::cout << "printing stuff..." << std::endl;
  std::cout << storage.get_fix_x(0) << std::endl;
  std::cout << storage.get_fix_x(0)[234] << std::endl;

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

  std::cout << (storage.get_u_cur_size() == NUM_ITERATIONS*4*INPUT_SIZE / NUM_TILES_U * (NUM_TILES_U - NUM_ZERO_TILES_U) ? "true" : "false") << std::endl;
  std::cout << (storage.get_u_rec_size() == NUM_ITERATIONS*4*HIDDEN_SIZE / NUM_TILES_U * (NUM_TILES_U - NUM_ZERO_TILES_U) ? "true" : "false") << std::endl;
  std::cout << (storage.get_v_size() == NUM_ITERATIONS*4*2*HIDDEN_SIZE / NUM_TILES_V * (NUM_TILES_V - NUM_ZERO_TILES_V) ? "true" : "false") << std::endl;
  std::cout << (storage.get_s_size() == NUM_ITERATIONS*8 ? "true" : "false") << std::endl;

  SvdModel2LstmSDSoCV2(
    storage.get_fix_x(0),
    storage.get_fix_x(1),
    storage.get_fix_h(0),
    storage.get_fix_h(1),
    storage.get_fix_c(0),
    storage.get_fix_c(1),
    u_cur_uint,
    u_rec_uint,
    v_uint,
    s1_uint,
    s2_uint,
    storage.get_fix_bias(0),
    storage.get_fix_bias(1),
    storage.get_fix_nz_v(),
    storage.get_fix_nz_u(),
    storage.get_fix_h(0),
    storage.get_fix_h(1),
    storage.get_fix_c(0),
    storage.get_fix_c(1));

  std::cout << "Cleaning up." << std::endl;

  return 0;
}