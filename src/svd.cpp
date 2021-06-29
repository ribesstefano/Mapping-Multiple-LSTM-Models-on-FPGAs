#include "svd_params.h"
#include "svd_ip.h"
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
  const int kRefinementSteps = 2;
  const int kUCurSize = 1024;
  const int kURecSize = 512;
  const int kVSize = 512;
  const int kNumTilesU = 16;
  const int kNumZeroTilesU = 8;
  const int kNumTilesV = 32;
  const int kNumZeroTilesV = 4;


  std::cout << "Setting AcceleratorBlob." << std::endl;
  typedef lstm::AcceleratorBlob<float, ap_fixed<16, 9>, kNumTilesU, kNumTilesV> AccelDataType;
  AccelDataType storage = AccelDataType(kNumInputs, kRefinementSteps, kUCurSize,
    kURecSize, kVSize, kNumTilesU, kNumZeroTilesU, kNumTilesV, kNumZeroTilesV);
  std::cout << "Cleaning up." << std::endl;

  return 0;
}