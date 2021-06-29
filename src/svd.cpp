#include "svd_params.h"
#include "svd_ip.h"

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

  SvdIp2Inputs(x_port, u_port, s_port, v_port, nz_u_port, nz_v_port, y_port);

  return 0;
}