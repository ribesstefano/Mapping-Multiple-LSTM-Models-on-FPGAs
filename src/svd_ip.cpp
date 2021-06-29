#include "svd_ip.h"

void SvdIp2Inputs(
    const typename svd_params::ActivationD x_port[svd_params::N][svd_params::I],
    const typename svd_params::UPortD u_port[svd_params::PrunedSizeU],
    const typename svd_params::SPortD s_port[svd_params::N][svd_params::R],
    const typename svd_params::VPortD v_port[svd_params::PrunedSizeV],
    const ap_uint<svd_params::Tu> nz_u_port[svd_params::N],
    const ap_uint<svd_params::Tv> nz_v_port[svd_params::N],
    typename svd_params::ActivationD y_port[svd_params::N][svd_params::G][svd_params::H]) {
  SvdIP<svd_params>(x_port, u_port, s_port, v_port, nz_u_port, nz_v_port, y_port);
}