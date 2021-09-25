#ifndef SVD_IP_H_
#define SVD_IP_H_

#include "svd_params.h"
#include "kernel/svd_kernel.h"

namespace svd {

// template <typename params>
// inline void SvdIP(
//     const typename params::ActivationD x_port[params::N][params::I],
//     const typename params::UPortD u_port[params::R * params::PrunedSizeU],
//     const typename params::SPortD s_port[params::N][params::R],
//     const typename params::VPortD v_port[params::R * params::PrunedSizeV],
//     const typename params::UnzD nz_u_port[params::G * params::R],
//     const typename params::VnzD nz_v_port[params::G * params::R],
//     typename params::ActivationD y_port[params::N][params::G][params::H]) {
// #pragma HLS INLINE
// #pragma HLS DATAFLOW
//   assert(params::I % params::Tu == 0);
//   assert(params::H % params::Tv == 0);
//   assert(params::Tu - params::ZTu > 0);
//   assert(params::Tv - params::ZTv > 0);
//   svd::SvdStreams<params> streams;
//   svd::SvdBuffers<params> buffers;
//   SvdInDMA<params>(x_port, u_port, s_port, v_port, nz_u_port, nz_v_port, streams, buffers);
//   svd::SvdKernel<params>(streams);
//   SvdOutDMA<params>(streams, y_port);
// }

// void SvdIp2Inputs(
//     const typename svd_params::ActivationD x_port[svd_params::N][svd_params::I],
//     const typename svd_params::UPortD u_port[svd_params::R * svd_params::PrunedSizeU],
//     const typename svd_params::SPortD s_port[svd_params::N][svd_params::R],
//     const typename svd_params::VPortD v_port[svd_params::R * svd_params::PrunedSizeV],
//     const ap_uint<svd_params::Tu> nz_u_port[svd_params::G * svd_params::R],
//     const ap_uint<svd_params::Tv> nz_v_port[svd_params::G * svd_params::R],
//     typename svd_params::ActivationD y_port[svd_params::N][svd_params::G][svd_params::H]);

} // svd

#endif // end SVD_IP_H_