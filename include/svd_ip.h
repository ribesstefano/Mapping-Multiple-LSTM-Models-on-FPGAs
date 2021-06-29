#ifndef SVD_IP_H_
#define SVD_IP_H_

#include "svd_params.h"
#include "kernel/svd_kernel.h"

template <typename params>
inline void SvdIP(
    const typename params::ActivationD x_port[params::N][params::I],
    const typename params::UPortD u_port[params::PrunedSizeU],
    const typename params::SPortD s_port[params::N][params::R],
    const typename params::VPortD v_port[params::PrunedSizeV],
    const typename params::UnzD nz_u_port[params::R * params::G],
    const typename params::VnzD nz_v_port[params::R * params::G],
    typename params::ActivationD y_port[params::N][params::G][params::H]) {
#pragma HLS INLINE
#pragma HLS DATAFLOW
  assert(params::I % params::Tu == 0);
  assert(params::H % params::Tv == 0);
  assert(params::Tu - params::ZTu > 0);
  assert(params::Tv - params::ZTv > 0);
  svd::SvdStreams<params> streams;
  svd::SvdBuffers<params> buffers;
  SvdInDMA<params>(x_port, u_port, s_port, v_port, nz_u_port, nz_v_port, streams, buffers);
  SvdKernel<params>(streams);
  SvdOutDMA<params>(streams, y_port);
}

const int N = 2;
const int I = 256;
const int H = 128;
const int R = 16;
const int Tu = 16;
const int Tv = 32;
const int ZTu = 8;
const int ZTv = 8;
const int G = 4;
typedef svd::SvdParameters<N, I, H, R, Tu, Tv, ZTu, ZTv, G> svd_params;

void SvdIp2Inputs(
    const typename svd_params::ActivationD x_port[svd_params::N][svd_params::I],
    const typename svd_params::UPortD u_port[svd_params::PrunedSizeU],
    const typename svd_params::SPortD s_port[svd_params::N][svd_params::R],
    const typename svd_params::VPortD v_port[svd_params::PrunedSizeV],
    const ap_uint<svd_params::Tu> nz_u_port[svd_params::N],
    const ap_uint<svd_params::Tv> nz_v_port[svd_params::N],
    typename svd_params::ActivationD y_port[svd_params::N][svd_params::G][svd_params::H]);

#endif // end SVD_IP_H_