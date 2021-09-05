#ifndef KERNEL_SVD_KERNEL_H_
#define KERNEL_SVD_KERNEL_H_

#include "svd_params.h"
#include "dma/svd_dma.h"
#include "kernel/u_kernel.h"
#include "kernel/s_kernel.h"
#include "kernel/v_kernel.h"

namespace svd {

template <typename params>
inline void SvdKernel(svd::SvdStreams<params> &streams) {
#pragma HLS INLINE
#ifndef __VITIS_HLS__
#pragma HLS DATAFLOW
#endif
  svd::KernelU<params>(params::R, streams);
  svd::KernelS<params>(params::R, streams);
  svd::KernelV<params>(params::R, streams);
}

template <
  typename params,
  typename WrapperAxisGTv = svd::AxiStreamPort<params::VectGTvAxiWidth>
>
void SvdKernel(const int num_active_inputs,
    const int input_size,
    const int output_size,
    const int num_refinements[params::N],
    // const hls::vector<int, params::N> num_refinements,
    hls::stream<typename params::VectTuAxiPacketType>& x_port,
    hls::stream<typename params::VectTuAxiPacketType>& u_port,
    hls::stream<typename params::VectG_AxiPacketType>& s_port,
    hls::stream<typename params::VectTvAxiPacketType>& v_port,
    hls::stream<typename WrapperAxisGTv::PacketType>& y_port) {
#pragma HLS TOP name=SvdKernel
#pragma HLS INLINE
#pragma HLS DATAFLOW
#pragma HLS STABLE variable=s_port
#pragma HLS ARRAY_PARTITION variable=num_refinements complete
// #pragma HLS SHARED variable=num_refinements
  const bool pad_output = false;
  typedef svd::AxiStreamFifo<params::VectG_AxiWidth> WrapperFifoG;
  hls::stream<typename WrapperFifoG::PacketType> xu_port("xu_port");
  hls::stream<typename WrapperFifoG::PacketType> xus_port("xus_port");
#pragma HLS STREAM variable=xu_port depth=2
#pragma HLS STREAM variable=xus_port depth=2
  int num_refinements_u[params::N];
  int num_refinements_s[params::N];
  int num_refinements_v[params::N];
#pragma HLS ARRAY_PARTITION variable=num_refinements_u complete
#pragma HLS ARRAY_PARTITION variable=num_refinements_s complete
#pragma HLS ARRAY_PARTITION variable=num_refinements_v complete
  Duplicate_R_Stream:
  for (int i = 0; i < params::N; ++i) {
#pragma HLS UNROLL
    num_refinements_u[i] = num_refinements[i];
    num_refinements_s[i] = num_refinements[i];
    num_refinements_v[i] = num_refinements[i];
  }
  svd::KernelU<params, WrapperFifoG>(num_active_inputs, input_size,
    num_refinements_u, pad_output, x_port, u_port, xu_port);
  svd::KernelS<params, WrapperFifoG>(num_active_inputs, num_refinements_s,
    xu_port, s_port, xus_port);
  svd::KernelV<params, WrapperFifoG, WrapperAxisGTv>(num_active_inputs,
    output_size, num_refinements_v, xus_port, v_port, y_port);
}

} // svd

#endif // end KERNEL_SVD_KERNEL_H_