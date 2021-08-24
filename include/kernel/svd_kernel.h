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

template <typename params>
void SvdKernel(const int num_active_inputs,
    const int input_size,
    const int output_size,
    const hls::vector<int, params::N> num_refinements,
    hls::stream<typename params::VectTuAxiPacketType>& x_port,
    hls::stream<typename params::VectTuAxiPacketType>& u_port,
    hls::stream<typename params::VectG_AxiPacketType>& s_port,
    hls::stream<typename params::VectTvAxiPacketType>& v_port,
    hls::stream<typename params::VectGTvAxiPacketType>& y_port) {
#pragma HLS DATAFLOW
#pragma HLS INLINE
  const bool pad_output = false;
  hls::stream<typename params::VectG_AxiPacketType> xu_port("xu_port");
  hls::stream<typename params::VectG_AxiPacketType> xus_port("xus_port");
#pragma HLS STREAM variable=xu_port depth=2
#pragma HLS STREAM variable=xus_port depth=2
  svd::KernelU<params>(num_active_inputs, input_size, num_refinements,
    pad_output, x_port, u_port, xu_port);
  svd::KernelS<params>(num_active_inputs, num_refinements, xu_port,
    s_port, xus_port);
  svd::KernelV<params>(num_active_inputs, output_size,
    num_refinements, xus_port, v_port, y_port);
}

} // svd

#endif // end KERNEL_SVD_KERNEL_H_