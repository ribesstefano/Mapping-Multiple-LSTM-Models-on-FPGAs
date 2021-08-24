#include "kernel/v_kernel.h"

#include "hls_stream.h"
#ifdef __VITIS_HLS__
#include "hls_vector.h"
#endif

#include "assert.h"

#ifndef __VITIS_HLS__
#else
void HlsKernelV(const int num_active_inputs,
    const int output_size,
    const hls::vector<int, testv::params::N> num_refinements,
    hls::stream<typename testv::params::VectG_AxiPacketType>& xus_port,
    hls::stream<typename testv::params::VectTvAxiPacketType>& v_port,
    hls::stream<typename testv::params::VectGTvAxiPacketType>& y_port) {
#pragma HLS INTERFACE axis port=xus_port
#pragma HLS INTERFACE axis port=v_port
#pragma HLS INTERFACE axis port=y_port
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE s_axilite port=num_active_inputs
#pragma HLS INTERFACE s_axilite port=output_size
#pragma HLS INTERFACE s_axilite port=num_refinements
  svd::KernelV<testv::params>(num_active_inputs, output_size,
    num_refinements, xus_port, v_port, y_port);
}
#endif // end __VITIS_HLS__