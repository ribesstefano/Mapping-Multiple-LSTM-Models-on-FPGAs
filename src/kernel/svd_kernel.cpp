#include "kernel/svd_kernel.h"

void HlsSvdKernel(const int num_active_inputs,
    const int input_size,
    const int output_size,
    const int num_refinements[svd::svd_params::N],
    hls::stream<typename svd::svd_params::VectTuAxiPacketType>& x_port,
    hls::stream<typename svd::svd_params::VectTuAxiPacketType>& u_port,
    hls::stream<typename svd::svd_params::VectG_AxiPacketType>& s_port,
    hls::stream<typename svd::svd_params::VectTvAxiPacketType>& v_port,
    hls::stream<typename svd::svd_params::VectGTvAxiPacketType>& y_port) {
#pragma HLS INTERFACE axis port=x_port
#pragma HLS INTERFACE axis port=u_port
#pragma HLS INTERFACE axis port=s_port
#pragma HLS INTERFACE axis port=v_port
#pragma HLS INTERFACE axis port=y_port
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE s_axilite port=num_active_inputs
#pragma HLS INTERFACE s_axilite port=input_size
#pragma HLS INTERFACE s_axilite port=output_size
#pragma HLS INTERFACE s_axilite port=num_refinements
#pragma HLS DATAFLOW
  svd::SvdKernel<svd::svd_params>(num_active_inputs, input_size, output_size,
    num_refinements, x_port, u_port, s_port, v_port, y_port);
}

void HlsSvdKernelFixed(
    hls::stream<typename svd::svd_params::VectTuAxiPacketType>& x_port,
    hls::stream<typename svd::svd_params::VectTuAxiPacketType>& u_port,
    hls::stream<typename svd::svd_params::VectG_AxiPacketType>& s_port,
    hls::stream<typename svd::svd_params::VectTvAxiPacketType>& v_port,
    hls::stream<typename svd::svd_params::VectGTvAxiPacketType>& y_port) {
#pragma HLS INTERFACE axis port=x_port
#pragma HLS INTERFACE axis port=u_port
#pragma HLS INTERFACE axis port=s_port
#pragma HLS INTERFACE axis port=v_port
#pragma HLS INTERFACE axis port=y_port
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS DATAFLOW
  const int kNumActiveInputs = svd::svd_params::N;
  const int kInputSize = svd::svd_params::I;
  const int kOutputSize = svd::svd_params::H;
  const int kNumRefinements[svd::svd_params::N] = {svd::svd_params::R};
  svd::SvdKernel<svd::svd_params>(kNumActiveInputs, kInputSize, kOutputSize,
    kNumRefinements, x_port, u_port, s_port, v_port, y_port);
}