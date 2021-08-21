#include "kernel/svd_kernel.h"

void HlsSvdKernel(const int num_active_inputs,
    const int input_size,
    const int output_size,
    const hls::vector<int, svd::svd_params::N> num_refinements,
    hls::stream<typename svd::svd_params::VectTuAxiType>& x_port,
    hls::stream<typename svd::svd_params::VectTuAxiType>& u_port,
    hls::stream<typename svd::svd_params::VectG_AxiType>& s_port,
    hls::stream<typename svd::svd_params::VectTvAxiType>& v_port,
    hls::stream<typename svd::svd_params::VectGTvAxiType>& y_port) {
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
  const bool pad_output = false;
  hls::stream<typename svd::svd_params::VectG_AxiType> xu_port("xu_port");
  hls::stream<typename svd::svd_params::VectG_AxiType> xus_port("xus_port");
#pragma HLS STREAM variable=xu_port depth=2
#pragma HLS STREAM variable=xus_port depth=2
  svd::KernelU<svd::svd_params>(num_active_inputs, input_size, num_refinements,
    pad_output, x_port, u_port, xu_port);
  svd::KernelS<svd::svd_params>(num_active_inputs, num_refinements, xu_port,
    s_port, xus_port);
  svd::KernelV<svd::svd_params>(num_active_inputs, output_size,
    num_refinements, xus_port, v_port, y_port);
}

void HlsSvdKernelFixed(
    hls::stream<typename svd::svd_params::VectTuAxiType>& x_port,
    hls::stream<typename svd::svd_params::VectTuAxiType>& u_port,
    hls::stream<typename svd::svd_params::VectG_AxiType>& s_port,
    hls::stream<typename svd::svd_params::VectTvAxiType>& v_port,
    hls::stream<typename svd::svd_params::VectGTvAxiType>& y_port) {
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
  const hls::vector<int, svd::svd_params::N> kNumRefinements = svd::svd_params::R;
  const bool pad_output = false;
  hls::stream<typename svd::svd_params::VectG_AxiType> xu_port("xu_port");
  hls::stream<typename svd::svd_params::VectG_AxiType> xus_port("xus_port");
#pragma HLS STREAM variable=xu_port depth=2
#pragma HLS STREAM variable=xus_port depth=2
  svd::KernelU<svd::svd_params>(kNumActiveInputs, kInputSize, kNumRefinements,
    pad_output, x_port, u_port, xu_port);
  svd::KernelS<svd::svd_params>(kNumActiveInputs, kNumRefinements, xu_port,
    s_port, xus_port);
  svd::KernelV<svd::svd_params>(kNumActiveInputs, kOutputSize,
    kNumRefinements, xus_port, v_port, y_port);
}