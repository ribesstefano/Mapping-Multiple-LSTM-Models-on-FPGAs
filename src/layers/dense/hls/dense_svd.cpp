#include "layers/dense/hls/dense_svd.h"

#ifndef __VITIS_HLS__
#else
void HlsDenseSvd(const int num_active_inputs,
    const int input_size,
    const int output_size,
    const int num_refinements[svd::dense_params::N],
    // const hls::vector<int, svd::dense_params::N> num_refinements,
    hls::stream<typename svd::dense_params::VectTuAxiPacketType>& x_port,
    hls::stream<typename svd::dense_params::VectTuAxiPacketType>& u_port,
    hls::stream<typename svd::dense_params::VectG_AxiPacketType>& s_port,
    hls::stream<typename svd::dense_params::VectTvAxiPacketType>& v_port,
    hls::stream<typename svd::dense_params::VectGTvAxiPacketType>& bias_port,
    hls::stream<typename svd::dense_params::VectTvAxiPacketType>& y_port) {
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl
#pragma HLS INTERFACE s_axilite port=num_active_inputs bundle=ctrl
#pragma HLS INTERFACE s_axilite port=input_size bundle=ctrl
#pragma HLS INTERFACE s_axilite port=output_size bundle=ctrl
#pragma HLS INTERFACE s_axilite port=num_refinements bundle=ctrl
#pragma HLS INTERFACE axis port=x_port
#pragma HLS INTERFACE axis port=u_port
#pragma HLS INTERFACE axis port=s_port
#pragma HLS INTERFACE axis port=v_port
#pragma HLS INTERFACE axis port=bias_port
#pragma HLS INTERFACE axis port=y_port
  svd::DenseSvdKernel<svd::dense_params>(num_active_inputs, input_size,
    output_size, num_refinements, x_port, u_port, s_port, v_port, bias_port,
    y_port);
}

/**
 * @brief      HLS Wrapper that calls a DenseSvd accelerator.
 *
 *             Useful in Cosimulation.
 *
 * @param[in]  num_active_inputs  The number of active inputs
 * @param[in]  input_size         The input size
 * @param[in]  output_size        The output size
 * @param[in]  num_refinements    The number of refinements
 * @param[in]  x                  The input array. Shape: (N, I)
 * @param[in]  u                  The u array. Shape: (R, I, G)
 * @param[in]  s                  The s array. Shape: (R, N, G)
 * @param[in]  v                  The v array. Shape: (R, H, G)
 * @param[in]  bias               The bias array. Shape: (N, G, H)
 * @param      y                  The y array. Shape: (N, G, H)
 */
void HlsWrapperDenseSvd(const int num_active_inputs,
    const int input_size,
    const int output_size,
    const int num_refinements[svd::dense_params::N],
    const typename svd::dense_params::ActivationD* x,
    const typename svd::dense_params::ActivationD* u,
    const typename svd::dense_params::ActivationD* s,
    const typename svd::dense_params::ActivationD* v,
    const typename svd::dense_params::ActivationD* bias,
    typename svd::dense_params::ActivationD* y) {
#ifdef __VITIS_HLS__
  hls::stream<typename svd::dense_params::VectTuAxiPacketType> x_port("x_port");
  hls::stream<typename svd::dense_params::VectTuAxiPacketType> u_port("u_port");
  hls::stream<typename svd::dense_params::VectG_AxiPacketType> s_port("s_port");
  hls::stream<typename svd::dense_params::VectTvAxiPacketType> v_port("v_port");
  hls::stream<typename svd::dense_params::VectGTvAxiPacketType> bias_port("bias_port");
  hls::stream<typename svd::dense_params::VectGTvAxiPacketType> y_port("y_port");
  svd::SetDenseSvdInputs<svd::dense_params>(num_active_inputs, input_size,
    output_size, num_refinements, x, u, s, v, bias, x_port, u_port, s_port,
    v_port, bias_port);
  HlsDenseSvd(num_active_inputs, input_size, output_size, num_refinements,
    x_port, u_port, s_port, v_port, bias_port, y_port);
  svd::GetSvdKernelOutputs<svd::dense_params>(num_active_inputs, output_size,
    y_port, y);
#endif // __VITIS_HLS__
}

#endif