#ifndef LAYERS_DENSE_HLS_DENSE_SVD_H_
#define LAYERS_DENSE_HLS_DENSE_SVD_H_

#include "svd_params.h"
#include "kernel/svd_kernel.h"

#include "ap_int.h"


namespace svd {

static const int kDenseNumGates = 1;

typedef svd::SvdParameters<NUM_INPUTS, INPUT_SIZE, HIDDEN_SIZE, NUM_ITERATIONS,
    NUM_TILES_U, NUM_TILES_V, NUM_ZERO_TILES_U, NUM_ZERO_TILES_V,
    kDenseNumGates, ap_fixed<FIX_WIDTH, FIX_FRACT_WIDTH>,
    ap_fixed<FIX_WIDTH, FIX_FRACT_WIDTH>,
    ap_fixed<FIX_WIDTH, FIX_FRACT_WIDTH> > dense_params;

#ifndef __VITIS_HLS__
#else
template <typename params>
void DenseSvdKernel(const int num_active_inputs,
    const int input_size,
    const int output_size,
    const int num_refinements[params::N],
    // const hls::vector<int, params::N> num_refinements,
    hls::stream<typename params::VectTuAxiPacketType>& x_port,
    hls::stream<typename params::VectTuAxiPacketType>& u_port,
    hls::stream<typename params::VectG_AxiPacketType>& s_port,
    hls::stream<typename params::VectTvAxiPacketType>& v_port,
    hls::stream<typename params::VectGTvAxiPacketType>& bias_port,
    hls::stream<typename params::VectTvAxiPacketType>& y_port) {
#pragma HLS TOP name=DenseSvdKernel
// #pragma HLS INLINE
#pragma HLS DATAFLOW
#pragma HLS STABLE variable=s_port
#pragma HLS STABLE variable=v_port
#pragma HLS STABLE variable=bias_port
  static_assert(params::G == 1, "DenseSvdKernel must have params::G equal to one.");
  assert(params::G == 1);
  typedef typename params::ActivationD ActivationType;
  typedef svd::AxiStreamFifo<params::VectGTvAxiWidth> WrapperFifoGTv;
  hls::stream<typename WrapperFifoGTv::PacketType> y_fifo;
#pragma HLS STREAM variable=y_fifo depth=2
  auto y_axis = svd::AxiStreamFifo<params::VectGTvAxiWidth>(y_fifo);
  auto y_out_axis = svd::AxiStreamPort<params::VectGTvAxiWidth>(y_port);
  auto bias_axis = svd::AxiStreamPort<params::VectGTvAxiWidth>(bias_port);
  svd::SvdKernel<params, WrapperFifoGTv>(num_active_inputs, input_size,
    output_size, num_refinements, x_port, u_port, s_port, v_port, y_fifo);
  Apply_Bias:
  for (int i = 0; i < output_size / params::Tv; ++i) {
    for (int j = 0; j < num_active_inputs; ++j) {
#pragma HLS PIPELINE II=1
      const int kGTv = params::G * params::Tv; // NOTE: G is actually equal to 1.
      const auto y_val = y_axis.template PopVector<ActivationType, kGTv>();
      const auto bias_val = bias_axis.template PopVector<ActivationType, kGTv>();
      const auto y_out = y_val + bias_val;
// #pragma HLS BIND_OP variable=y_out op=add impl=dsp latency=3
      y_out_axis.template PushVector<ActivationType, kGTv>(y_out);
    }
  }
}
#endif // end __VITIS_HLS__

/**
 * @brief      Sets the DenseSvd kernel inputs, i.e. streams from arrays into
 *             hls::streams.
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
 * @param      x_port             The x port to be used as argument to SvdKernel
 * @param      u_port             The u port to be used as argument to SvdKernel
 * @param      s_port             The s port to be used as argument to SvdKernel
 * @param      v_port             The v port to be used as argument to SvdKernel
 * @param      bias_port          The bias port to be used as argument to
 *                                SvdKernel
 *
 * @tparam     params             Collection of SVD configuration params.
 */
#ifdef __VITIS_HLS__
template <typename params>
void SetDenseSvdInputs(const int num_active_inputs,
    const int input_size,
    const int output_size,
    const int num_refinements[params::N],
    const typename params::ActivationD* x,
    const typename params::ActivationD* u,
    const typename params::ActivationD* s,
    const typename params::ActivationD* v,
    const typename params::ActivationD* bias,
    hls::stream<typename params::VectTuAxiPacketType>& x_port,
    hls::stream<typename params::VectTuAxiPacketType>& u_port,
    hls::stream<typename params::VectG_AxiPacketType>& s_port,
    hls::stream<typename params::VectTvAxiPacketType>& v_port,
    hls::stream<typename params::VectGTvAxiPacketType>& bias_port) {
  typedef typename params::ActivationD ActivationType;
  const int kG = params::G; // NOTE: G is actually equal to 1.
  const int kTu = params::Tu;
  const int kTv = params::Tv;
  const int kGTv = kG * kTv;
  const int kNumTilesU = input_size / kTu;
  const int kNumTilesV = output_size / kTv;
  auto bias_axis = svd::AxiStreamPort<params::VectGTvAxiWidth>(bias_port);
  typename params::VectGTvType bias_val;
  for (int i = 0; i < kNumTilesV; ++i) {
    for (int j = 0; j < num_active_inputs; ++j) {
      for (int k = 0; k < kTv; ++k) {
        for (int ii = 0; ii < kG; ++ii) {
          int bias_idx = j * output_size * kG + ii * output_size + i * kTv + k;
          bias_val[k * kG + ii] = bias[bias_idx];
        }
      }
      bias_axis.template PushVector<ActivationType, kGTv>(bias_val);
    }
  }
  svd::SetSvdKernelInputs<params>(num_active_inputs, input_size,
    output_size, num_refinements, x, u, s, v, x_port, u_port, s_port, v_port);
}
#endif // __VITIS_HLS__

} // svd

void HlsDenseSvd(const int num_active_inputs,
  const int input_size,
  const int output_size,
  const int num_refinements[svd::dense_params::N],
  hls::stream<typename svd::dense_params::VectTuAxiPacketType>& x_port,
  hls::stream<typename svd::dense_params::VectTuAxiPacketType>& u_port,
  hls::stream<typename svd::dense_params::VectG_AxiPacketType>& s_port,
  hls::stream<typename svd::dense_params::VectTvAxiPacketType>& v_port,
  hls::stream<typename svd::dense_params::VectGTvAxiPacketType>& bias_port,
  hls::stream<typename svd::dense_params::VectTvAxiPacketType>& y_port);


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
  typename svd::dense_params::ActivationD* y);

#endif // end DENSE_HLS_DENSE_SVD_H_);
