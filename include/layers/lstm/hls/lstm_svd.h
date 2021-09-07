#ifndef LSTM_HLS_LSTM_SVD_H_
#define LSTM_HLS_LSTM_SVD_H_

#include "svd_params.h"
#include "kernel/svd_kernel.h"
#include "math_utils/activation_functions.h"
#include "layers/dense/hls/dense_svd.h"

#include "ap_int.h"
#include "hls_stream.h"

namespace svd {

#ifdef SDS_DESIGN
// =============================================================================
// Ports using DMAs
// =============================================================================
#pragma SDS data copy(x1_port[0:INPUT_SIZE])
#pragma SDS data copy(x2_port[0:INPUT_SIZE])
#pragma SDS data copy(h_t1_prev_port[0:HIDDEN_SIZE])
#pragma SDS data copy(h_t2_prev_port[0:HIDDEN_SIZE])
#pragma SDS data copy(c_t1_prev_port[0:HIDDEN_SIZE])
#pragma SDS data copy(c_t2_prev_port[0:HIDDEN_SIZE])
#pragma SDS data copy(bias1_port[0:4*HIDDEN_SIZE])
#pragma SDS data copy(bias2_port[0:4*HIDDEN_SIZE])
#pragma SDS data copy(nz_v_port[0:NUM_ITERATIONS * 8])
#pragma SDS data copy(nz_u_port[0:NUM_ITERATIONS * 8])
#pragma SDS data copy(h_t1_curr_port[0:HIDDEN_SIZE])
#pragma SDS data copy(h_t2_curr_port[0:HIDDEN_SIZE])
#pragma SDS data copy(c_t1_curr_port[0:HIDDEN_SIZE])
#pragma SDS data copy(c_t2_curr_port[0:HIDDEN_SIZE])
// Data Movers
#pragma SDS data data_mover(x1_port:AXIDMA_SIMPLE)
#pragma SDS data data_mover(x2_port:AXIDMA_SIMPLE)
#pragma SDS data data_mover(h_t1_prev_port:AXIDMA_SIMPLE)
#pragma SDS data data_mover(h_t2_prev_port:AXIDMA_SIMPLE)
#pragma SDS data data_mover(c_t1_prev_port:AXIDMA_SIMPLE)
#pragma SDS data data_mover(c_t2_prev_port:AXIDMA_SIMPLE)
#pragma SDS data data_mover(bias1_port:AXIDMA_SIMPLE)
#pragma SDS data data_mover(bias2_port:AXIDMA_SIMPLE)
#pragma SDS data data_mover(nz_v_port:AXIDMA_SIMPLE)
#pragma SDS data data_mover(nz_u_port:AXIDMA_SIMPLE)
#pragma SDS data data_mover(h_t1_curr_port:AXIDMA_SIMPLE)
#pragma SDS data data_mover(h_t2_curr_port:AXIDMA_SIMPLE)
#pragma SDS data data_mover(c_t1_curr_port:AXIDMA_SIMPLE)
#pragma SDS data data_mover(c_t2_curr_port:AXIDMA_SIMPLE)
// Port mapping
// #pragma SDS data sys_port(x1_port:ps_e_S_AXI_HPC0_FPD) // Coherent HP port
// #pragma SDS data sys_port(x2_port:ps_e_S_AXI_HPC0_FPD) // Coherent HP port
// #pragma SDS data sys_port(h_t1_prev_port:ps_e_S_AXI_HPC0_FPD) // Coherent HP port
// #pragma SDS data sys_port(h_t2_prev_port:ps_e_S_AXI_HPC0_FPD) // Coherent HP port
// #pragma SDS data sys_port(c_t1_prev_port:ps_e_S_AXI_HPC0_FPD) // Coherent HP port
// #pragma SDS data sys_port(c_t2_prev_port:ps_e_S_AXI_HPC0_FPD) // Coherent HP port
// #pragma SDS data sys_port(bias1_port:ps_e_S_AXI_HPC0_FPD) // Coherent HP port
// #pragma SDS data sys_port(bias2_port:ps_e_S_AXI_HPC0_FPD) // Coherent HP port
// #pragma SDS data sys_port(nz_v_port:ps_e_S_AXI_HPC0_FPD) // Coherent HP port
// #pragma SDS data sys_port(nz_u_port:ps_e_S_AXI_HPC0_FPD) // Coherent HP port
// #pragma SDS data sys_port(h_t1_curr_port:ps_e_S_AXI_HPC1_FPD) // Coherent HP port
// #pragma SDS data sys_port(h_t2_curr_port:ps_e_S_AXI_HPC1_FPD) // Coherent HP port
// #pragma SDS data sys_port(c_t1_curr_port:ps_e_S_AXI_HPC1_FPD) // Coherent HP port
// #pragma SDS data sys_port(c_t2_curr_port:ps_e_S_AXI_HPC1_FPD) // Coherent HP port
// =============================================================================
// Weight ports not using DMAs
// =============================================================================
// #pragma SDS data zero_copy(u_cur_port[0:NUM_ITERATIONS*INPUT_SIZE / NUM_TILES_U * (NUM_TILES_U - NUM_ZERO_TILES_U)])
// #pragma SDS data zero_copy(u_rec_port[0:NUM_ITERATIONS*HIDDEN_SIZE / NUM_TILES_U * (NUM_TILES_U - NUM_ZERO_TILES_U)])
// #pragma SDS data zero_copy(v_port[0:NUM_ITERATIONS*HIDDEN_SIZE / NUM_TILES_V * (NUM_TILES_V - NUM_ZERO_TILES_V)])
// #pragma SDS data zero_copy(s1_port[0:NUM_ITERATIONS])
// #pragma SDS data zero_copy(s2_port[0:NUM_ITERATIONS])
// =============================================================================
// Weight ports using DMAs
// =============================================================================
#pragma SDS data copy(u_cur_port[0:NUM_ITERATIONS*INPUT_SIZE / NUM_TILES_U * (NUM_TILES_U - NUM_ZERO_TILES_U)])
#pragma SDS data copy(u_rec_port[0:NUM_ITERATIONS*HIDDEN_SIZE / NUM_TILES_U * (NUM_TILES_U - NUM_ZERO_TILES_U)])
#pragma SDS data copy(v_port[0:NUM_ITERATIONS*HIDDEN_SIZE / NUM_TILES_V * (NUM_TILES_V - NUM_ZERO_TILES_V)])
#pragma SDS data copy(s1_port[0:NUM_ITERATIONS])
#pragma SDS data copy(s2_port[0:NUM_ITERATIONS])
// Platform Port Mapping, available options:
// - ACP Coherent ports: ps_e_S_AXI_HPC[0-1]_FPD
// - HP ports: ps_e_S_AXI_HP[0-3]_FPD
// #pragma SDS data sys_port(u_cur_port:ps_e_S_AXI_HP0_FPD) // HP2
// #pragma SDS data sys_port(u_rec_port:ps_e_S_AXI_HP1_FPD) // HP3
// #pragma SDS data sys_port(v_port:ps_e_S_AXI_HP2_FPD) // HP3
// #pragma SDS data sys_port(s1_port:ps_e_S_AXI_HP3_FPD) // HP3
// #pragma SDS data sys_port(s2_port:ps_e_S_AXI_HP3_FPD) // HP3
// =============================================================================
// Other Configurations
// =============================================================================
// Compiler hint on allocation
#pragma SDS data mem_attribute(x1_port:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(x2_port:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(h_t1_prev_port:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(h_t2_prev_port:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(c_t1_prev_port:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(c_t2_prev_port:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(u_cur_port:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(u_rec_port:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(v_port:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(s1_port:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(s2_port:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(bias1_port:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(bias2_port:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(nz_v_port:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(nz_u_port:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(h_t1_curr_port:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(h_t2_curr_port:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(c_t1_curr_port:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(c_t2_curr_port:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
// NOTE: All ports are accessed sequentially.
#pragma SDS data access_pattern(x1_port:SEQUENTIAL)
#pragma SDS data access_pattern(x2_port:SEQUENTIAL)
#pragma SDS data access_pattern(h_t1_prev_port:SEQUENTIAL)
#pragma SDS data access_pattern(h_t2_prev_port:SEQUENTIAL)
#pragma SDS data access_pattern(c_t1_prev_port:SEQUENTIAL)
#pragma SDS data access_pattern(c_t2_prev_port:SEQUENTIAL)
#pragma SDS data access_pattern(u_cur_port:SEQUENTIAL)
#pragma SDS data access_pattern(u_rec_port:SEQUENTIAL)
#pragma SDS data access_pattern(v_port:SEQUENTIAL)
#pragma SDS data access_pattern(s1_port:SEQUENTIAL)
#pragma SDS data access_pattern(s2_port:SEQUENTIAL)
#pragma SDS data access_pattern(bias1_port:SEQUENTIAL)
#pragma SDS data access_pattern(bias2_port:SEQUENTIAL)
#pragma SDS data access_pattern(nz_v_port:SEQUENTIAL)
#pragma SDS data access_pattern(nz_u_port:SEQUENTIAL)
#pragma SDS data access_pattern(h_t1_curr_port:SEQUENTIAL)
#pragma SDS data access_pattern(h_t2_curr_port:SEQUENTIAL)
#pragma SDS data access_pattern(c_t1_curr_port:SEQUENTIAL)
#pragma SDS data access_pattern(c_t2_curr_port:SEQUENTIAL)
#endif // end SDS_DESIGN
void SvdModel2LstmSDSoCV2(
    const svd::ActivationD x1_port[INPUT_SIZE],
    const svd::ActivationD x2_port[INPUT_SIZE],
    const svd::ActivationD h_t1_prev_port[HIDDEN_SIZE],
    const svd::ActivationD h_t2_prev_port[HIDDEN_SIZE],
    const svd::ActivationD c_t1_prev_port[HIDDEN_SIZE],
    const svd::ActivationD c_t2_prev_port[HIDDEN_SIZE],
    const ap_uint<FIX_WIDTH * 4> *u_cur_port, // [NUM_ITERATIONS*4*INPUT_SIZE / NUM_TILES_U * (NUM_TILES_U - NUM_ZERO_TILES_U)],
    const ap_uint<FIX_WIDTH * 4> *u_rec_port, // [NUM_ITERATIONS*4*HIDDEN_SIZE / NUM_TILES_U * (NUM_TILES_U - NUM_ZERO_TILES_U)],
    const ap_uint<FIX_WIDTH * 8> *v_port, // [NUM_ITERATIONS*4*2*HIDDEN_SIZE / NUM_TILES_V * (NUM_TILES_V - NUM_ZERO_TILES_V)],
    const ap_uint<FIX_WIDTH * 8> *s1_port, // [NUM_ITERATIONS*8],
    const ap_uint<FIX_WIDTH * 8> *s2_port, // [NUM_ITERATIONS*8],
    const svd::WeightD bias1_port[4 * HIDDEN_SIZE],
    const svd::WeightD bias2_port[4 * HIDDEN_SIZE],
    const ap_uint<NUM_TILES_V> nz_v_port[NUM_ITERATIONS * 8],
    const ap_uint<NUM_TILES_U> nz_u_port[NUM_ITERATIONS * 8],
    svd::ActivationD h_t1_curr_port[HIDDEN_SIZE],
    svd::ActivationD h_t2_curr_port[HIDDEN_SIZE],
    svd::ActivationD c_t1_curr_port[HIDDEN_SIZE],
    svd::ActivationD c_t2_curr_port[HIDDEN_SIZE]);

typedef svd::SvdParameters<NUM_INPUTS, INPUT_SIZE, HIDDEN_SIZE, NUM_ITERATIONS,
    NUM_TILES_U, NUM_TILES_V, NUM_ZERO_TILES_U, NUM_ZERO_TILES_V, NUM_GATES,
    ap_fixed<FIX_WIDTH, FIX_FRACT_WIDTH>,
    ap_fixed<FIX_WIDTH, FIX_FRACT_WIDTH>,
    ap_fixed<FIX_WIDTH, FIX_FRACT_WIDTH> > lstm_params;

#ifndef __VITIS_HLS__
#else
template <typename params>
void LstmSvdKernel(const int num_active_inputs,
    const int input_size,
    const int output_size,
    const int num_refinements[params::N],
    // Current Gates
    hls::stream<typename params::VectTuAxiPacketType>& x_port,
    hls::stream<typename params::VectTuAxiPacketType>& u_cur_port,
    hls::stream<typename params::VectG_AxiPacketType>& s_cur_port,
    hls::stream<typename params::VectTvAxiPacketType>& v_cur_port,
    // Recurrent Gates
    hls::stream<typename params::VectTuAxiPacketType>& h_prev_port,
    hls::stream<typename params::VectTuAxiPacketType>& u_rec_port,
    hls::stream<typename params::VectG_AxiPacketType>& s_rec_port,
    hls::stream<typename params::VectTvAxiPacketType>& v_rec_port,
    // Non-Linearities
    hls::stream<typename params::VectGTvAxiPacketType>& bias_port,
    hls::stream<typename params::VectTvAxiPacketType>& c_prev_port,
    hls::stream<typename params::VectTvAxiPacketType>& h_curr_port,
    hls::stream<typename params::VectTvAxiPacketType>& c_curr_port) {
#pragma HLS TOP name=LstmSvdKernel
// #pragma HLS INLINE
#pragma HLS DATAFLOW
#pragma HLS STABLE variable=s_cur_port
#pragma HLS STABLE variable=s_rec_port
#pragma HLS STABLE variable=v_cur_port
#pragma HLS STABLE variable=v_rec_port
#pragma HLS STABLE variable=bias_port
#pragma HLS STABLE variable=c_prev_port
#pragma HLS ARRAY_PARTITION variable=num_refinements complete
  int refinements[2][params::N];
#pragma HLS ARRAY_PARTITION variable=refinements complete dim=0
  for (int i = 0; i < 2; ++i) {
#pragma HLS UNROLL region
    for (int j = 0; j < params::N; ++j) {
      refinements[i][j] = num_refinements[j];
    }
  }
  typedef typename params::ActivationD ActivationType;
  typedef svd::AxiStreamFifo<params::VectGTvAxiWidth> WrapperFifoGTv;
  hls::stream<typename WrapperFifoGTv::PacketType> y_cur_fifo;
  hls::stream<typename WrapperFifoGTv::PacketType> y_rec_fifo;
#pragma HLS STREAM variable=y_cur_fifo depth=2
#pragma HLS STREAM variable=y_rec_fifo depth=2
  auto y_cur_axis = svd::AxiStreamFifo<params::VectGTvAxiWidth>(y_cur_fifo);
  auto y_rec_axis = svd::AxiStreamFifo<params::VectGTvAxiWidth>(y_rec_fifo);
  auto bias_axis = svd::AxiStreamPort<params::VectGTvAxiWidth>(bias_port);
  auto c_prev_axis = svd::AxiStreamPort<params::VectTvAxiWidth>(c_prev_port);
  auto c_curr_axis = svd::AxiStreamPort<params::VectTvAxiWidth>(c_curr_port);
  auto h_curr_axis = svd::AxiStreamPort<params::VectTvAxiWidth>(h_curr_port);
  // Current Gates
  svd::SvdKernel<params, WrapperFifoGTv>(num_active_inputs, input_size,
    output_size, refinements[0], x_port, u_cur_port, s_cur_port,
    v_cur_port, y_cur_fifo);
  // Recurrent Gates
  svd::SvdKernel<params, WrapperFifoGTv>(num_active_inputs, output_size,
    output_size, refinements[1], h_prev_port, u_rec_port, s_rec_port,
    v_rec_port, y_rec_fifo);
  // Non-Linearities
  const int kTypeBitwidth = hlsutils::Bitwidth<ActivationType>::value;
  const int kLutSize = (kTypeBitwidth > 16) ? 256 : 512;
  const bool kApplyBias = true;
  NonLinearities:
  for (int i = 0; i < output_size / params::Tv; ++i) {
    for (int k = 0; k < num_active_inputs; ++k) {
#pragma HLS PIPELINE II=1
      const int kGTv = params::G * params::Tv;
      auto y_cur = y_cur_axis.template PopVector<ActivationType, kGTv>();
      auto y_rec = y_rec_axis.template PopVector<ActivationType, kGTv>();
      auto bias = bias_axis.template PopVector<ActivationType, kGTv>();
      auto c_prev = c_prev_axis.template PopVector<ActivationType, params::Tv>();
      ActivationType c_curr[params::Tv];
      ActivationType h_curr[params::Tv];
#pragma HLS ARRAY_PARTITION variable=c_curr complete dim=0
#pragma HLS ARRAY_PARTITION variable=h_curr complete dim=0
      for (int j = 0; j < params::Tv; ++j) {
        svd::LstmNonLinearFunctions<ActivationType, ActivationType, kLutSize>(
          kApplyBias,
          y_cur[j * params::G + 0], y_cur[j * params::G + 1],
          y_cur[j * params::G + 2], y_cur[j * params::G + 3],
          y_rec[j * params::G + 0], y_rec[j * params::G + 1],
          y_rec[j * params::G + 2], y_rec[j * params::G + 3],
          bias[j * params::G + 0], bias[j * params::G + 1],
          bias[j * params::G + 2], bias[j * params::G + 3],
          c_prev[j], c_curr[j], h_curr[j]);
      }
      const bool kIsLast = i == output_size / params::Tv - 1 && k == num_active_inputs - 1;
      c_curr_axis.template PushBuffer<ActivationType>(params::Tv, c_curr, kIsLast);
      h_curr_axis.template PushBuffer<ActivationType>(params::Tv, h_curr, kIsLast);
    }
  }
}
#endif // end __VITIS_HLS__

/**
 * @brief      Sets the LstmSvd kernel inputs, i.e. streams from arrays into
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
void SetLstmSvdInputs(const int num_active_inputs,
    const int input_size,
    const int output_size,
    const int num_refinements[params::N],
    // Current Gates
    const typename params::ActivationD* x,
    const typename params::ActivationD* u_cur,
    const typename params::ActivationD* s_cur,
    const typename params::ActivationD* v_cur,
    // Recurrent Gates
    const typename params::ActivationD* h,
    const typename params::ActivationD* u_rec,
    const typename params::ActivationD* s_rec,
    const typename params::ActivationD* v_rec,
    // Non-Linearities
    const typename params::ActivationD* bias,
      // Current Gates
    hls::stream<typename params::VectTuAxiPacketType>& x_port,
    hls::stream<typename params::VectTuAxiPacketType>& u_cur_port,
    hls::stream<typename params::VectG_AxiPacketType>& s_cur_port,
    hls::stream<typename params::VectTvAxiPacketType>& v_cur_port,
    // Recurrent Gates
    hls::stream<typename params::VectTuAxiPacketType>& h_prev_port,
    hls::stream<typename params::VectTuAxiPacketType>& u_rec_port,
    hls::stream<typename params::VectG_AxiPacketType>& s_rec_port,
    hls::stream<typename params::VectTvAxiPacketType>& v_rec_port,
    // Non-Linearities
    hls::stream<typename params::VectGTvAxiPacketType>& bias_port,
    hls::stream<typename params::VectTvAxiPacketType>& c_prev_port
    ) {
  svd::SetDenseSvdInputs<params>(num_active_inputs, input_size, output_size,
    num_refinements, x, u_cur, s_cur, v_cur, bias, x_port, u_cur_port,
    s_cur_port, v_cur_port, bias_port);
  svd::SetSvdKernelInputs<params>(num_active_inputs, output_size, output_size,
    num_refinements, h, u_rec, s_rec, v_rec, h_prev_port, u_rec_port,
    s_rec_port, v_rec_port);

  // typedef typename params::ActivationD ActivationType;
  // const int kG = params::G; // NOTE: G is actually equal to 1.
  // const int kTu = params::Tu;
  // const int kTv = params::Tv;
  // const int kGTv = kG * kTv;
  // const int kNumTilesU = input_size / kTu;
  // const int kNumTilesV = output_size / kTv;
  // auto bias_axis = svd::AxiStreamPort<params::VectGTvAxiWidth>(bias_port);
  // typename params::VectGTvType bias_val;
  // for (int i = 0; i < kNumTilesV; ++i) {
  //   for (int j = 0; j < num_active_inputs; ++j) {
  //     for (int k = 0; k < kTv; ++k) {
  //       for (int ii = 0; ii < kG; ++ii) {
  //         int bias_idx = j * output_size * kG + ii * output_size + i * kTv + k;
  //         bias_val[k * kG + ii] = bias[bias_idx];
  //       }
  //     }
  //     bias_axis.template PushVector<ActivationType, kGTv>(bias_val);
  //   }
  // }
  // svd::SetSvdKernelInputs<params>(num_active_inputs, input_size,
  //   output_size, num_refinements, x, u, s, v, x_port, u_port, s_port, v_port);
}
#endif // end __VITIS_HLS__

#ifdef __VITIS_HLS__
template <typename params>
void GetLstmSvdOutputs(const int num_active_inputs,
    const int input_size,
    const int output_size,
    const int num_refinements[params::N],
    // Non-Linearities
    const typename params::ActivationD* h_curr,
    const typename params::ActivationD* c_curr,
    // Non-Linearities
    hls::stream<typename params::VectTvAxiPacketType>& h_curr_port,
    hls::stream<typename params::VectTvAxiPacketType>& c_curr_port
    ) {
  // svd::SetDenseSvdInputs<params>(num_active_inputs, input_size, output_size,
  //   num_refinements, x, u_cur, s_cur, v_cur, bias, x_port, u_cur_port,
  //   s_cur_port, v_cur_port, bias_port);
  // svd::SetSvdKernelInputs<params>(num_active_inputs, output_size, output_size,
  //   num_refinements, h, u_rec, s_rec, v_rec, h_prev_port, u_rec_port,
  //   s_rec_port, v_rec_port);

  // typedef typename params::ActivationD ActivationType;
  // const int kG = params::G; // NOTE: G is actually equal to 1.
  // const int kTu = params::Tu;
  // const int kTv = params::Tv;
  // const int kGTv = kG * kTv;
  // const int kNumTilesU = input_size / kTu;
  // const int kNumTilesV = output_size / kTv;
  // auto bias_axis = svd::AxiStreamPort<params::VectGTvAxiWidth>(bias_port);
  // typename params::VectGTvType bias_val;
  // for (int i = 0; i < kNumTilesV; ++i) {
  //   for (int j = 0; j < num_active_inputs; ++j) {
  //     for (int k = 0; k < kTv; ++k) {
  //       for (int ii = 0; ii < kG; ++ii) {
  //         int bias_idx = j * output_size * kG + ii * output_size + i * kTv + k;
  //         bias_val[k * kG + ii] = bias[bias_idx];
  //       }
  //     }
  //     bias_axis.template PushVector<ActivationType, kGTv>(bias_val);
  //   }
  // }
  // svd::SetSvdKernelInputs<params>(num_active_inputs, input_size,
  //   output_size, num_refinements, x, u, s, v, x_port, u_port, s_port, v_port);
}
#endif // end __VITIS_HLS__

} // svd

void HlsLstmSvd(const int num_active_inputs,
    const int input_size,
    const int output_size,
    const int num_refinements[svd::svd_params::N],
    // const hls::vector<int, svd::svd_params::N> num_refinements,
    // Current Gates
    hls::stream<typename svd::svd_params::VectTuAxiPacketType>& x_port,
    hls::stream<typename svd::svd_params::VectTuAxiPacketType>& u_cur_port,
    hls::stream<typename svd::svd_params::VectG_AxiPacketType>& s_cur_port,
    hls::stream<typename svd::svd_params::VectTvAxiPacketType>& v_cur_port,
    // Recurrent Gates
    hls::stream<typename svd::svd_params::VectTuAxiPacketType>& h_prev_port,
    hls::stream<typename svd::svd_params::VectTuAxiPacketType>& u_rec_port,
    hls::stream<typename svd::svd_params::VectG_AxiPacketType>& s_rec_port,
    hls::stream<typename svd::svd_params::VectTvAxiPacketType>& v_rec_port,
    // Non-Linearities
    hls::stream<typename svd::svd_params::VectGTvAxiPacketType>& bias_port,
    hls::stream<typename svd::svd_params::VectTvAxiPacketType>& c_prev_port,
    hls::stream<typename svd::svd_params::VectTvAxiPacketType>& h_curr_port,
    hls::stream<typename svd::svd_params::VectTvAxiPacketType>& c_curr_port);

#endif // end LSTM_HLS_LSTM_SVD_H_