#include "kernel/u_kernel.h"
#include "kernel/gemv_kernel.h"
#include "hls_utils/adder_tree.h"
#include "dma/svd_dma.h"
#include "dma/axis_lib.h"

#include "assert.h"
#include "ap_axi_sdata.h"
#include "hls_stream.h"
#ifdef __VITIS_HLS__
#include "hls_vector.h"
#endif

#ifndef __VITIS_HLS__
/**
 * @brief      Synthesizeable Kernel-U.
 * @deprecated Compile time parametrization only.
 *
 * @param[in]  num_refinements  The number refinements
 * @param[in]  x_port           The x port
 * @param[in]  u_port           The u port
 * @param      xu_port          The xu port
 */
void HlsKernelU(const int num_refinements,
  const typename testu::params::ActivationD x_port[testu::params::N][testu::params::I],
  const typename testu::params::UPortD u_port[testu::params::R * testu::params::PrunedSizeU],
  typename testu::params::ActivationD xu_port[testu::params::N][testu::params::G * testu::params::R]) {
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl
#pragma HLS INTERFACE s_axilite port=num_refinements bundle=ctrl
#pragma HLS INTERFACE m_axi port=x_port offset=slave depth=testu::params::I
#pragma HLS INTERFACE m_axi port=u_port offset=slave depth=testu::params::R*testu::params::PrunedSizeU
#pragma HLS INTERFACE m_axi port=xu_port offset=slave depth=testu::params::R
#pragma HLS DATAFLOW
  svd::SvdStreams<testu::params> streams;
  svd::SvdBuffers<testu::params> buffers;
  svd::InputDMA<testu::params>(num_refinements, x_port, streams, buffers);
  svd::StreamSplitter(num_refinements * testu::params::G * testu::params::PrunedSizeU, u_port, streams.u_dma);
  U_Dispatcher:
  for (int i = 0; i < num_refinements; ++i) {
    for (int j = 0; j < testu::params::PeU; ++j) {
      for (int k = 0; k < testu::params::PrunedSizeU / testu::params::PeU; ++k) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN
        for (int g = 0; g < testu::params::G; ++g) {
          streams.u[g][j].write(streams.u_dma[g].read());
        }
      }
    }
  }
  svd::KernelU<testu::params>(num_refinements, streams);
  for (int i = 0; i < num_refinements; ++i) {
    for (int j = 0; j < testu::params::N; ++j) {
#pragma HLS PIPELINE II=1
      for (int k = 0; k < testu::params::G; ++k) {
        auto tmp = hlsutils::adder_tree<testu::params::AccumulationD, testu::params::PeU>(streams.xu[j][k]);
        xu_port[j][k * num_refinements + i] = tmp;
      }
    }
  }
}
#else
/**
 * @brief      Synthesizeable flexible Kernel-U.
 *
 * @param[in]  num_active_inputs  The number of active inputs
 * @param[in]  input_size         The input size
 * @param[in]  num_refinements    The number of refinements steps (R) per input:
 *                                the Rs must be positive, greater than zero and
 *                                in ASCENDING ORDER. Their amount must be less
 *                                or equal to num_active_inputs.
 * @param[in]  pad_output         Wether to pad output with zeroes
 * @param      x_port             The input x port
 * @param      u_port             The input u port
 * @param      xu_port            The output xu port
 */
void HlsKernelU(const int num_active_inputs,
    const int input_size,
    const int num_refinements[testu::params::N],
    const bool pad_output,
    hls::stream<typename testu::params::VectTuAxiPacketType>& x_port,
    hls::stream<typename testu::params::VectTuAxiPacketType>& u_port,
    hls::stream<typename testu::params::VectG_AxiPacketType>& xu_port) {
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE s_axilite port=num_active_inputs
#pragma HLS INTERFACE s_axilite port=input_size
#pragma HLS INTERFACE s_axilite port=num_refinements
#pragma HLS INTERFACE s_axilite port=pad_output
#pragma HLS INTERFACE axis port=x_port
#pragma HLS INTERFACE axis port=u_port
#pragma HLS INTERFACE axis port=xu_port
#pragma HLS ARRAY_PARTITION variable=num_refinements complete dim=1  
  svd::KernelU<testu::params>(num_active_inputs, input_size, num_refinements,
    pad_output, x_port, u_port, xu_port);
}

void HlsKernelU_Pruned(const int num_active_inputs,
    const int input_size,
    const int num_refinements[testu::params::N],
    const int num_zero_tiles_u,
    hls::stream<typename testu::params::VectGZTuAxiPacketType>& unz_idx_port,
    hls::stream<typename testu::params::VectTuAxiPacketType>& x_port,
    hls::stream<typename testu::params::VectTuAxiPacketType>& u_port,
    hls::stream<typename testu::params::VectG_AxiPacketType>& xu_port) {
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE s_axilite port=num_active_inputs
#pragma HLS INTERFACE s_axilite port=input_size
#pragma HLS INTERFACE s_axilite port=num_refinements
#pragma HLS INTERFACE s_axilite port=num_zero_tiles_u
#pragma HLS INTERFACE axis port=unz_idx_port
#pragma HLS INTERFACE axis port=x_port
#pragma HLS INTERFACE axis port=u_port
#pragma HLS INTERFACE axis port=xu_port
#pragma HLS ARRAY_PARTITION variable=num_refinements complete dim=1  
  svd::KernelU_Pruned<testu::params>(num_active_inputs, input_size,
    num_refinements, num_zero_tiles_u, unz_idx_port, x_port, u_port, xu_port);
}

#endif // __VITIS_HLS__

namespace svd {

} // svd