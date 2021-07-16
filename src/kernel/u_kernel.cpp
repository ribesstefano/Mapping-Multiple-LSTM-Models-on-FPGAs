#include "kernel/u_kernel.h"
#include "hls_utils/adder_tree.h"
#include "dma/svd_dma.h"

#include "hls_stream.h"
#ifdef __VITIS_HLS__
#include <hls_vector.h>
#endif

static const int kNumInputs = 2;
static const int kInputSize = 256;
static const int Tu = 8;
// NOTE: The rest of the parameters are unused for now.
static const int kDummySize = 1;
static const int kDummyRefinements = 1;
static const int Tv = 1;
static const int ZTu = 0;
static const int ZTv = 0;
static const int G = 1;

typedef svd::SvdParameters<kNumInputs, kInputSize, kDummySize, kDummyRefinements,
    Tu, Tv, ZTu, ZTv, G, svd::ActivationD, svd::WeightD, svd::AccumD> u_params;

#ifndef __VITIS_HLS__
void HlsKernelU(const int num_refinements,
  const typename u_params::ActivationD x_port[u_params::N][u_params::I],
  const typename u_params::UPortD u_port[u_params::R * u_params::PrunedSizeU],
  typename u_params::ActivationD xu_port[u_params::N][u_params::G * u_params::R]) {
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl
#pragma HLS INTERFACE s_axilite port=num_refinements bundle=ctrl
#pragma HLS INTERFACE m_axi port=x_port offset=slave depth=u_params::I
#pragma HLS INTERFACE m_axi port=u_port offset=slave depth=u_params::R*u_params::PrunedSizeU
#pragma HLS INTERFACE m_axi port=xu_port offset=slave depth=u_params::R
#pragma HLS DATAFLOW
  svd::SvdStreams<u_params> streams;
  svd::SvdBuffers<u_params> buffers;
  svd::InputDMA<u_params>(num_refinements, x_port, streams, buffers);
  svd::StreamSplitter(num_refinements * u_params::G * u_params::PrunedSizeU, u_port, streams.u_dma);
  U_Dispatcher:
  for (int i = 0; i < num_refinements; ++i) {
    for (int j = 0; j < u_params::PeU; ++j) {
      for (int k = 0; k < u_params::PrunedSizeU / u_params::PeU; ++k) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN
        for (int g = 0; g < u_params::G; ++g) {
          streams.u[g][j].write(streams.u_dma[g].read());
        }
      }
    }
  }
  svd::KernelU<u_params>(num_refinements, streams);
  for (int i = 0; i < num_refinements; ++i) {
    for (int j = 0; j < u_params::N; ++j) {
#pragma HLS PIPELINE II=1
      for (int k = 0; k < u_params::G; ++k) {
        auto tmp = hlsutils::adder_tree<u_params::AccumulationD, u_params::PeU>(streams.xu[j][k]);
        xu_port[j][k * num_refinements + i] = tmp;
      }
    }
  }
}
#else
void HlsKernelU(const int num_refinements,
  const hls::vector<typename u_params::ActivationD, u_params::N> x_port[u_params::I],
  const typename u_params::UPortD u_port[u_params::R * u_params::PrunedSizeU],
  hls::vector<typename u_params::ActivationD, u_params::N> xu_port[u_params::G * u_params::R]) {
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl
#pragma HLS INTERFACE s_axilite port=num_refinements bundle=ctrl
#pragma HLS INTERFACE m_axi port=x_port offset=slave depth=u_params::I
#pragma HLS INTERFACE m_axi port=u_port offset=slave depth=u_params::R*u_params::PrunedSizeU
#pragma HLS INTERFACE m_axi port=xu_port offset=slave depth=u_params::R
#pragma HLS DATAFLOW
  svd::SvdStreams<u_params> streams;
  svd::SvdBuffers<u_params> buffers;

  typename u_params::ActivationD x_tmp[u_params::N][u_params::I];
#pragma HLS ARRAY_PARTITION variable=x_tmp complete dim=1
#pragma HLS STREAM variable=x_tmp depth=2

  X_DMA:
  for (int i = 0; i < u_params::I; ++i) {
#pragma HLS PIPELINE II=1
    for (int j = 0; j < u_params::N; ++j) {
      x_tmp[j][i] = x_port[i][j];
    }
  }
  svd::InputDMA<u_params>(num_refinements, x_tmp, streams, buffers);
  svd::StreamSplitter(num_refinements * u_params::G * u_params::PrunedSizeU, u_port, streams.u_dma);
  U_Dispatcher:
  for (int i = 0; i < num_refinements; ++i) {
    for (int j = 0; j < u_params::PeU; ++j) {
      for (int k = 0; k < u_params::PrunedSizeU / u_params::PeU; ++k) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN
        for (int g = 0; g < u_params::G; ++g) {
          streams.u[g][j].write(streams.u_dma[g].read());
        }
      }
    }
  }
  svd::KernelU<u_params>(num_refinements, streams);
  for (int i = 0; i < num_refinements; ++i) {
    for (int j = 0; j < u_params::G; ++j) {
#pragma HLS PIPELINE II=1
      for (int k = 0; k < u_params::N; ++k) {
        auto tmp = hlsutils::adder_tree<u_params::AccumulationD, u_params::PeU>(streams.xu[k][j]);
        xu_port[j * num_refinements + i][k] = tmp;
      }
    }
  }
}
#endif

namespace svd {

} // svd