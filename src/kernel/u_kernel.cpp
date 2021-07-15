#include "kernel/u_kernel.h"

#include "hls_stream.h"

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

void HlsKernelU(const int num_refinements,
  typename u_params::ActivationS x[u_params::N][u_params::G][u_params::PeU],
  typename u_params::WeightS u[u_params::G][u_params::PeU],
  typename u_params::AccumulationS xu[u_params::N][u_params::G][u_params::PeU]) {
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl
#pragma HLS INTERFACE s_axilite port=num_refinements bundle=ctrl
#pragma HLS INTERFACE axis port=x
#pragma HLS INTERFACE axis port=u
#pragma HLS INTERFACE axis port=xu
#pragma HLS ARRAY_PARTITION variable=x complete dim=0
#pragma HLS ARRAY_PARTITION variable=u complete dim=0
#pragma HLS ARRAY_PARTITION variable=xu complete dim=0
#pragma HLS DATAFLOW
  svd::SvdStreams<u_params> streams;
  U_DMA:
  for (int i = 0; i < u_params::G; ++i) {
    for (int j = 0; j < u_params::PeU; ++j) {
      for (int k = 0; k < num_refinements; ++k) {
#pragma HLS PIPELINE II=1
        streams.u[i][j].write(u[i][j].read());
      }
    }
  }
  X_DMA:
  for (int i = 0; i < u_params::G; ++i) {
    for (int j = 0; j < u_params::PeU; ++j) {
      for (int k = 0; k < u_params::N; ++k) {
        for (int ii = 0; ii < num_refinements; ++ii) {
#pragma HLS PIPELINE II=1
          streams.x[k][i][j].write(x[k][i][j].read());
        }
      }
    }
  }
  svd::KernelU<u_params>(num_refinements, streams);
  XU_DMA:
  for (int i = 0; i < u_params::G; ++i) {
    for (int j = 0; j < u_params::PeU; ++j) {
      for (int k = 0; k < u_params::N; ++k) {
        for (int ii = 0; ii < num_refinements; ++ii) {
#pragma HLS PIPELINE II=1
          xu[k][i][j].write(streams.xu[k][i][j].read());
        }
      }
    }
  }
}

namespace svd {

} // svd