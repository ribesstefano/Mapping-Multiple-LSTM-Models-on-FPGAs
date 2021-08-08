#include "kernel/s_kernel.h"
#include "dma/axis_lib.h"

#include "hls_stream.h"
#ifdef __VITIS_HLS__
#include "hls_vector.h"
#endif

#ifndef __VITIS_HLS__
#else
void HlsKernelS(const hls::vector<int, tests::params::N> num_refinements,
  hls::stream<typename tests::params::VectG_AxiType>& xu_port,
  hls::stream<typename tests::params::VectG_AxiType>& s_port,
  hls::stream<typename tests::params::VectG_AxiType>& xus_port) {
#if 0
  svd::GenericKernelS<tests::params>(num_refinements, xu_port, s_port, xus_port);
#else
#pragma HLS INTERFACE axis port=xu_port
#pragma HLS INTERFACE axis port=s_port
#pragma HLS INTERFACE axis port=xus_port
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE s_axilite port=num_refinements
#pragma HLS DATAFLOW
  int R_max = num_refinements[tests::params::N - 1];
  int R_total = num_refinements[0] * tests::params::N; // Total elements.
  Get_Total_R:
  for (int i = 1; i < tests::params::N; ++i) {
#pragma HLS PIPELINE II=1
    if (num_refinements[i] > R_max) {
      R_max = num_refinements[i];
    }
    R_total += (num_refinements[i] - num_refinements[i - 1]) * (tests::params::N - i);
  }
  auto xu_axis = svd::AxiStreamInterface<tests::params::VectG_AxiWidth>(xu_port);
  auto s_axis = svd::AxiStreamInterface<tests::params::VectG_AxiWidth>(s_port);
  auto xus_axis = svd::AxiStreamInterface<tests::params::VectG_AxiWidth>(xus_port);
  S_Kernel:
  for (int i = 0; i < R_total; ++i) {
#pragma HLS PIPELINE II=1
    typedef typename tests::params::ActivationD ActivationType;
    auto xu_val = xu_axis.PopVector<ActivationType, tests::params::G>();
    auto s_val = s_axis.PopVector<ActivationType, tests::params::G>();
    auto xus_val = xu_val * s_val;
    const bool kIsLast = (i == R_total - 1) ? true : false;
    xus_axis.PushVector<ActivationType, tests::params::G>(xus_val, kIsLast);
  }
#endif
}

#endif

namespace svd {

} // svd