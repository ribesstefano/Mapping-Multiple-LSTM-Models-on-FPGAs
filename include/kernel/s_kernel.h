#ifndef KERNEL_S_KERNEL_H_
#define KERNEL_S_KERNEL_H_

#include "svd_params.h"
#include "hls_utils/adder_tree.h"
#include "dma/axis_lib.h"
#include "hls_utils/hls_metaprogramming.h"

#include "hls_stream.h"

namespace svd {

template <typename params>
void KernelS(const int num_refinements, svd::SvdStreams<params> &streams) {
  typedef typename params::AccumulationD accum_t;
  for (int i = 0; i < num_refinements; ++i) {
#pragma HLS PIPELINE II=1
    for (int j = 0; j < params::N; ++j) {
      for (int k = 0; k < params::G; ++k) {
        auto sum = hlsutils::adder_tree<accum_t, params::PeU>(streams.xu[j][k]);
        auto xs = sum * streams.s[j][k].read();
        for (int ii = 0; ii < params::PeV; ++ii) {
          streams.xus[j][k][ii].write(xs);
        }
      }
    }
  }
}

template<int Ni, int Gi = 1, typename ActivationD_tp = ap_fixed<16, 3> >
struct KernelS_Params {
  static const int N = Ni;
  static const int G = Gi;
  static const int ActivationWidth = hlsutils::Bitwidth<ActivationD>::value;
  static const int VectG_AxiWidth = ActivationWidth * G;
  typedef ActivationD_tp ActivationD;
  typedef typename svd::AxiStreamPort<VectG_AxiWidth>::AxiuPacketType VectG_AxiPacketType;
#ifdef __VITIS_HLS__
  typedef hls::vector<ActivationD, G> VectG_Type;
#endif
};

#ifndef __VITIS_HLS__
#else
template <
  typename params,
  typename PortWrapper = svd::AxiStreamPort<params::VectG_AxiWidth>
>
void KernelS(const int num_active_inputs,
    const int num_refinements[params::N],
    hls::stream<typename PortWrapper::PacketType>& xu_port,
    hls::stream<typename params::VectG_AxiPacketType>& s_port,
    hls::stream<typename PortWrapper::PacketType>& xus_port) {
#pragma HLS TOP name=KernelS
#pragma HLS DATAFLOW
#pragma HLS INLINE
#pragma HLS STABLE variable=xu_port
#pragma HLS STABLE variable=s_port
#pragma HLS STABLE variable=xus_port
  assert(num_active_inputs <= params::N);
  assert(num_active_inputs > 0);
  int R_max = num_refinements[0];
  int R_total = num_refinements[0] * num_active_inputs; // Total elements.
  Get_Total_R:
  for (int i = 1; i < num_active_inputs; ++i) {
#pragma HLS PIPELINE II=1
    if (num_refinements[i] > R_max) {
      R_max = num_refinements[i];
    }
    assert(num_refinements[i] >= num_refinements[i - 1]);
    R_total += (num_refinements[i] - num_refinements[i - 1]) * (num_active_inputs - i);
  }
  auto xu_axis = svd::AxiStreamInterface<PortWrapper>(xu_port);
  auto s_axis = svd::AxiStreamPort<params::VectG_AxiWidth>(s_port);
  auto xus_axis = svd::AxiStreamInterface<PortWrapper>(xus_port);
  S_Kernel:
  for (int i = 0; i < R_total; ++i) {
#pragma HLS PIPELINE II=1
    typedef typename params::ActivationD ActivationType;
    auto xu_val = xu_axis.template PopVector<ActivationType, params::G>();
    auto s_val = s_axis.template PopVector<ActivationType, params::G>();
    auto xus_val = xu_val * s_val;
    const bool kIsLast = i == R_total - 1;
    xus_axis.template PushVector<ActivationType, params::G>(xus_val, kIsLast);
  }
}
#endif // __VITIS_HLS__

} // svd

namespace tests {

static const int kNumInputs = 2;
static const int kInputSize = 512;
static const int Tu = 4;
// NOTE: The rest of the parameters are unused for now.
static const int kDummySize = 1;
static const int R = 8;
static const int Tv = 1;
static const int ZTu = 0;
static const int ZTv = 0;
static const int G = 4;

typedef svd::SvdParameters<tests::kNumInputs, tests::kInputSize,
    tests::kDummySize, tests::R, tests::Tu, tests::Tv, tests::ZTu, tests::ZTv,
    tests::G,
    // svd::ActivationD, svd::WeightD, svd::AccumD> params;
    short, short, short> params;

} // tests

#ifndef __VITIS_HLS__
#else
void HlsKernelS(
  const int num_refinements[tests::params::N],
  // const hls::vector<int, tests::params::N> num_refinements,
  hls::stream<typename tests::params::VectG_AxiPacketType>& xu_port,
  hls::stream<typename tests::params::VectG_AxiPacketType>& s_port,
  hls::stream<typename tests::params::VectG_AxiPacketType>& xus_port);
#endif

#endif // end KERNEL_S_KERNEL_H_