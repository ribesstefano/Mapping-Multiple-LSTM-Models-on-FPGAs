#ifndef KERNEL_S_KERNEL_H_
#define KERNEL_S_KERNEL_H_

#include "svd_params.h"
#include "hls_utils/adder_tree.h"

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

} // svd

#endif // end KERNEL_S_KERNEL_H_