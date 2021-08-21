#include "kernel/s_kernel.h"
#include "dma/axis_lib.h"

#include "hls_stream.h"
#ifdef __VITIS_HLS__
#include "hls_vector.h"
#endif

#ifdef __VITIS_HLS__
void HlsKernelS(const int num_active_inputs,
    const hls::vector<int, tests::params::N> num_refinements,
    hls::stream<typename tests::params::VectG_AxiType>& xu_port,
    hls::stream<typename tests::params::VectG_AxiType>& s_port,
    hls::stream<typename tests::params::VectG_AxiType>& xus_port) {
#pragma HLS INTERFACE axis port=xu_port
#pragma HLS INTERFACE axis port=s_port
#pragma HLS INTERFACE axis port=xus_port
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE s_axilite port=num_active_inputs
#pragma HLS INTERFACE s_axilite port=num_refinements
  svd::KernelS<tests::params>(num_active_inputs, num_refinements, xu_port,
    s_port, xus_port);
}
#endif

namespace svd {

} // svd