#ifndef KERNEL_SVD_KERNEL_H_
#define KERNEL_SVD_KERNEL_H_

#include "svd_params.h"
#include "dma/svd_dma.h"
#include "kernel/u_kernel.h"
#include "kernel/s_kernel.h"
#include "kernel/v_kernel.h"

template <typename params>
inline void SvdKernel(svd::SvdStreams<params> &streams) {
#pragma HLS INLINE
#ifndef __VITIS_HLS__
#pragma HLS DATAFLOW
#endif
  KernelU<params>(streams);
  KernelS<params>(streams);
  KernelV<params>(streams);
}

#endif // end KERNEL_SVD_KERNEL_H_