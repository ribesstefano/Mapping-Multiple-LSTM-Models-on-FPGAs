#ifndef KERNEL_SVD_KERNEL_H_
#define KERNEL_SVD_KERNEL_H_

#include "svd_params.h"
#include "dma/svd_dma.h"
#include "dma/axis_lib.h"
#include "kernel/u_kernel.h"
#include "kernel/s_kernel.h"
#include "kernel/v_kernel.h"

namespace svd {

template <typename params>
inline void SvdKernel(svd::SvdStreams<params> &streams) {
#pragma HLS INLINE
#ifndef __VITIS_HLS__
#pragma HLS DATAFLOW
#endif
  svd::KernelU<params>(params::R, streams);
  svd::KernelS<params>(params::R, streams);
  svd::KernelV<params>(params::R, streams);
}

template <
  typename params,
  typename WrapperAxisGTv = svd::AxiStreamPort<params::VectGTvAxiWidth>
>
void SvdKernel(const int num_active_inputs,
    const int input_size,
    const int output_size,
    const int num_refinements[params::N],
    hls::stream<typename params::VectTuAxiPacketType>& x_port,
    hls::stream<typename params::VectTuAxiPacketType>& u_port,
    hls::stream<typename params::VectG_AxiPacketType>& s_port,
    hls::stream<typename params::VectTvAxiPacketType>& v_port,
    hls::stream<typename WrapperAxisGTv::PacketType>& y_port) {
#pragma HLS TOP name=SvdKernel
#pragma HLS INLINE
#pragma HLS DATAFLOW
#ifndef __VITIS_HLS__
#pragma HLS STABLE variable=x_port
#pragma HLS STABLE variable=u_port
#pragma HLS STABLE variable=s_port
#pragma HLS STABLE variable=v_port
#pragma HLS STABLE variable=y_port
#endif
#pragma HLS ARRAY_PARTITION variable=num_refinements complete
  const bool pad_output = false;
  typedef svd::AxiStreamFifo<params::VectG_AxiWidth> WrapperFifoG;
  hls::stream<typename WrapperFifoG::PacketType> xu_port("xu_port");
  hls::stream<typename WrapperFifoG::PacketType> xus_port("xus_port");
#pragma HLS STREAM variable=xu_port depth=2
#pragma HLS STREAM variable=xus_port depth=2
  int num_refinements_u[params::N];
  int num_refinements_s[params::N];
  int num_refinements_v[params::N];
#pragma HLS ARRAY_PARTITION variable=num_refinements_u complete
#pragma HLS ARRAY_PARTITION variable=num_refinements_s complete
#pragma HLS ARRAY_PARTITION variable=num_refinements_v complete
  Duplicate_R_Stream:
  for (int i = 0; i < params::N; ++i) {
#pragma HLS UNROLL
    num_refinements_u[i] = num_refinements[i];
    num_refinements_s[i] = num_refinements[i];
    num_refinements_v[i] = num_refinements[i];
  }
  svd::KernelU<params, WrapperFifoG>(num_active_inputs, input_size,
    num_refinements_u, pad_output, x_port, u_port, xu_port);
  svd::KernelS<params, WrapperFifoG>(num_active_inputs, num_refinements_s,
    xu_port, s_port, xus_port);
  svd::KernelV<params, WrapperFifoG, WrapperAxisGTv>(num_active_inputs,
    output_size, num_refinements_v, xus_port, v_port, y_port);
}

/**
 * @brief      Sets the SVD kernel inputs, i.e. streams from arrays into
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
 * @param      x_port             The x port to be used as argument to SvdKernel
 * @param      u_port             The u port to be used as argument to SvdKernel
 * @param      s_port             The s port to be used as argument to SvdKernel
 * @param      v_port             The v port to be used as argument to SvdKernel
 *
 * @tparam     params             Collection of SVD configuration params.
 */
#ifdef __VITIS_HLS__
template<typename params>
void SetSvdKernelInputs(const int num_active_inputs,
    const int input_size,
    const int output_size,
    const int num_refinements[params::N],
    const typename params::ActivationD* x,
    const typename params::ActivationD* u,
    const typename params::ActivationD* s,
    const typename params::ActivationD* v,
    hls::stream<typename params::VectTuAxiPacketType>& x_port,
    hls::stream<typename params::VectTuAxiPacketType>& u_port,
    hls::stream<typename params::VectG_AxiPacketType>& s_port,
    hls::stream<typename params::VectTvAxiPacketType>& v_port) {
  typedef typename params::ActivationD ActivationType;
  const int kG = params::G;
  const int kTu = params::Tu;
  const int kTv = params::Tv;
  const int kGTv = kG * kTv;
  const int kNumTilesU = input_size / kTu;
  const int kNumTilesV = output_size / kTv;
  auto x_axis = svd::AxiStreamPort<params::VectTuAxiWidth>(x_port);
  auto u_axis = svd::AxiStreamPort<params::VectTuAxiWidth>(u_port);
  auto s_axis = svd::AxiStreamPort<params::VectG_AxiWidth>(s_port);
  auto v_axis = svd::AxiStreamPort<params::VectTvAxiWidth>(v_port);
  int max_R = num_refinements[0];
  typename params::VectTuType x_val;
  typename params::VectTuType u_val;
  typename params::VectG_Type s_val;
  typename params::VectTvType v_val;
  for (int i = i; i < params::N; ++i) {
    if (num_refinements[i] > max_R) {
      max_R = num_refinements[i];
    }
  }
  for (int j = 0; j < kNumTilesU; ++j) {
    for (int i = 0; i < num_active_inputs; ++i) {
      for (int k = 0; k < kTu; ++k) {
        x_val[k] = x[i * input_size + j * kTu + k];
      }
      x_axis.template PushVector<ActivationType, kTu>(x_val);
    }
  }
  for (int i = 0; i < max_R; ++i) {
    for (int j = 0; j < kNumTilesU; ++j) {
      for (int k = 0; k < kG; ++k) {
        for (int ii = 0; ii < kTu; ++ii) {
          u_val[ii] = u[i * kNumTilesU * kTu * kG + (j * kTu + ii) * kG + k];
        }
        u_axis.template PushVector<ActivationType, kTu>(u_val);
      }
    }
  }
  for (int i = 0; i < max_R; ++i) {
    for (int j = 0; j < num_active_inputs; ++j) {
      if (i < num_refinements[j]) {
        for (int k = 0; k < kG; ++k) {
          s_val[k] = s[i * num_active_inputs * kG + j * kG + k];
        }
        s_axis.template PushVector<ActivationType, kG>(s_val);
      }
    }
  }
  for (int i = 0; i < max_R; ++i) {
    for (int j = 0; j < kNumTilesV; ++j) {
      for (int k = 0; k < kG; ++k) {
        for (int ii = 0; ii < kTv; ++ii) {
          v_val[ii] = v[i * kNumTilesV * kTv * kG + (j * kTv + ii) * kG + k];
        }
        v_axis.template PushVector<ActivationType, kTv>(v_val);
      }
    }
  }
}
#endif // __VITIS_HLS__

/**
 * @brief      Gets the svd kernel outputs, i.e. fills in an array from
 *             hls::streams.
 *
 * @param[in]  num_active_inputs  The number active inputs
 * @param[in]  output_size        The output size (H)
 * @param      y_port             The y port to be used as argument to SvdKernel
 * @param      y                  The output array. Shape: (N, G, H)
 *
 * @tparam     params             Collection of SVD configuration params.
 */
#ifdef __VITIS_HLS__
template<typename params>
void GetSvdKernelOutputs(const int num_active_inputs, const int output_size,
    hls::stream<typename params::VectGTvAxiPacketType>& y_port,
    typename params::ActivationD* y) {
  typedef typename params::ActivationD ActivationType;
  const int kG = params::G;
  const int kTv = params::Tv;
  const int kGTv = kG * kTv;
  const int kNumTilesV = output_size / kTv;
  auto y_axis = svd::AxiStreamPort<params::VectGTvAxiWidth>(y_port);
  for (int j = 0; j < kNumTilesV; ++j) {
    for (int i = 0; i < num_active_inputs; ++i) {
      auto y_val = y_axis.template PopVector<ActivationType, kGTv>();
      for (int k = 0; k < kTv; ++k) {
        for (int ii = 0; ii < kG; ++ii) {
          int y_idx = i * output_size * kG + ii * output_size + j * kTv + k;
          y[y_idx] = y_val[k * kG + ii];
        }
      }
    }
  }
}
#endif // __VITIS_HLS__

} // svd

void HlsSvdKernel(const int num_active_inputs,
    const int input_size,
    const int output_size,
    const int num_refinements[svd::svd_params::N],
    hls::stream<typename svd::svd_params::VectTuAxiPacketType>& x_port,
    hls::stream<typename svd::svd_params::VectTuAxiPacketType>& u_port,
    hls::stream<typename svd::svd_params::VectG_AxiPacketType>& s_port,
    hls::stream<typename svd::svd_params::VectTvAxiPacketType>& v_port,
    hls::stream<typename svd::svd_params::VectGTvAxiPacketType>& y_port);

#endif // end KERNEL_SVD_KERNEL_H_