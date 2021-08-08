#include "kernel/v_kernel.h"

#include "hls_stream.h"
#ifdef __VITIS_HLS__
#include "hls_vector.h"
#endif

#include "assert.h"

#ifndef __VITIS_HLS__
#else
void HlsKernelV(const unsigned int output_size, 
  const hls::vector<int, testv::params::N> num_refinements,
  hls::stream<typename testv::params::VectG_AxiType>& xus_port,
  hls::stream<typename testv::params::VectTvAxiType>& v_port,
  hls::stream<typename testv::params::VectGTvAxiType>& y_port) {
#pragma HLS INTERFACE axis port=xus_port
#pragma HLS INTERFACE axis port=v_port
#pragma HLS INTERFACE axis port=y_port
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE s_axilite port=num_refinements
#pragma HLS DATAFLOW
  assert(num_refinements >= 1);
  assert(testv::params::H % testv::params::Tv == 0);
  assert(output_size % testv::params::Tv == 0);
  assert(output_size <= testv::params::H);
  typedef typename testv::params::ActivationD ActivationType;
  const int kNumTilesV = output_size / testv::params::Tv;
  const int kMaxNumTilesV = testv::params::H / testv::params::Tv;
  const int kStreamDepth_V = 8 + kMaxNumTilesV * testv::params::N;
  assert(kNumTilesV <= kMaxNumTilesV);
  auto xus_axis = svd::AxiStreamInterface<testv::params::VectG_AxiWidth>(xus_port);
  auto v_axis = svd::AxiStreamInterface<testv::params::VectTvAxiWidth>(v_port);
  auto y_axis = svd::AxiStreamInterface<testv::params::VectGTvAxiWidth>(y_port);
  hls::stream<testv::params::VectTvType> v_streams[testv::params::G];
  testv::params::VectTvType y_buffer[testv::params::G][testv::params::N][kMaxNumTilesV];
#pragma HLS STREAM variable=v_streams depth=kStreamDepth_V
#pragma HLS ARRAY_PARTITION variable=v_streams complete dim=1
#pragma HLS ARRAY_PARTITION variable=y_buffer complete dim=1
#pragma HLS BIND_STORAGE variable=y_buffer type=ram_t2p impl=bram latency=2

  int R_max = num_refinements[testv::params::N - 1];
  int R_total = num_refinements[0] * testv::params::N; // Total elements.
  Get_Total_R:
  for (int i = 1; i < testv::params::N; ++i) {
#pragma HLS PIPELINE II=1
    if (num_refinements[i] > R_max) {
      R_max = num_refinements[i];
    }
    R_total += (num_refinements[i] - num_refinements[i - 1]) * (testv::params::N - i);
  }

  V_DMA:
  for (int i = 0; i < testv::params::R; ++i) {
    for (int j = 0; j < kNumTilesV; ++j) {
      for (int k = 0; k < testv::params::G; ++k) {
        for (int ii = 0; ii < testv::params::N; ++ii) {
#pragma HLS PIPELINE II=1
          auto v_val = v_axis.PopVector<ActivationType, testv::params::Tv>();
          v_streams[k].write(v_val);
        }
      }
    }
  }

  V_Kernel:
  for (int i = 0; i < testv::params::R; ++i) {
    for (int j = 0; j < kNumTilesV; ++j) {
      for (int k = 0; k < testv::params::N; ++k) {
#pragma HLS PIPELINE II=1
        auto xus_val = xus_axis.PopVector<ActivationType, testv::params::G>();
        for (int ii = 0; ii < testv::params::G; ++ii) {
          auto v_val = v_streams[ii].read();
          y_buffer[ii][k][j] += v_val * xus_val[ii];
        }
      }
    }
    if (i == testv::params::R - 1) {
      testv::params::VectGTvType y_out = testv::params::VectGTvType(0);
#pragma HLS LOOP_MERGE
      for (int j = 0; j < kNumTilesV; ++j) {
        for (int k = 0; k < testv::params::N; ++k) {
          for (int ii = 0; ii < testv::params::Tv; ++ii) {
            for (int jj = 0; jj < testv::params::G; ++jj) {
#pragma HLS PIPELINE II=1
              y_out[ii * testv::params::G + jj] = y_buffer[jj][k][j][ii];
            }
          }
          const bool kIsLast = (j == kNumTilesV - 1 && k == testv::params::N - 1) ? true : false;
          y_axis.PushVector<ActivationType, testv::params::G * testv::params::Tv>(y_out);
        }
      }
    }
  }
}
#endif