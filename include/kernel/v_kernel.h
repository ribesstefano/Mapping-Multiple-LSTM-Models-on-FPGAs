#ifndef KERNEL_V_KERNEL_H_
#define KERNEL_V_KERNEL_H_

#include "svd_params.h"

namespace svd {

template <typename params>
void KernelV(const int num_refinements, svd::SvdStreams<params> &streams) {
#pragma HLS INLINE
#ifndef __VITIS_HLS__
#pragma HLS DATAFLOW
#endif
  typename params::VnzD nz_idx[params::G];
  typename params::AccumulationD xs[params::N][params::G][params::PeV];
  typename params::AccumulationD acc_buffer[params::N][params::G][params::PeV][params::Tv];
  typename params::ActivationS xusv[params::N][params::G][params::PeV];
#pragma HLS ARRAY_PARTITION variable=xs complete dim=0
#pragma HLS ARRAY_PARTITION variable=acc_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc_buffer complete dim=2
#pragma HLS ARRAY_PARTITION variable=acc_buffer complete dim=3
#pragma HLS RESOURCE variable=acc_buffer core=RAM_2P_BRAM

#ifndef __SYNTHESIS__
  Init_Acc_Buffer:
  for (int i = 0; i < params::Tv; ++i) {
#pragma HLS PIPELINE II=1
    for (int j = 0; j < params::PeV; ++j) {
      for (int k = 0; k < params::N; ++k) {
        for (int g = 0; g < params::G; ++g) {
          acc_buffer[k][g][j][i] = 0;
        }
      }
    }
  }
#endif
  if (params::ZTv > 0) { // constexpr
    V_Nz_Converter:
    for (int i = 0; i < num_refinements; ++i) {
      for (int j = 0; j < params::PrunedSizeV / params::PeV; ++j) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN
        for (int k = 0; k < params::G; ++k) {
          if (j == 0) {
            nz_idx[k] = streams.nz_v[k].read();
          }
          // NOTE: Vivado HLS is unable to properly inline the Priority Encoder,
          // resulting in a dependency on the function call (which generates a
          // II way above 1).
          typename params::VnzIdxD bit_idx = 0;
          Priority_Encoder:
          for (int ii = params::TvBits - 1; ii >= 0; --ii) {
            if (nz_idx[k][ii] == 1) {
              bit_idx = ii;
            }
          }
          for (int ii = 0; ii < params::PeV; ++ii) {
            streams.nz_v_idx[k][ii].write(bit_idx); // broadcast to parallel fifos
          }
          nz_idx[k][bit_idx] = 0; // unset the first set bit from left
        }
      }
    }
  }
  V_Unit:
  for (int i = 0; i < num_refinements; ++i) {
    for (int j = 0; j < params::PrunedSizeV / params::PeV; ++j) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN
      for (int g = 0; g < params::G; ++g) {
        for (int k = 0; k < params::PeV; ++k) {
          auto nz_idx = params::ZTv > 0 ? streams.nz_v_idx[g][k].read() : ap_uint<params::TvBits>(j);
          auto v = streams.v[g][k].read();
          for (int ii = 0; ii < params::N; ++ii) {
            if (j == 0) {
              xs[ii][g][k] = streams.xus[ii][g][k].read();
            }
            typedef typename params::AccumulationD accum_t;
            accum_t xusv = (xs[ii][g][k] * v) + acc_buffer[ii][g][k][nz_idx];
#pragma HLS RESOURCE variable=xusv core=DSP48 latency=3
// #pragma HLS DEPENDENCE variable=acc_buffer RAW false inter
            acc_buffer[ii][g][k][nz_idx] = xusv;
          }
        }
      }
    }
  }

  V_Writeback:
  for (int i = 0; i < params::Tv; ++i) {
#pragma HLS PIPELINE II=1
    for (int j = 0; j < params::PeV; ++j) {
      for (int k = 0; k < params::N; ++k) {
        for (int g = 0; g < params::G; ++g) {
          streams.xusv[k][g][j].write(acc_buffer[k][g][j][i]);
        }
      }
    }
  }
}

template <int VectLength, int NumTiles, int NumZeroTiles, int NumIter, int NumTimesteps>
void VDotUnit2LstmV2(const bool has_bias,
                   svd::WeightStream *bias1,
                   svd::WeightStream *bias2,
                   svd::ActivationStream &gate_dot1_streams,
                   svd::ActivationStream &gate_dot2_streams,
                   svd::WeightStream &gate_s1_streams,
                   svd::WeightStream &gate_s2_streams,
                   svd::WeightStream (&gate_v_streams)[VectLength / NumTiles],
                   hls::stream<ap_uint<NumTiles> > &nz_port,
                   svd::ActivationStream (&gate_out1_streams)[VectLength / NumTiles],
                   svd::ActivationStream (&gate_out2_streams)[VectLength / NumTiles]) {
#pragma HLS INLINE
#pragma HLS DATAFLOW
  assert(VectLength % NumTiles == 0);
  assert(NumTiles > NumZeroTiles);
  assert(NumTiles % 2 == 0);
  assert(NumIter % 2 == 0);
  const int kNumInputs = 2;
  const int kFifoResizeFactor = 4;
  const int kNonZeroTiles = NumTiles - NumZeroTiles;
  const int kTileSize = VectLength / NumTiles;
  // NOTE: By the time the dot products are available at the ports, the weight
  // values s1, s2 and v should be already at the FIFO ports.
  const int kStreamDepth = NumIter / kFifoResizeFactor;
  hls::stream<svd::MultD> xus_streams[kNumInputs][kTileSize];
#pragma HLS STREAM variable=xus_streams depth=kStreamDepth dim=0
  S_Kernel:
  for (int i = 0; i < NumIter; ++i) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
    svd::MultD xus_val[kNumInputs];
#pragma HLS ARRAY_PARTITION variable=xus_val complete dim=0
    xus_val[0] = gate_s1_streams.read() * gate_dot1_streams.read();
    xus_val[1] = gate_s2_streams.read() * gate_dot2_streams.read();
#pragma HLS RESOURCE variable=xus_val[0] core=DSP48 latency=3
#pragma HLS RESOURCE variable=xus_val[1] core=DSP48 latency=3
    for (int j = 0; j < kTileSize; ++j) {
      for (int k = 0; k < kNumInputs; ++k) {
        xus_streams[k][j].write(xus_val[k]);
      }
    }
  }
  svd::WeightStream bias_streams[kNumInputs][kTileSize];
  if (has_bias) {
#pragma HLS ARRAY_PARTITION variable=bias_streams complete dim=0
#pragma HLS STREAM variable=bias_streams depth=NumTiles
    Bias_DMA:
    for (int i = 0; i < NumTiles; ++i) {
      for (int j = 0; j < kTileSize; ++j) {
#pragma HLS PIPELINE II=1
        bias_streams[0][j].write(bias1->read());
        bias_streams[1][j].write(bias2->read());
      }
    }
  }
  const int kCombStreamDepth = NumIter * NumZeroTiles / (kFifoResizeFactor * 2);
  const int kNzBitLength = hlsutils::log2<NumTiles>::value;
  hls::stream<ap_uint<kNzBitLength> > nz_idx_streams[kTileSize];
#pragma HLS STREAM variable=nz_idx_streams depth=kCombStreamDepth dim=1
#pragma HLS RESOURCE variable=nz_idx_streams core=FIFO_SRL
// #define V_UNIT_USE_PRIORITY_ENCODER
#ifdef V_UNIT_USE_PRIORITY_ENCODER
  // ===========================================================================
  // NOTE: The critical path is HUGE here. So we go for the other solution.
  // ===========================================================================
  ap_uint<NumTiles> z_idx = 0;
  ZIndex_Converter:
  for (int i = 0; i < NumIter; ++i) {
    for (int j = 0; j < NumTiles - NumZeroTiles; ++j) {
#pragma HLS PIPELINE II=1
      if (j == 0) {
        z_idx = nz_port.read();
        int set_idx = PriorityEncoderLSB<NumTiles>(z_idx);
        assert(set_idx < NumTiles);
        for (int k = 0; k < kTileSize; ++k) {
          nz_idx_streams[k].write(set_idx);
        }
        z_idx[set_idx] = 0;
      } else {
        int set_idx = PriorityEncoderLSB<NumTiles>(z_idx);
        assert(set_idx < NumTiles);
        for (int k = 0; k < kTileSize; ++k) {
          nz_idx_streams[k].write(set_idx);
        }
        z_idx[set_idx] = 0;
      }
    }
  }
#else
  ap_uint<NumTiles> z_idx;
  int nz_cnt = 0;
  assert(nz_cnt < kNonZeroTiles);
  ZIndex_Converter:
  for (int i = 0; i < NumIter; ++i) {
#pragma HLS INLINE off
    for (int j = 0; j < NumTiles; ++j) {
#pragma HLS LOOP_FLATTEN
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=kNonZeroTiles max=kNonZeroTiles
      if (j == 0) {
        z_idx = nz_port.read();
        if (z_idx[0] == 1) {
          for (int k = 0; k < kTileSize; ++k) {
            nz_idx_streams[k].write(0);
          }
          nz_cnt++;
        }
      } else {
        if (z_idx[j] == 1) {
          for (int k = 0; k < kTileSize; ++k) {
            nz_idx_streams[k].write(j);
          }
          if (nz_cnt == kNonZeroTiles - 1) {
            nz_cnt = 0;
            break;
          } else {
            nz_cnt++;
          }
        }
      }
    }
  }
#endif
  V_Kernel: {
#pragma HLS INLINE off
    svd::AccumD acc_buffer[kNumInputs][kTileSize][NumTiles];
#pragma HLS ARRAY_PARTITION variable=acc_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc_buffer complete dim=2
    Init_buffer:
    for (int i = 0; i < NumTiles; ++i) {
#pragma HLS PIPELINE II=1
      for (int j = 0; j < kTileSize; ++j) {
        for (int k = 0; k < kNumInputs; ++k) {
          acc_buffer[k][j][i] = 0;
        }
      }
    }
    ap_uint<kNzBitLength> nz_idx[kTileSize];
    svd::AccumD xus[kNumInputs][kTileSize];
    svd::AccumD mac[kNumInputs][kTileSize];
    svd::AccumD acc[kNumInputs][kTileSize];
    svd::WeightD v[kTileSize];
#pragma HLS ARRAY_PARTITION variable=nz_idx complete
#pragma HLS ARRAY_PARTITION variable=xus complete dim=0
#pragma HLS ARRAY_PARTITION variable=mac complete dim=0
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0
#pragma HLS ARRAY_PARTITION variable=v complete
    for (int i = 0; i < NumIter; ++i) {
      for (int k = 0; k < kNonZeroTiles; ++k) {
#pragma HLS LOOP_FLATTEN
#pragma HLS PIPELINE II=1
        for (int j = 0; j < kTileSize; ++j) {
          if (k == 0) {
            for (int ii = 0; ii < kNumInputs; ++ii) {
              xus[ii][j] = xus_streams[ii][j].read();
            }
          }
          nz_idx[j] = nz_idx_streams[j].read();
          v[j] = gate_v_streams[j].read();
          for (int ii = 0; ii < kNumInputs; ++ii) {
            mac[ii][j] = (xus[ii][j] * v[j]) + acc_buffer[ii][j][nz_idx[j]];
#pragma HLS RESOURCE variable=mac[ii][j] core=DSP48 latency=3
#pragma HLS DEPENDENCE variable=acc_buffer RAW false inter distance=kNonZeroTiles
            acc_buffer[ii][j][nz_idx[j]] = mac[ii][j];  
          }
        }
      }
    }
    V_DMA:
    for (int i = 0; i < NumTiles; ++i) {
#pragma HLS PIPELINE II=1
      for (int j = 0; j < kTileSize; ++j) {
        if (has_bias) {
          auto acc_1 = acc_buffer[0][j][i] + bias_streams[0][j].read();
          auto acc_2 = acc_buffer[1][j][i] + bias_streams[1][j].read();
#pragma HLS RESOURCE variable=acc_1 core=AddSub_DSP
#pragma HLS RESOURCE variable=acc_2 core=AddSub_DSP
          gate_out1_streams[j].write(acc_1);
          gate_out2_streams[j].write(acc_2);
        } else {
          gate_out1_streams[j].write(acc_buffer[0][j][i]);
          gate_out2_streams[j].write(acc_buffer[1][j][i]);
        }
      }
    }
  } // end V_Function
}


#ifndef __VITIS_HLS__
#else
template <
  typename params,
  typename WrapperAxisG = svd::AxiStreamPort<params::VectG_AxiWidth>,
  typename WrapperAxisGTv = svd::AxiStreamPort<params::VectGTvAxiWidth>
>
void KernelV(const int num_active_inputs,
    const int output_size,
    const int num_refinements[params::N],
    hls::stream<typename WrapperAxisG::PacketType>& xus_port,
    hls::stream<typename params::VectTvAxiPacketType>& v_port,
    hls::stream<typename WrapperAxisGTv::PacketType>& y_port) {
#pragma HLS TOP name=KernelV
#pragma HLS DATAFLOW
#pragma HLS INLINE
#ifndef __VITIS_HLS__
#pragma HLS STABLE variable=xus_port
#pragma HLS STABLE variable=v_port
#pragma HLS STABLE variable=y_port
#endif
  assert(num_active_inputs <= params::N);
  assert(num_active_inputs > 0);
  assert(params::H % params::Tv == 0);
  assert(output_size % params::Tv == 0);
  assert(output_size <= params::H);
  typedef typename params::ActivationD ActivationType;
  const int kMaxNumTilesV = params::H / params::Tv;
  const int kNumTilesV = output_size / params::Tv;
  const int kStreamDepth_V = 32 + kMaxNumTilesV * params::G;
  assert(kNumTilesV <= kMaxNumTilesV);
  auto xus_axis = svd::AxiStreamInterface<WrapperAxisG>(xus_port);
  auto v_axis = svd::AxiStreamPort<params::VectTvAxiWidth>(v_port);
  auto y_axis = svd::AxiStreamInterface<WrapperAxisGTv>(y_port);
  hls::stream<typename params::VectTvType, kStreamDepth_V> v_streams[params::G];
  // NOTE: Having y_buffer as static made cosim work in one-process configuration.
  static ActivationType y_buffer[params::G][params::N][params::Tv][kMaxNumTilesV] = {0};
  typename params::VectTvType v_val;
  typename params::VectG_Type xus_val[params::N];
  typename params::VectGTvType y_out;
// NOTE: I'm not accessing dimension N of y_buffer in parallel.
#pragma HLS ARRAY_PARTITION variable=v_streams complete
#pragma HLS ARRAY_PARTITION variable=y_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=y_buffer complete dim=3
#pragma HLS BIND_STORAGE variable=y_buffer type=ram_t2p impl=bram latency=1
  int R_max = num_refinements[0];
  Get_Max_R:
  for (int i = 1; i < num_active_inputs; ++i) {
#pragma HLS PIPELINE II=1
    if (num_refinements[i] > R_max) {
      R_max = num_refinements[i];
    }
  }
  V_DMA:
  for (int i = 0; i < R_max; ++i) {
    for (int j = 0; j < kNumTilesV; ++j) {
      for (int k = 0; k < params::G; ++k) {
        for (int ii = 0; ii < num_active_inputs; ++ii) {
#pragma HLS PIPELINE II=1
          if (ii == 0) {
            v_val = v_axis.template PopVector<ActivationType, params::Tv>();
          }
          if (i < num_refinements[ii]) {
            v_streams[k] << v_val;
          }
        }
      }
    }
  }
  V_Kernel:
  for (int i = 0; i < R_max; ++i) {
    for (int j = 0; j < kNumTilesV; ++j) {
      for (int k = 0; k < num_active_inputs; ++k) {
#pragma HLS PIPELINE II=1
        for (int ii = 0; ii < params::G; ++ii) {
          assert(j < kMaxNumTilesV);
          assert(k < params::N);
          if (i < num_refinements[k]) {
            assert(i < 512);
            if (j == 0 && ii == 0) {
              xus_val[k] = xus_axis.template PopVector<ActivationType, params::G>();
            }
            auto v_val = v_streams[ii].read();
            for (int jj = 0; jj < params::Tv; ++jj) {
              ActivationType y_val;
              if (i == 0) {
                y_val = v_val[jj] * xus_val[k][ii];
              } else {
                y_val = y_buffer[ii][k][jj][j] + v_val[jj] * xus_val[k][ii];
              }
              y_buffer[ii][k][jj][j] = y_val;
// #pragma HLS DEPENDENCE raw inter variable=y_buffer false distance=kNumTilesV
            }
          }
        }
        if (i == R_max - 1) {
          for (int jj = 0; jj < params::G; ++jj) {
            for (int ii = 0; ii < params::Tv; ++ii) {
              y_out[ii * params::G + jj] = y_buffer[jj][k][ii][j];
            }
          }
          const bool kIsLast = j == kNumTilesV-1 && k == num_active_inputs-1;
          const int kGTv = params::G * params::Tv;
          y_axis.template PushVector<ActivationType, kGTv>(y_out, kIsLast);
        }
      }
    }
//     if (i == R_max - 1) {
//       for (int j = 0; j < kNumTilesV; ++j) {
//         for (int k = 0; k < num_active_inputs; ++k) {
//           for (int jj = 0; jj < params::G; ++jj) {
//             for (int ii = 0; ii < params::Tv; ++ii) {
// #pragma HLS PIPELINE II=1
//               y_out[ii * params::G + jj] = y_buffer[jj][k][ii][j];
//             }
//           }
//           const bool kIsLast = j == kNumTilesV-1 && k == num_active_inputs-1;
//           const int kGTv = params::G * params::Tv;
//           y_axis.template PushVector<ActivationType, kGTv>(y_out, kIsLast);
//         }
//       }
//     }
  }
//   DMA_Out:
//   for (int j = 0; j < kNumTilesV; ++j) {
//     for (int k = 0; k < num_active_inputs; ++k) {
// #pragma HLS PIPELINE II=1
//       for (int jj = 0; jj < params::G; ++jj) {
//         for (int ii = 0; ii < params::Tv; ++ii) {
//           assert(ii * params::G + jj < params::G * params::Tv);
//           y_out[ii * params::G + jj] = y_buffer[jj][k][ii][j];
//           // y_out[ii * params::G + jj] = y_val[jj][ii];
//         }
//       }
//       const bool kIsLast = j == kNumTilesV-1 && k == num_active_inputs-1;
//       const int kGTv = params::G * params::Tv;
//       y_axis.template PushVector<ActivationType, kGTv>(y_out, kIsLast);
//     }
//   }
}
#endif // end __VITIS_HLS__

} // svd

namespace testv {

static const int kNumInputs = 2;
static const int kInputSize = 512;
static const int Tu = 4;
// NOTE: The rest of the parameters are unused for now.
static const int kOutputSize = 512;
static const int R = 64;
static const int Tv = 4;
static const int ZTu = 0;
static const int ZTv = 0;
static const int G = 4;

typedef svd::SvdParameters<testv::kNumInputs, testv::kInputSize,
    testv::kOutputSize, testv::R, testv::Tu, testv::Tv, testv::ZTu, testv::ZTv,
    testv::G,
    // svd::ActivationD, svd::WeightD, svd::AccumD> params;
    short, short, short> params;
    // ap_fixed<FIX_WIDTH, FIX_FRACT_WIDTH, AP_TRN_ZERO>, ap_fixed<FIX_WIDTH, FIX_FRACT_WIDTH, AP_TRN_ZERO>, ap_fixed<FIX_WIDTH, FIX_FRACT_WIDTH, AP_TRN_ZERO> > params;
    // float, float, float > params;

} // testv

#ifndef __VITIS_HLS__
#else
void HlsKernelV(const int num_active_inputs,
    const int output_size,
    const int num_refinements[testv::params::N],
    // const hls::vector<int, testv::params::N> num_refinements,
    hls::stream<typename testv::params::VectG_AxiPacketType>& xus_port,
    hls::stream<typename testv::params::VectTvAxiPacketType>& v_port,
    hls::stream<typename testv::params::VectGTvAxiPacketType>& y_port);
#endif // end __VITIS_HLS__

#endif // end KERNEL_V_KERNEL_H_
