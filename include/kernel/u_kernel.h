#ifndef KERNEL_U_KERNEL_H_
#define KERNEL_U_KERNEL_H_

#include "svd_params.h"
#include "hls_utils/hls_metaprogramming.h"
#include "hls_utils/hls_debugging.h"
#include "hls_utils/adder_tree.h"
#include "dma/axis_lib.h"

#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "assert.h"
#ifdef __VITIS_HLS__
#include "hls_vector.h"
#endif

namespace svd {

/**
 * @brief      Kernel performing x @ U.
 * @deprecated Compile time parametrization only.
 *
 * @param[in]  num_refinements  The number refinements
 * @param      streams          The streams group
 * @param[in]  size  The size, ideally: params::R * params::PrunedSizeU / params::R
 *                   / params::PeU
 *
 * @tparam     params           The algorithm characteristics
 */
template <typename params>
void KernelU(const int num_refinements, svd::SvdStreams<params> &streams) {
  typename params::AccumulationD xu[params::N][params::G][params::PeU];
#pragma HLS ARRAY_PARTITION variable=xu complete dim=0
  for (int i = 0; i < num_refinements; ++i) {
    for (int j = 0; j < params::PrunedSizeU / params::PeU; ++j) {
#pragma HLS PIPELINE II=1 style=frp
      for (int k = 0; k < params::PeU; ++k) {
        for (int g = 0; g < params::G; ++g) {
          auto u = streams.u[g][k].read();
          for (int ii = 0; ii < params::N; ++ii) {
            if (j == 0) {
              xu[ii][g][k] = 0;
            }
            xu[ii][g][k] += u * streams.x[ii][g][k].read();
#pragma HLS RESOURCE variable=xu[ii][g][k] core=DSP48 latency=3
            if (j == params::PrunedSizeU / params::PeU - 1) {
              streams.xu[ii][g][k].write(xu[ii][g][k]);
            }
          }
        }
      }
    }
  }
}

/**
 * @brief      Performs MAC operation. The weight values are read directly from
 * a port (which can be modeled as a FIFO).
 *
 * @param      x1_stream      The x1 stream
 * @param      x2_stream      The x2 stream
 * @param[in]  gate_u_stream  The gate u stream
 * @param      acc1_stream    The output acc1 stream
 * @param      acc2_stream    The output acc2 stream
 *
 * @tparam     VectLength     The input length
 * @tparam     NumTiles       The number of tiles
 * @tparam     NumIter        The number of refinement steps
 * @tparam     NumTimesteps   The number of LSTM timesteps
 */
template <typename ActivationType, typename WeightType, typename AccumType>
void UDotUnit2LstmPe(const int vect_length, const int num_tiles,
                     const int num_iter, const int num_timesteps, 
                     hls::stream<ActivationType> &x1_stream,
                     hls::stream<ActivationType> &x2_stream,
                     hls::stream<WeightType> &gate_u_stream,
                     hls::stream<AccumType> &acc1_stream,
                     hls::stream<AccumType> &acc2_stream) {
#pragma HLS INLINE off
#pragma HLS FUNCTION_INSTANTIATE variable=vect_length
#pragma HLS FUNCTION_INSTANTIATE variable=num_iter
#pragma HLS FUNCTION_INSTANTIATE variable=num_tiles
#pragma HLS FUNCTION_INSTANTIATE variable=num_timesteps
  assert(vect_length % num_tiles == 0);
  const int kNumElemsPerTile = vect_length / num_tiles;
  AccumType y1_mac = 0;
  AccumType y2_mac = 0;
  U_PE_Loop:
  for (int i = 0; i < num_iter * num_timesteps; ++i) {
    for (int j = 0; j < kNumElemsPerTile; ++j) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II=1 style=frp
#else
#pragma HLS PIPELINE II=1
#endif
      if (j == 0) {
        y1_mac = 0;
        y2_mac = 0;
      }
      auto u_val = gate_u_stream.read();
      auto mac1 = u_val * x1_stream.read();
      auto mac2 = u_val * x2_stream.read();
      mac1 += y1_mac;
      mac2 += y2_mac;
#pragma HLS RESOURCE variable=mac1 core=DSP48
#pragma HLS RESOURCE variable=mac2 core=DSP48
      y1_mac = mac1;
      y2_mac = mac2;
      if (j == kNumElemsPerTile - 1) {
        acc1_stream.write(y1_mac);
        acc2_stream.write(y2_mac);
      }
    }
  }
}

/**
 * @brief      Reduce Product Unit of an LSTM gate. It Computes the parallel dot
 *             product between input x and a U vector. It also performs the
 *             refinement steps and feeds the Element Wise Product Unit.
 *
 * @param[in]  x1_streams      The input x of LSTM n.1
 * @param[in]  x2_streams      The input x of LSTM n.2
 * @param[in]  gate_u_streams  The common U weight vector component
 * @param[out] y1              The accumulated output y1
 * @param[out] y2              The accumulated output y2
 *
 * @tparam     VectLength      The length of the weight vector
 * @tparam     NumTiles        The number of tiles the vector is divided into
 * @tparam     NumZeroTiles    The number of zeroed, i.e. pruned, tiles
 * @tparam     NumIter         The number of refinement steps
 * @tparam     NumTimesteps    The number of LSTM timesteps
 */
template <int VectLength, int NumTiles, int NumZeroTiles, int NumIter,
  int NumTimesteps>
void UDotUnit2Lstm(svd::ActivationStream (&x1_streams)[NumTiles-NumZeroTiles],
                   svd::ActivationStream (&x2_streams)[NumTiles-NumZeroTiles],
                   WeightStream (&gate_u_streams)[NumTiles-NumZeroTiles],
                   svd::ActivationStream &y1,
                   svd::ActivationStream &y2) {
  assert(VectLength % NumTiles == 0);
  assert(NumZeroTiles < NumTiles);
  assert(NumTiles % 2 == 0);
#pragma HLS DATAFLOW
#pragma HLS INLINE
  const unsigned kNumPEs = NumTiles - NumZeroTiles;
  const unsigned kStreamDepth = 2;
  hls::stream<svd::AccumD> acc1_streams[kNumPEs];
  hls::stream<svd::AccumD> acc2_streams[kNumPEs];
#pragma HLS ARRAY_PARTITION variable=acc1_streams complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc1_streams complete dim=1
#pragma HLS STREAM variable=acc1_streams depth=kStreamDepth
#pragma HLS STREAM variable=acc2_streams depth=kStreamDepth
  PE_Loop:
  for (int pe = 0; pe < kNumPEs; ++pe) {
#pragma HLS UNROLL
    UDotUnit2LstmPe<svd::ActivationD, svd::WeightD, svd::AccumD>(VectLength,
      NumTiles, NumIter, NumTimesteps, x1_streams[pe], x2_streams[pe],
      gate_u_streams[pe], acc1_streams[pe], acc2_streams[pe]);
  }
  U_AdderTree_Loop:
  for (int i = 0; i < NumIter * NumTimesteps; ++i) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II=1 style=frp
#else
#pragma HLS PIPELINE II=1
#endif
    y1.write(hlsutils::adder_tree<svd::AccumD, kNumPEs>(acc1_streams));
    y2.write(hlsutils::adder_tree<svd::AccumD, kNumPEs>(acc2_streams));
  }
}

#ifdef __VITIS_HLS__
/**
 * @brief      Flexible Kernel-U.
 *
 * @param[in]  num_active_inputs  The number of active inputs
 * @param[in]  input_size         The input size
 * @param[in]  num_refinements    The number of refinements steps (R) per input:
 *                                the Rs must be positive, greater than zero and
 *                                in ASCENDING ORDER. Their amount must be less
 *                                or equal to num_active_inputs. There should be
 *                                #num_active_inputs defined Rs (with no gaps),
 *                                as only the first #num_active_inputs Rs will
 *                                be considered.
 * @param[in]  pad_output         Wether to pad output with zeroes
 * @param      x_port             The input x port
 * @param      u_port             The input u port
 * @param      xu_port            The output xu port
 *
 * @tparam     params             The collection of fixed parameters and
 *                                configurations.
 */
template <
  typename params,
  typename WrapperAxisG = svd::AxiStreamPort<params::VectG_AxiWidth>
>
void KernelU(const int num_active_inputs,
    const int input_size,
    const int num_refinements[params::N],
    const bool pad_output,
    hls::stream<typename params::VectTuAxiPacketType>& x_port,
    hls::stream<typename params::VectTuAxiPacketType>& u_port,
    hls::stream<typename WrapperAxisG::PacketType>& xu_port) {
#pragma HLS TOP name=KernelU
#pragma HLS DATAFLOW
#pragma HLS INLINE
#pragma HLS STABLE variable=x_port
#pragma HLS STABLE variable=u_port
#pragma HLS STABLE variable=xu_port
  typedef typename params::ActivationD ActivationType;
  const unsigned int kNumTilesU = input_size / params::Tu;
  const unsigned int kMaxNumTilesU = params::I / params::Tu;
  const unsigned int kStreamDepth_X = 2 + kMaxNumTilesU * params::N;
  const unsigned int kStreamDepth_U = 8 + kMaxNumTilesU * params::N;
  const unsigned int kStreamDepth_XU = 2 + params::G;
  assert(num_active_inputs > 0);
  assert(kNumTilesU > 0);
  assert(num_active_inputs <= params::N);
  assert(params::I % params::Tu == 0);
  assert(input_size % params::Tu == 0);
  assert(input_size <= params::I);
  assert(kNumTilesU <= kMaxNumTilesU);
  auto x_axis = svd::AxiStreamPort<params::VectTuAxiWidth>(x_port);
  auto u_axis = svd::AxiStreamPort<params::VectTuAxiWidth>(u_port);
  auto xu_axis = svd::AxiStreamInterface<WrapperAxisG>(xu_port);
  hls::stream<typename params::VectTuType> x_stream("x_stream");
  hls::stream<typename params::VectTuType> u_streams[params::G];
  hls::stream<ActivationType> xu_streams[params::G];
  ActivationType x_buffer[params::N][params::Tu][kMaxNumTilesU];
#pragma HLS STREAM variable=x_stream depth=kStreamDepth_X
#pragma HLS STREAM variable=u_streams depth=kStreamDepth_U
#pragma HLS STREAM variable=xu_streams depth=kStreamDepth_XU
#pragma HLS ARRAY_PARTITION variable=u_streams complete dim=1
#pragma HLS ARRAY_PARTITION variable=x_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=x_buffer complete dim=2
#pragma HLS BIND_STORAGE variable=x_buffer type=ram_t2p impl=bram latency=1
  /*
   * Ideally, if the Rs are ordered, it would be: R0 * N + (R1-R0) * (N-1) +
   * (R2-R1) * (N-2)
   *
   * Imagine we have: R0 = 2, R1 = 3, R2 = 6
   *
   * This means:
   *  - till refinement 2 we have input 0 to process
   *  - till refinement 3 we have input 1 to process
   *  - till refinement 6 we have input 2 to process
   *
   * So it would become:
   *
   * R_total = 2 * 3 + (3-2) * (3-1) + (6-3) * (3-2)
   */
  int R_max = num_refinements[0];
  int R_total = num_refinements[0] * num_active_inputs; // Total elements.
  Get_Total_R:
  for (int i = 1; i < num_active_inputs; ++i) {
#pragma HLS PIPELINE II=1 style=frp
    if (num_refinements[i] > R_max) {
      R_max = num_refinements[i];
    }
    assert(num_refinements[i] >= num_refinements[i - 1]);
    R_total += (num_refinements[i] - num_refinements[i - 1]) * (num_active_inputs - i);
  }

  int R_prev = 0;
  X_DMA:
  for (int ii = 0; ii < num_active_inputs; ++ii) {
    Stream_X_Tiles:
    for (int i = 0; i < num_refinements[ii] - R_prev; ++i) {
      assert(num_refinements[ii] - R_prev >= 1);
      for (int j = 0; j < kNumTilesU; ++j) {
        for (int k = 0; k < num_active_inputs - ii; ++k) {
#pragma HLS PIPELINE II=1 style=frp
          assert(num_active_inputs - ii >= 1);
          if (ii == 0 && i == 0) {
            auto x_val = x_axis.template PopVector<ActivationType, params::Tu>();
            x_stream << x_val;
            for (int jj = 0; jj < params::Tu; ++jj) {
              x_buffer[k][jj][j] = x_val[jj];
            }
          } else {
            assert(k + ii < params::N);
            typename params::VectTuType x_val;
            for (int jj = 0; jj < params::Tu; ++jj) {
              x_val[jj] = x_buffer[k + ii][jj][j];
            }
            x_stream << x_val;
          }
        }
      }
    }
    R_prev = num_refinements[ii];
  }
  U_DMA:
  for (int i = 0; i < R_max; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=params::R max=params::R
    for (int j = 0; j < kNumTilesU; ++j) {
      for (int k = 0; k < params::G; ++k) {
        auto u_val = u_axis.template PopVector<ActivationType, params::Tu>();
        for (int ii = 0; ii < num_active_inputs; ++ii) {
#pragma HLS PIPELINE II=1 style=frp
          if (i < num_refinements[ii]) {
            u_streams[k] << u_val;
          }
        }
      }
    }
  }
  U_Kernel:
  for (int i = 0; i < R_total; ++i) {
    for (int j = 0; j < kNumTilesU; ++j) {
#pragma HLS PIPELINE II=1 style=frp
      auto x_val = x_stream.read();
      for (int k = 0; k < params::G; ++k) {
        xu_streams[k] << hlsutils::adder_tree<ActivationType, params::Tu>(x_val * u_streams[k].read());
        // xu_streams[k] << (x_val * u_streams[k].read()).reduce_add();
      }
    }
  }
  int iter_cnt = 0;
  XU_DMA:
  for (int i = 0; i < R_max; ++i) {
    typename params::VectG_Type xu_out[params::N] = {typename params::VectG_Type(0)};
#pragma HLS ARRAY_PARTITION variable=xu_out complete dim=1
    for (int j = 0; j < kNumTilesU; ++j) {
      for (int k = 0; k < num_active_inputs; ++k) {
#pragma HLS PIPELINE II=1 style=frp
        for (int ii = 0; ii < params::G; ++ii) {
          if (i < num_refinements[k]) {
            xu_out[k][ii] += xu_streams[ii].read();
#pragma HLS BIND_OP variable=xu_out[k][ii] op=add impl=dsp
          }
        }
        if (i < num_refinements[k] && j == kNumTilesU - 1) {
          const bool kIsLast = (iter_cnt == R_total - 1 && !pad_output);
          xu_axis.template PushVector<ActivationType, params::G>(xu_out[k], kIsLast);
          ++iter_cnt;
        } else if (pad_output) {
          const bool kIsLast = i == R_max - 1 && j == kNumTilesU - 1 && k == num_active_inputs - 1; 
          xu_axis.template PushVector<ActivationType, params::G>(xu_out[k], kIsLast);
          ++iter_cnt;
        }
      }
    }
  }
}


template <
  typename params,
  typename WrapperAxisG = svd::AxiStreamPort<params::VectG_AxiWidth>
>
void KernelU_Pruned(const int num_active_inputs,
    const int input_size,
    const int num_refinements[params::N],
    const int num_zero_tiles_u,
    hls::stream<typename params::VectGZTuAxiPacketType>& unz_idx_port,
    hls::stream<typename params::VectTuAxiPacketType>& x_port,
    hls::stream<typename params::VectTuAxiPacketType>& u_port,
    hls::stream<typename WrapperAxisG::PacketType>& xu_port) {
#pragma HLS TOP name=KernelU
#pragma HLS DATAFLOW
// #pragma HLS INLINE
#pragma HLS STABLE variable=unz_idx_port
#pragma HLS STABLE variable=x_port
#pragma HLS STABLE variable=u_port
#pragma HLS STABLE variable=xu_port
  typedef typename params::ActivationD ActivationType;
  const unsigned int kNumTilesU = input_size / params::Tu;
  const unsigned int kMaxNumTilesU = params::I / params::Tu;
  const unsigned int kStreamDepth_X = 2 + kMaxNumTilesU * params::N;
  const unsigned int kStreamDepth_U = 8 + kMaxNumTilesU * params::N;
  const unsigned int kStreamDepth_XU = 2 + params::G;
  assert(num_active_inputs > 0);
  assert(kNumTilesU > 0);
  assert(num_active_inputs <= params::N);
  assert(params::I % params::Tu == 0);
  assert(input_size % params::Tu == 0);
  assert(input_size <= params::I);
  assert(kNumTilesU <= kMaxNumTilesU);
  auto uz_axis = svd::AxiStreamPort<params::NumGTuBitsAligned>(unz_idx_port);
  auto x_axis = svd::AxiStreamPort<params::VectTuAxiWidth>(x_port);
  auto u_axis = svd::AxiStreamPort<params::VectTuAxiWidth>(u_port);
  auto xu_axis = svd::AxiStreamInterface<WrapperAxisG>(xu_port);
  hls::stream<typename params::VectTuType> x_stream[params::G];
  hls::stream<typename params::VectTuType> u_streams[params::G];
  hls::stream<ActivationType> xu_streams[params::G];
  ActivationType x_buffer[params::N][params::Tu][kMaxNumTilesU];
#pragma HLS STREAM variable=x_stream depth=kStreamDepth_X
#pragma HLS STREAM variable=u_streams depth=kStreamDepth_U
#pragma HLS STREAM variable=xu_streams depth=kStreamDepth_XU
#pragma HLS ARRAY_PARTITION variable=u_streams complete dim=1
#pragma HLS ARRAY_PARTITION variable=x_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=x_buffer complete dim=2
#pragma HLS BIND_STORAGE variable=x_buffer type=ram_t2p impl=bram latency=1
  /*
   * Ideally, if the Rs are ordered, it would be: R0 * N + (R1-R0) * (N-1) +
   * (R2-R1) * (N-2)
   *
   * Imagine we have: R0 = 2, R1 = 3, R2 = 6
   *
   * This means:
   *  - till refinement 2 we have input 0 to process
   *  - till refinement 3 we have input 1 to process
   *  - till refinement 6 we have input 2 to process
   *
   * So it would become:
   *
   * R_total = 2 * 3 + (3-2) * (3-1) + (6-3) * (3-2)
   */
  int num_refinements_init[params::N];
  int num_refinements_x_dma[params::N];
  int num_refinements_u_dma[params::N];
  int num_refinements_xu_dma[params::N];
#pragma HLS ARRAY_PARTITION variable=num_refinements_init complete
#pragma HLS ARRAY_PARTITION variable=num_refinements_x_dma complete
#pragma HLS ARRAY_PARTITION variable=num_refinements_u_dma complete
#pragma HLS ARRAY_PARTITION variable=num_refinements_xu_dma complete
  for (int i = 0; i < params::N; ++i) {
#pragma HLS UNROLL
    num_refinements_init[i] = num_refinements[i];
    num_refinements_x_dma[i] = num_refinements[i];
    num_refinements_u_dma[i] = num_refinements[i];
    num_refinements_xu_dma[i] = num_refinements[i];
  }

  std::cout << "[INFO] Get total R." << std::endl;
  // ===========================================================================
  // TODO: Same as non-pruned version -> wrap into a function (be careful to NTu-ZTu)
  // ===========================================================================
  int R_max = num_refinements_init[0];
  int R_total = num_refinements_init[0] * num_active_inputs; // Total elements.
  Get_Total_R:
  for (int i = 1; i < num_active_inputs; ++i) {
#pragma HLS PIPELINE II=1 style=frp
    if (num_refinements_init[i] > R_max) {
      R_max = num_refinements_init[i];
    }
    assert(num_refinements_init[i] >= num_refinements_init[i - 1]);
    R_total += (num_refinements_init[i] - num_refinements_init[i - 1]) * (num_active_inputs - i);
  }

  std::cout << "[INFO] X_DAM_in." << std::endl;
  // Added
  X_DAM_in:
  for (int i = 0; i < num_active_inputs; ++i) {
    for (int j = 0; j < kNumTilesU; ++j) {
#pragma HLS LOOP_FLATTEN      
#pragma HLS PIPELINE II=1 style=frp
      auto x_val = x_axis.template PopVector<ActivationType, params::Tu>();
      for (int k = 0; k < params::Tu; ++k) {
        x_buffer[i][k][j] = x_val[k];
      }
    }
  }

  typedef ap_uint<params::NumGTuBitsAligned> ZIndexType;
  auto get_idx = [](const ZIndexType nz_idx, const int i) {
    const int kHi = (i + 1) * params::NumTuBits - 1;
    const int kLo = i * params::NumTuBits;
    return nz_idx.range(kHi, kLo).to_int();
  };
  auto set_nz_idx = [](const int x) {
#pragma HLS PIPELINE II=1 style=frp
    ZIndexType nz_idx;
    const auto tmp = ap_uint<params::NumTuBits>(x);
    for (int i = 0; i < params::G; ++i) {
      const int kHi = (i + 1) * params::NumTuBits - 1;
      const int kLo = i * params::NumTuBits;
      nz_idx.range(kHi, kLo) = tmp.range();
    }
    return nz_idx;
  };
  std::cout << "[INFO] X_DMA_Dispatcher." << std::endl;
  // Changed
  int R_prev = 0;
  X_DMA_Dispatcher:
  for (int ii = 0; ii < num_active_inputs; ++ii) {
    for (int i = 0; i < num_refinements_x_dma[ii] - R_prev; ++i) {
      assert(num_refinements_x_dma[ii] - R_prev >= 1);
      for (int j = 0; j < kNumTilesU - num_zero_tiles_u; ++j) {
        auto nz_idx = num_zero_tiles_u > 0 ? uz_axis.template Pop<ZIndexType>() : set_nz_idx(j);
        for (int k = 0; k < num_active_inputs - ii; ++k) {
          assert(num_active_inputs - ii >= 1);
          assert(k + ii < params::N);
          for (int kk = 0; kk < params::G; ++kk) {
#pragma HLS LOOP_FLATTEN
#pragma HLS PIPELINE II=1 style=frp
            typename params::VectTuType x_val;
            for (int jj = 0; jj < params::Tu; ++jj) {
              x_val[jj] = x_buffer[k + ii][jj][get_idx(nz_idx, kk)];
            }
            x_stream[kk] << x_val;
          }
        }
      }
    }
    R_prev = num_refinements_x_dma[ii];
  }

  // ===========================================================================
  // TODO: Same as non-pruned version -> wrap into a function (be careful to NTv-ZTv)
  // ===========================================================================
  U_DMA:
  for (int i = 0; i < R_max; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=params::R max=params::R
    for (int j = 0; j < kNumTilesU - num_zero_tiles_u; ++j) {
      for (int k = 0; k < params::G; ++k) {
        auto u_val = u_axis.template PopVector<ActivationType, params::Tu>();
        for (int ii = 0; ii < num_active_inputs; ++ii) {
#pragma HLS PIPELINE II=1 style=frp
          if (i < num_refinements_u_dma[ii]) {
            u_streams[k] << u_val;
          }
        }
      }
    }
  }

  // Changed
  U_Kernel:
  for (int i = 0; i < R_total; ++i) {
    for (int j = 0; j < kNumTilesU - num_zero_tiles_u; ++j) {
#pragma HLS PIPELINE II=1 style=frp
      for (int k = 0; k < params::G; ++k) {
        xu_streams[k] << hlsutils::adder_tree<ActivationType, params::Tu>(
          x_stream[k].read() * u_streams[k].read());
      }
    }
  }
  
  // ===========================================================================
  // TODO: Same as non-pruned version -> wrap into a function (be careful to NTv-ZTv)
  // ===========================================================================
  int iter_cnt = 0;
  XU_DMA:
  for (int i = 0; i < R_max; ++i) {
    typename params::VectG_Type xu_out[params::N] = {typename params::VectG_Type(0)};
#pragma HLS ARRAY_PARTITION variable=xu_out complete dim=1
    for (int j = 0; j < kNumTilesU  - num_zero_tiles_u; ++j) {
      for (int k = 0; k < num_active_inputs; ++k) {
#pragma HLS PIPELINE II=1 style=frp
        for (int ii = 0; ii < params::G; ++ii) {
          if (i < num_refinements_xu_dma[k]) {
            xu_out[k][ii] += xu_streams[ii].read();
#pragma HLS BIND_OP variable=xu_out[k][ii] op=add impl=dsp
          }
        }
        if (i < num_refinements_xu_dma[k] && j == kNumTilesU  - num_zero_tiles_u - 1) {
          const bool kIsLast = iter_cnt == R_total - 1;
          xu_axis.template PushVector<ActivationType, params::G>(xu_out[k], kIsLast);
          ++iter_cnt;
        }
      }
    }
  } 
}
#endif // end __VITIS_HLS__

} // svd

namespace testu {

static const int kNumInputs = 2;
static const int kInputSize = 1024;
static const int Tu = 4;
static const int ZTu = 0;
// NOTE: The rest of the parameters are unused for now.
static const int kDummySize = 1;
static const int R = 8;
static const int Tv = 1;
static const int ZTv = 0;
static const int G = 4;

typedef svd::SvdParameters<testu::kNumInputs, testu::kInputSize,
    testu::kDummySize, testu::R, testu::Tu, testu::Tv, testu::ZTu, testu::ZTv,
    testu::G,
    // svd::ActivationD, svd::WeightD, svd::AccumD> params;
    short, short, short> params;
} // testu

#ifndef __VITIS_HLS__

/**
 * @brief      Synthesizeable Kernel-U.
 * @deprecated Compile time parametrization only.
 *
 * @param[in]  num_refinements  The number refinements
 * @param[in]  x_port           The x port
 * @param[in]  u_port           The u port
 * @param      xu_port          The xu port
 */
void HlsKernelU(const int num_refinements,
  const typename testu::params::ActivationD x_port[testu::params::N][testu::params::I],
  const typename testu::params::UPortD u_port[testu::params::R * testu::params::PrunedSizeU],
  typename testu::params::ActivationD xu_port[testu::params::N][testu::params::G * testu::params::R]);

#else

/**
 * @brief      Synthesizeable flexible Kernel-U.
 *
 * @param[in]  num_active_inputs  The number of active inputs
 * @param[in]  input_size         The input size
 * @param[in]  num_refinements    The number of refinements steps (R) per input:
 *                                the Rs must be positive, greater than zero and
 *                                in ASCENDING ORDER. Their amount must be less
 *                                or equal to num_active_inputs.
 * @param[in]  pad_output         Wether to pad output with zeroes
 * @param      x_port             The input x port
 * @param      u_port             The input u port
 * @param      xu_port            The output xu port
 */
void HlsKernelU(const int num_active_inputs,
  const int input_size,
  const int num_refinements[testu::params::N],
  const bool pad_output,
  // const int num_zero_tiles_u,
  // hls::stream<ap_uint<testu::NumTuBits> >& unz_idx_port,
  hls::stream<typename testu::params::VectTuAxiPacketType>& x_port,
  hls::stream<typename testu::params::VectTuAxiPacketType>& u_port,
  hls::stream<typename testu::params::VectG_AxiPacketType>& xu_port);

void HlsKernelU_Pruned(const int num_active_inputs,
    const int input_size,
    const int num_refinements[testu::params::N],
    const int num_zero_tiles_u,
    hls::stream<typename testu::params::VectGZTuAxiPacketType>& unz_idx_port,
    hls::stream<typename testu::params::VectTuAxiPacketType>& x_port,
    hls::stream<typename testu::params::VectTuAxiPacketType>& u_port,
    hls::stream<typename testu::params::VectG_AxiPacketType>& xu_port);

#endif // end __VITIS_HLS__

#endif // end KERNEL_U_KERNEL_H_