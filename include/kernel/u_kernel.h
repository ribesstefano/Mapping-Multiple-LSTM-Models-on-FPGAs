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
 *
 * @param      streams  The streams group
 * @param[in]  size  The size, ideally: params::R * params::PrunedSizeU / params::R
 *                   / params::PeU
 *
 * @tparam     params   The algorithm characteristics
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
template <typename ActivationType, typename WeightType, typename AccumType, int NumGates, int NumPEs>
void UDotUnit2LstmPe(const int vect_length, const int num_tiles,
                     const int num_iter,
                     hls::stream<ActivationType> x1_stream[NumGates][NumPEs],
                     hls::stream<ActivationType> x2_stream[NumGates][NumPEs],
                     hls::stream<WeightType> gate_u_stream[NumGates][NumPEs],
                     hls::stream<AccumType> acc1_stream[NumGates][NumPEs],
                     hls::stream<AccumType> acc2_stream[NumGates][NumPEs]) {
#pragma HLS INLINE off
#pragma HLS FUNCTION_INSTANTIATE variable=vect_length
#pragma HLS FUNCTION_INSTANTIATE variable=num_iter
#pragma HLS FUNCTION_INSTANTIATE variable=num_tiles
#pragma HLS FUNCTION_INSTANTIATE variable=num_timesteps
  assert(vect_length % num_tiles == 0);
  const int kNumInputs = 2;
  const int kTileSize = vect_length / num_tiles;
  AccumType y_mac[NumGates][NumPEs][kNumInputs] = {0};
  AccumType x_val[NumGates][NumPEs][kNumInputs] = {0};
#pragma HLS ARRAY_PARTITION variable=y_mac complete dim=0
#pragma HLS ARRAY_PARTITION variable=x_val complete dim=0
  U_Unit_PE:
  for (int i = 0; i < num_iter; ++i) {
    for (int j = 0; j < kTileSize; ++j) {
#pragma HLS PIPELINE II=1 style=frp
      for (int k = 0; k < NumGates; ++k) {
        for (int ii = 0; ii < NumPEs; ++ii) {
          auto u_val = gate_u_stream[k][ii].read();
          x_val[k][ii][0] = x1_stream[k][ii].read();
          x_val[k][ii][1] = x2_stream[k][ii].read();
          for (int jj = 0; jj < kNumInputs; ++jj) {
            if (j == 0) {
              y_mac[k][ii][jj] = 0;
            }
            auto mac = u_val * x_val[k][ii][jj];
            mac += y_mac[k][ii][jj];
#pragma HLS RESOURCE variable=mac core=DSP48
            y_mac[k][ii][jj] = mac;
          }
          if (j == kTileSize - 1) {
            acc1_stream[k][ii].write(y_mac[k][ii][0]);
            acc2_stream[k][ii].write(y_mac[k][ii][1]);
          }
        }
      }
    }
  }
}

/**
 * @brief      Reduce Product Unit of an LSTM gate. It Computes the parallel dot
 *             product between input x and a U vector. It also performs the
 *             refinement steps and feeds the Element Wise Product Unit.
 *
 * @todo (22/03/2019 - algorithm): The INTERNAL_BUFFER design needs to be
 * updated with the NumIter and NumTimesteps iterations.
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
 */
template <int VectLength, int NumTiles, int NumZeroTiles, int NumIter,
  int NumGates>
void UDotUnit2Lstm(svd::ActivationStream x1_streams[NumGates][NumTiles-NumZeroTiles],
                         svd::ActivationStream x2_streams[NumGates][NumTiles-NumZeroTiles],
                         WeightStream gate_u_streams[NumGates][NumTiles-NumZeroTiles],
                         svd::ActivationStream y1[NumGates],
                         svd::ActivationStream y2[NumGates]) {
  assert(VectLength % NumTiles == 0);
  assert(NumZeroTiles < NumTiles);
  assert(NumTiles >= 8);
  assert(NumTiles % 2 == 0);
// =============================================================================
#define REDUCE_PROD_2LSTM_DATAFLOW_DESIGN
// #define REDUCE_PROD_2LSTM_MERGE_DSP // the accuracy is killed, possible error.
// =============================================================================
#if !defined(REDUCE_PROD_2LSTM_DATAFLOW_DESIGN) && \
  defined(REDUCE_PROD_2LSTM_MERGE_DSP) && FIX_WIDTH == 8
#pragma HLS DATAFLOW
  // ===========================================================================
  // Implements shared DSP and LUT function for computing 2 mac ops in 1 DSP.
  // ===========================================================================
  const int kNumNonZeroTiles = NumTiles - NumZeroTiles;
  const int kNumPEs = kNumNonZeroTiles;
  const int kTileSize = VectLength / NumTiles;
  const int kStreamDepth = NumIter * kTileSize;
  svd::AccumD y1_mul[kNumPEs];
  svd::AccumD y2_mul[kNumPEs];
#pragma HLS ARRAY_PARTITION variable=y1_mul complete dim=1
#pragma HLS ARRAY_PARTITION variable=y2_mul complete dim=1
#pragma HLS STREAM variable=y1_mul depth=kStreamDepth
#pragma HLS STREAM variable=y2_mul depth=kStreamDepth

  svd::AccumD y1_acc = 0;
  svd::AccumD y2_acc = 0;
#pragma HLS RESOURCE variable=y1_acc core=AddSub_DSP
#pragma HLS RESOURCE variable=y2_acc core=AddSub_DSP

  for (int n = 0; n < NumIter * NumTimesteps; ++n) {
    ReduceProd_PE_Loop:
    for (int i = 0; i < kNumPEs; ++i) {
#if FIX_WIDTH == 8
#pragma HLS ALLOCATION instances=dot_prod_dsp_lut limit=kNumPEs function
#else
#pragma HLS ALLOCATION instances=dot_prod_dsp_lut_generic limit=kNumPEs function
#endif
#pragma HLS UNROLL
      y1_mul[i] = 0;
      y2_mul[i] = 0;
      ReduceProd_Tile_Loop:
      for (int j = 0; j < kTileSize / 2; ++j) {
#pragma HLS PIPELINE II=1 style=frp
        // auto p0_tmp = y_dsp * w_dsp + y_lut * w_lut;
        // auto p1_tmp = x_dsp * w_dsp + x_lut * w_lut;
        // p0 += p0_tmp;
        // p1 += p1_tmp;
#if FIX_WIDTH == 8
        svd::AccumD x_dsp = 0;  // x1_streams[i].read();
        svd::AccumD y_dsp = 0;  // x2_streams[i].read();
        svd::AccumD w_dsp = 0;  // gate_u_streams[i].read();
        svd::AccumD x_lut = 0;  // x1_streams[i].read();
        svd::AccumD y_lut = 0;  // x2_streams[i].read();
        svd::AccumD w_lut = 0;  // gate_u_streams[i].read();
        x_dsp.range() = x1_streams[i].read().range();
        y_dsp.range() = x2_streams[i].read().range();
        w_dsp.range() = gate_u_streams[i].read().range();
        x_lut.range() = x1_streams[i].read().range();
        y_lut.range() = x2_streams[i].read().range();
        w_lut.range() = gate_u_streams[i].read().range();
        dot_prod_dsp_lut(x_dsp, y_dsp, w_dsp, x_lut, y_lut, w_lut,
          y2_mul[i], y1_mul[i]);
#else
        svd::AccumD x_dsp = x1_streams[i].read();
        svd::AccumD y_dsp = x2_streams[i].read();
        svd::AccumD w_dsp = gate_u_streams[i].read();
        svd::AccumD x_lut = x1_streams[i].read();
        svd::AccumD y_lut = x2_streams[i].read();
        svd::AccumD w_lut = gate_u_streams[i].read();
        dot_prod_dsp_lut_generic(x_dsp, y_dsp, w_dsp, x_lut, y_lut, w_lut,
          y2_mul[i], y1_mul[i]);
#endif
      }
    }
  }

  ReduceProd_Accumulation_Loop:
  for (int i = 0; i < NumIter * NumTimesteps; ++i) {
    for (int j = 0; j < kNumPEs; ++j) {
#pragma HLS PIPELINE II=1 style=frp
      y1_acc += y1_mul[j];
      y2_acc += y2_mul[j];
    }
    y1.write(y1_acc);
    y2.write(y2_acc);
  }
#else
// =============================================================================
// Implements #mac_PEs = NumTiles - NumZeroTiles & #Adder_Tree = 1
// =============================================================================
#pragma HLS DATAFLOW
#pragma HLS INLINE
  const int kNumNonZeroTiles = NumTiles - NumZeroTiles;
  const int kNumInputs = 2;
  // NOTE: both PE and adder-tree have II=1, but the adder-tree reads in round
  // robin fashion from the PE queues. Hence, before the adder-tree reads again
  // from the same PE queue, kNumPEs cycles pass. This contrains the depth of
  // the queues to kNumPEs. (THIS WON'T WORK, TOO LOW CONSUMER RATE)
  // FIXED: Using an adder tree allows to use a stream of depth 1.
  const int kStreamDepth = 2; // VectLength / NumTiles;
  hls::stream<svd::AccumD> acc_streams[kNumInputs][NumGates][kNumNonZeroTiles];
#pragma HLS ARRAY_PARTITION variable=acc_streams complete dim=0
#pragma HLS STREAM variable=acc_streams depth=kStreamDepth
  svd::UDotUnit2LstmPe<svd::ActivationD, svd::WeightD, svd::AccumD, NumGates, kNumNonZeroTiles>(VectLength,
    NumTiles, NumIter, x1_streams, x2_streams, gate_u_streams,
    acc_streams[0], acc_streams[1]);
  UAccumUnit:
  for (int i = 0; i < NumIter; ++i) {
#pragma HLS PIPELINE II=1 style=frp
    for (int j = 0; j < NumGates; ++j) {      
      auto y1_val = svd::ActivationD(hlsutils::adder_tree<svd::AccumD, kNumNonZeroTiles>(acc_streams[0][j]));
      auto y2_val = svd::ActivationD(hlsutils::adder_tree<svd::AccumD, kNumNonZeroTiles>(acc_streams[1][j]));
      y1[j].write(y1_val);
      y2[j].write(y2_val);
    }
  }
#endif // end REDUCE_PROD_2LSTM_DATAFLOW_DESIGN
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
#ifndef __VITIS_HLS__
#pragma HLS STABLE variable=x_port
#pragma HLS STABLE variable=u_port
#pragma HLS STABLE variable=xu_port
#endif
  assert(num_active_inputs <= params::N);
  assert(num_active_inputs > 0);
  assert(params::I % params::Tu == 0);
  assert(input_size % params::Tu == 0);
  assert(input_size <= params::I);
  const int kNumTilesU = input_size / params::Tu;
  const int kMaxNumTilesU = params::I / params::Tu;
  const int kStreamDepth_X = 2 + kMaxNumTilesU * params::N;
  const int kStreamDepth_U = 8 + kMaxNumTilesU * params::N;
  const int kStreamDepth_XU = 2 + params::G;
  assert(kNumTilesU <= kMaxNumTilesU);
  typedef typename params::ActivationD ActivationType;
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
#endif // end __VITIS_HLS__

} // svd

namespace testu {

static const int kNumInputs = 4;
static const int kInputSize = 1024;
static const int Tu = 4;
// NOTE: The rest of the parameters are unused for now.
static const int kDummySize = 1;
static const int R = 8;
static const int Tv = 1;
static const int ZTu = 0;
static const int ZTv = 0;
static const int G = 4;

typedef svd::SvdParameters<testu::kNumInputs, testu::kInputSize,
    testu::kDummySize, testu::R, testu::Tu, testu::Tv, testu::ZTu, testu::ZTv,
    testu::G,
    // svd::ActivationD, svd::WeightD, svd::AccumD> params;
    short, short, short> params;

static const int VectTuAxiBitwidth = hlsutils::Bitwidth<typename params::ActivationD>::value * params::Tu;
static const int VectN_AxiBitwidth = hlsutils::Bitwidth<typename params::ActivationD>::value * params::N;
static const int VectGN_AxiBitwidth = hlsutils::Bitwidth<typename params::ActivationD>::value * params::G * params::N;
#ifdef __VITIS_HLS__
typedef hls::vector<typename params::ActivationD, params::Tu> VectTuType;
typedef hls::vector<typename params::ActivationD, params::N> VectN_Type;
typedef hls::vector<typename params::ActivationD, params::G * params::N> VectGN_Type;
#endif
typedef svd::AxiStreamPort<VectTuAxiBitwidth>::PacketType VectTuAxiPacketType;
typedef svd::AxiStreamPort<VectN_AxiBitwidth>::PacketType VectN_AxiPacketType;
typedef svd::AxiStreamPort<VectGN_AxiBitwidth>::PacketType VectGN_AxiPacketType;

} // testu

#ifndef __VITIS_HLS__
void HlsKernelU(const int num_refinements,
  const typename testu::params::ActivationD x_port[testu::params::N][testu::params::I],
  const typename testu::params::UPortD u_port[testu::params::R * testu::params::PrunedSizeU],
  typename testu::params::ActivationD xu_port[testu::params::N][testu::params::G * testu::params::R]);
#else
void HlsVectorKernelU(const int num_refinements,
  hls::stream<hls::vector<typename testu::params::ActivationD, testu::params::Tu> >& x_port,
  hls::stream<hls::vector<typename testu::params::ActivationD, testu::params::Tu> >& u_port,
  hls::stream<hls::vector<typename testu::params::ActivationD, testu::params::N> >& xu_port);

void HlsAxisKernelU(const int num_refinements,
  hls::stream<typename testu::VectTuAxiPacketType>& x_port,
  hls::stream<typename testu::VectTuAxiPacketType>& u_port,
  hls::stream<typename testu::VectGN_AxiPacketType>& xu_port);

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
  // const hls::vector<int, testu::params::N> num_refinements,
  const bool pad_output,
  hls::stream<typename testu::params::VectTuAxiPacketType>& x_port,
  hls::stream<typename testu::params::VectTuAxiPacketType>& u_port,
  hls::stream<typename testu::params::VectG_AxiPacketType>& xu_port);

#endif // end __VITIS_HLS__

#endif // end KERNEL_U_KERNEL_H_