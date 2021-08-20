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
#pragma HLS PIPELINE II=1
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
// #pragma HLS INTERFACE ap_ctrl_none port=return
  assert(vect_length % num_tiles == 0);

  const int kNumElemsPerTile = vect_length / num_tiles;
  AccumType y1_mac = 0;
  AccumType y2_mac = 0;

  ReduceProd_PE_IterTimesteps_Loop:
  for (int i = 0; i < num_iter * num_timesteps; ++i) {
    ReduceProd_PE_Loop:
    for (int j = 0; j < kNumElemsPerTile; ++j) {
#pragma HLS PIPELINE II=1
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
 * @brief      Accumulate partial results from ReduceProd PEs.
 *
 * @param      acc1_streams     The acc 1 streams, each from a PE
 * @param      acc2_streams     The acc 2 streams, each from a PE
 * @param      y1_stream        The single y_1 stream
 * @param      y2_stream        The single y_2 stream
 *
 * @tparam     VectLength       The input vector dimension
 * @tparam     NumTiles         The number of used tiles (to determine the
 *                              number of PEs)
 * @tparam     NumZeroTiles     The number of pruned tiles (to determine the
 *                              number of PEs)
 * @tparam     NumIter          The number of refinement steps (to make the
 *                              pipeline longer)
 * @tparam     NumTimesteps     The number of LSTM timesteps (to make the
 *                              pipeline longer)
 * @tparam     AdderTreeDesign  Enable or disable AdderTree design. Default is
 *                              active, i.e. true.
 */
template <int VectLength, int NumTiles, int NumZeroTiles, int NumIter,
  int NumTimesteps, bool AdderTreeDesign = true>
void UDotUnit2LstmAccumulator(svd::AccumStream (&acc1_streams)[NumTiles-NumZeroTiles],
    svd::AccumStream (&acc2_streams)[NumTiles-NumZeroTiles],
    svd::ActivationStream &y1_stream,
    svd::ActivationStream &y2_stream) {
#pragma HLS INLINE off
// #pragma HLS INTERFACE ap_ctrl_none port=return
  const int kNumPEs = NumTiles - NumZeroTiles;

  if (AdderTreeDesign) {
    // Determine the number of ranks for the adder tree and declare array
    // - The adder_tree is larger than required as each rank only needs to be half the size of the previous rank
    const unsigned kNumPEsLog2 = hlsutils::log2<kNumPEs>::value;
    const unsigned kNumPEsSub1Log2 = hlsutils::log2<kNumPEs - 1>::value;
    const unsigned kNumRanks = kNumPEsLog2 != kNumPEsSub1Log2 ? kNumPEsLog2 : kNumPEsLog2 + 1;
    svd::AccumD adder_tree1[kNumRanks][kNumPEs];
    svd::AccumD adder_tree2[kNumRanks][kNumPEs];

    unsigned rank_size = kNumPEs;

    for (int i = 0; i < NumIter * NumTimesteps; ++i) {
#pragma HLS PIPELINE II=1
      add_level_loop:
      for(int adder_tree_rank = kNumRanks - 1; adder_tree_rank >= 0; --adder_tree_rank) {
        const bool kLoopInit = adder_tree_rank == kNumRanks - 1 ? true : false;
        const bool kLoopEpilog = adder_tree_rank == 0 ? true : false;

        if (kLoopInit) {
          rank_size = kNumPEs;
        }

        const bool prev_rank_is_odd = rank_size % 2 == 0 ? false : true;
        rank_size = (rank_size + 1) / 2;
        // std::cout << "[" << adder_tree_rank << "] rank_size: " << rank_size << "\n";

        add_col_loop:
        for(int jj = 0; jj < (kNumPEs + 1) / 2; ++jj) {
          if (jj < rank_size) {
            if (prev_rank_is_odd && jj == rank_size - 1) {
              // Bypass, no adder required.
              if (kLoopInit) {
                adder_tree1[adder_tree_rank][jj] = acc1_streams[jj * 2].read();
                adder_tree2[adder_tree_rank][jj] = acc2_streams[jj * 2].read();
                // std::cout << "\t\tstream[" << adder_tree_rank << "][" << jj * 2 << "] = [" << jj << "]\n";
              } else {
                adder_tree1[adder_tree_rank][jj] = adder_tree1[adder_tree_rank + 1][jj * 2];
                adder_tree2[adder_tree_rank][jj] = adder_tree2[adder_tree_rank + 1][jj * 2];
                // std::cout << "\t\tbuffer[" << adder_tree_rank << "][" << jj * 2 << "] = [" << adder_tree_rank + 1 << "][" << jj << "]\n";
              }
            } else {
              if (kLoopInit) {
                auto y1_acc = acc1_streams[jj * 2].read() + acc1_streams[jj * 2 + 1].read();
                auto y2_acc = acc2_streams[jj * 2].read() + acc2_streams[jj * 2 + 1].read();
#pragma HLS RESOURCE variable=y1_acc core=AddSub_DSP
#pragma HLS RESOURCE variable=y2_acc core=AddSub_DSP
                adder_tree1[adder_tree_rank][jj] = y1_acc;
                adder_tree2[adder_tree_rank][jj] = y2_acc;
                // std::cout << "\tstreams[" << adder_tree_rank << "][" << jj << "] = [" << jj * 2 << "] + [" << jj * 2 + 1 << "]\n";
              } else{
                auto y1_acc = adder_tree1[adder_tree_rank + 1][jj * 2] + adder_tree1[adder_tree_rank + 1][jj * 2 + 1];
                auto y2_acc = adder_tree2[adder_tree_rank + 1][jj * 2] + adder_tree2[adder_tree_rank + 1][jj * 2 + 1];
#pragma HLS RESOURCE variable=y1_acc core=AddSub_DSP
#pragma HLS RESOURCE variable=y2_acc core=AddSub_DSP
                adder_tree1[adder_tree_rank][jj] = y1_acc;
                adder_tree2[adder_tree_rank][jj] = y2_acc;
                // std::cout << "\tbuffer[" << adder_tree_rank << "][" << jj << "] = [" << adder_tree_rank + 1 << "][" << jj * 2 << "] + [" << adder_tree_rank  + 1 << "][" << jj * 2 + 1 << "]\n";
              }
            }
          }
        }
        if (kLoopEpilog) {
          y1_stream.write(adder_tree1[0][0]);
          y2_stream.write(adder_tree2[0][0]);
          // std::cout << "\n";
        }
      }
    }
  } else {
    svd::AccumD y1_acc = 0;
    svd::AccumD y2_acc = 0;
    for (int i = 0; i < NumIter * NumTimesteps; ++i) {
      AdderTree_PE_Loop:
      for (int j = 0; j < kNumPEs; ++j) {
#pragma HLS PIPELINE II=1
        if (j == 0) {
          y1_acc = 0;
          y2_acc = 0;
        }
        auto acc1 = y1_acc + acc1_streams[j].read();
        auto acc2 = y2_acc + acc2_streams[j].read();
#pragma HLS RESOURCE variable=acc1 core=AddSub_DSP
#pragma HLS RESOURCE variable=acc2 core=AddSub_DSP
        y1_acc = acc1;
        y2_acc = acc2;
        if (j == kNumPEs - 1) {
          y1_stream.write(y1_acc);
          y2_stream.write(y2_acc);
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
  int NumTimesteps>
void UDotUnit2Lstm(svd::ActivationStream (&x1_streams)[NumTiles-NumZeroTiles],
                         svd::ActivationStream (&x2_streams)[NumTiles-NumZeroTiles],
                         WeightStream (&gate_u_streams)[NumTiles-NumZeroTiles],
                         svd::ActivationStream &y1,
                         svd::ActivationStream &y2) {
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
  const int kNumElemsPerTile = VectLength / NumTiles;
  const int kStreamDepth = NumIter * kNumElemsPerTile;
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
      for (int j = 0; j < kNumElemsPerTile / 2; ++j) {
#pragma HLS PIPELINE II=1
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
#pragma HLS PIPELINE II=1
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
// #pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS DATAFLOW
// #pragma HLS INLINE

  const unsigned kNumNonZeroTiles = NumTiles - NumZeroTiles;
  const unsigned kNumPEs = kNumNonZeroTiles;
  // NOTE: both PE and adder-tree have II=1, but the adder-tree reads in round
  // robin fashion from the PE queues. Hence, before the adder-tree reads again
  // from the same PE queue, kNumPEs cycles pass. This contrains the depth of
  // the queues to kNumPEs. (THIS WON'T WORK, TOO LOW CONSUMER RATE)
  // FIXED: Using an adder tree allows to use a stream of depth 1.
  const unsigned kStreamDepth = 1; // VectLength / NumTiles;

  hls::stream<svd::AccumD> acc1_streams[kNumNonZeroTiles];
  hls::stream<svd::AccumD> acc2_streams[kNumNonZeroTiles];
#pragma HLS ARRAY_PARTITION variable=acc1_streams complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc1_streams complete dim=1
#pragma HLS STREAM variable=acc1_streams depth=kStreamDepth
#pragma HLS STREAM variable=acc2_streams depth=kStreamDepth

  PE_Loop:
  for (int pe = 0; pe < kNumPEs; ++pe) {
#pragma HLS UNROLL
    UDotUnit2LstmPe<svd::ActivationD, svd::WeightD, svd::AccumD>(VectLength,
      NumTiles, NumIter, NumTimesteps,
      x1_streams[pe], x2_streams[pe], gate_u_streams[pe], acc1_streams[pe],
      acc2_streams[pe]);
  }
  UDotUnit2LstmAccumulator<VectLength, NumTiles, NumZeroTiles, NumIter, NumTimesteps>(
    acc1_streams, acc2_streams, y1, y2);
#endif // end REDUCE_PROD_2LSTM_DATAFLOW_DESIGN
}

#ifdef __VITIS_HLS__
template <typename params>
void KernelU(const int num_active_inputs,
  const int input_size,
  const hls::vector<int, params::N> num_refinements,
  const bool pad_output,
  hls::stream<typename params::VectTuAxiType>& x_port,
  hls::stream<typename params::VectTuAxiType>& u_port,
  hls::stream<typename params::VectG_AxiType>& xu_port) {
  /*
   Notes: Add a parameter for ignoring processing an input, effectively
   utilizing the full and single pipeline for only a few inputs.
  */
#pragma HLS INTERFACE axis port=x_port
#pragma HLS INTERFACE axis port=u_port
#pragma HLS INTERFACE axis port=xu_port
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE s_axilite port=pad_output
#pragma HLS INTERFACE s_axilite port=num_refinements
#pragma HLS DATAFLOW
  assert(num_active_inputs <= params::N);
  assert(num_refinements >= 1);
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

  auto x_axis = svd::AxiStreamInterface<params::VectTuAxiWidth>(x_port);
  auto u_axis = svd::AxiStreamInterface<params::VectTuAxiWidth>(u_port);
  auto xu_axis = svd::AxiStreamInterface<params::VectG_AxiWidth>(xu_port);

  hls::stream<typename params::VectTuType> x_stream("x_stream");
  hls::stream<typename params::VectTuType> u_streams[params::G];
  hls::stream<ActivationType> xu_streams[params::G];
  typename params::VectTuType x_buffer[params::N][kMaxNumTilesU] = {0};
#pragma HLS STREAM variable=x_stream depth=kStreamDepth_X
#pragma HLS STREAM variable=u_streams depth=kStreamDepth_U
#pragma HLS STREAM variable=xu_streams depth=kStreamDepth_XU
#pragma HLS ARRAY_PARTITION variable=u_streams complete dim=1
#pragma HLS ARRAY_PARTITION variable=x_buffer complete dim=1
#pragma HLS BIND_STORAGE variable=x_buffer type=ram_t2p impl=bram latency=2
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
#pragma HLS PIPELINE II=1
    if (num_refinements[i] > R_max) {
      R_max = num_refinements[i];
    }
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
#pragma HLS PIPELINE II=1
          if (ii == 0 && i == 0) {
            auto x_val = x_axis.template PopVector<ActivationType, params::Tu>();
// #pragma HLS AGGREGATE variable=x_val
            x_buffer[k][j] = x_val;
            x_stream << x_val;
          } else {
            assert(k + ii < params::N);
            x_stream << x_buffer[k + ii][j];
          }
          // std::cout << "\t[KernelU] Sending x[R." << i+R_prev << "][N." << k+ii << "][T." << j << "]" << std::endl;
        }
      }
    }
    R_prev = num_refinements[ii];
  }
  // std::cout << std::endl;

  U_DMA:
  for (int i = 0; i < R_max; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=params::R max=params::R
    for (int j = 0; j < kNumTilesU; ++j) {
      for (int k = 0; k < params::G; ++k) {
        auto u_val = u_axis.template PopVector<ActivationType, params::Tu>();
        for (int ii = 0; ii < num_active_inputs; ++ii) {
#pragma HLS PIPELINE II=1
          if (i < num_refinements[ii]) {
            u_streams[k] << u_val;
            // std::cout << "\t[KernelU] Sending u[R." << i << "][G." << k << "][N." << ii << "][T." << j << "]" << std::endl;
          }
        }
      }
    }
  }
  // std::cout << std::endl;

  U_Kernel:
  for (int i = 0; i < R_total; ++i) {
    for (int j = 0; j < kNumTilesU; ++j) {
#pragma HLS PIPELINE II=1
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
#pragma HLS PIPELINE II=1
        for (int ii = 0; ii < params::G; ++ii) {
          if (i < num_refinements[k]) {
            xu_out[k][ii] += xu_streams[ii].read();
#pragma HLS BIND_OP variable=xu_out[k][ii] op=add impl=dsp
            // std::cout << "\t[KernelU] Reading xu[R." << i << "][G." << ii << "][N." << k << "][T." << j << "]" << std::endl;
          }
        }
        if (i < num_refinements[k] && j == kNumTilesU - 1) {
          const bool kIsLast = (iter_cnt == R_total - 1 && !pad_output) ? true : false;
          xu_axis.template PushVector<ActivationType, params::G>(xu_out[k], kIsLast);
          // std::cout << "\t[KernelU] Sending xu[R." << i << "][N." << k << "]" << std::endl;
          ++iter_cnt;
        } else if (pad_output) {
          const bool last_condition = i == R_max - 1 && j == kNumTilesU - 1 && k == num_active_inputs - 1; 
          const bool kIsLast = (last_condition) ? true : false;
          xu_axis.template PushVector<ActivationType, params::G>(xu_out[k], kIsLast);
          ++iter_cnt;
        }
      }
    }
  }
  // std::cout << "[KernelU] iter_cnt: " << iter_cnt << std::endl;
}
#endif

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
typedef hls::vector<typename params::ActivationD, params::Tu> VectTuType;
typedef hls::vector<typename params::ActivationD, params::N> VectN_Type;
typedef hls::vector<typename params::ActivationD, params::G * params::N> VectGN_Type;
typedef svd::AxiStreamInterface<VectTuAxiBitwidth>::AxiuPacketType VectTuAxiType;
typedef svd::AxiStreamInterface<VectN_AxiBitwidth>::AxiuPacketType VectN_AxiType;
typedef svd::AxiStreamInterface<VectGN_AxiBitwidth>::AxiuPacketType VectGN_AxiType;

} // testu

#ifndef __VITIS_HLS__
void HlsKernelU(const int num_refinements,
  const typename testu::params::ActivationD x_port[testu::params::N][testu::params::I],
  const typename testu::params::UPortD u_port[testu::params::R * testu::params::PrunedSizeU],
  typename testu::params::ActivationD xu_port[testu::params::N][testu::params::G * testu::params::R]);
#else
void HlsKernelU(const int num_refinements,
  hls::vector<typename testu::params::ActivationD, testu::params::N>* x_port,
  hls::vector<typename testu::params::ActivationD, testu::params::G>* u_port,
  hls::vector<typename testu::params::ActivationD, testu::params::N>* xu_port);

void HlsVectorKernelU(const int num_refinements,
  hls::stream<hls::vector<typename testu::params::ActivationD, testu::params::Tu> >& x_port,
  hls::stream<hls::vector<typename testu::params::ActivationD, testu::params::Tu> >& u_port,
  hls::stream<hls::vector<typename testu::params::ActivationD, testu::params::N> >& xu_port);

void HlsAxisKernelU(const int num_refinements,
  hls::stream<typename testu::VectTuAxiType>& x_port,
  hls::stream<typename testu::VectTuAxiType>& u_port,
  hls::stream<typename testu::VectGN_AxiType>& xu_port);

/**
 * @brief      Flexible Kernel-U.
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
void HlsKernelU_ManySampling(const int num_active_inputs,
  const int input_size,
  const hls::vector<int, testu::params::N> num_refinements,
  const bool pad_output,
  hls::stream<typename testu::params::VectTuAxiType>& x_port,
  hls::stream<typename testu::params::VectTuAxiType>& u_port,
  hls::stream<typename testu::params::VectG_AxiType>& xu_port);

#endif // end __VITIS_HLS__

#endif // end KERNEL_U_KERNEL_H_