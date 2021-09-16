#ifndef DMA_SVD_DMA_H_
#define DMA_SVD_DMA_H_

#include "svd_params.h"
#include "hls_utils/hls_metaprogramming.h"
#include "hls_utils/priority_encoder.h"
#include "dma/width_converter.h"
#include "dma/axis_lib.h"

#include "hls_stream.h"
#include "assert.h"

#include <iostream>

namespace svd {

template <typename Din, typename Dout>
void StreamSplitter(const int output_size,
    const Din *x,
    hls::stream<Dout> (&y)[hlsutils::Bitwidth<Din>::value / hlsutils::Bitwidth<Dout>::value]) {
#pragma HLS ARRAY_PARTITION variable=y complete dim=1
  const int kDivider = hlsutils::Bitwidth<Din>::value / hlsutils::Bitwidth<Dout>::value;
  const int kInputSize = output_size / kDivider;
  assert(hlsutils::Bitwidth<Din>::value % hlsutils::Bitwidth<Dout>::value == 0);
  assert(hlsutils::Bitwidth<Din>::value >= hlsutils::Bitwidth<Dout>::value);
  assert(output_size % kDivider == 0);
  DMA_Loop:
  for (int i = 0; i < kInputSize; ++i) {
#pragma HLS PIPELINE II=1
    Parallel_Write_Loop:
    for (int j = 0; j < kDivider; ++j) {
      const int kHi = (j + 1) * hlsutils::Bitwidth<Dout>::value - 1;
      const int kLo = j * hlsutils::Bitwidth<Dout>::value;
      ap_uint<hlsutils::Bitwidth<Dout>::value> x_val = x[i].range(kHi, kLo);
      y[j].write(*((Dout*)&x_val));
    }
  }
}

template <typename params>
void NzDMA(const typename params::UnzD nz_u_port[params::R * params::G],
           const typename params::VnzD nz_v_port[params::R * params::G],
           svd::SvdStreams<params> &streams) {
  Nz_DMA:
  for (int i = 0; i < params::R; ++i) {
    for (int j = 0; j < params::G; ++j) {
#pragma HLS PIPELINE II=1
      streams.nz_v[j].write(nz_v_port[i * params::G + j]);
      streams.nz_u[j].write(nz_u_port[i * params::G + j]);
    }
  }
}


template <typename params>
void S_DMA(const typename params::SPortD s_port[params::N][params::R],
           svd::SvdStreams<params> &streams) {
  S_DMA:
  for (int i = 0; i < params::N; ++i) {
#pragma HLS UNROLL
    StreamSplitter(params::G * params::R, s_port[i], streams.s[i]);
  }
}

template <typename params>
void U_Dispatcher(const typename params::UPortD u_port[params::R * params::PrunedSizeU],
                  svd::SvdStreams<params> &streams) {
  U_Dispatcher:
  for (int i = 0; i < params::R; ++i) {
    for (int j = 0; j < params::PeU; ++j) {
      for (int k = 0; k < params::PrunedSizeU / params::PeU; ++k) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN
        for (int g = 0; g < params::G; ++g) {
          streams.u[g][j].write(streams.u_dma[g].read());
        }
      }
    }
  }
}

template <typename params>
void NzIdxConverter(svd::SvdStreams<params> &streams) {
  typename params::UnzD nz_idx[params::G];
#pragma HLS ARRAY_PARTITION variable=nz_idx complete
  NZ_to_Idx:
  for (int i = 0; i < params::R; ++i) {
    for (int j = 0; j < params::PeU; ++j) {
#pragma HLS PIPELINE II=1
      for (int k = 0; k < params::G; ++k) {
        if (j == 0) {
          nz_idx[k] = streams.nz_u[k].read();
        }
        int set_idx = 0;
        for (int i = params::TuBits - 1; i >= 0; --i) {
          if (nz_idx[k][i] == 1) {
          set_idx = i;
          }
        }
        assert(set_idx < params::Tu);
        for (int ii = 0; ii < params::N; ++ii) {
          streams.tile_idx_stream[ii][k][j].write(set_idx);
        }
        nz_idx[k][set_idx] = 0;
      }
    }
  }
}

template <typename params>
void InputDMA(const int num_refinements,
    const typename params::ActivationD x_port[params::N][params::I],
    svd::SvdStreams<params> &streams,
    svd::SvdBuffers<params> &buffers) {
#pragma HLS INLINE
  typename params::UnzIdxD tile_idx[params::N][params::G][params::PeU];
#pragma HLS ARRAY_PARTITION variable=tile_idx complete dim=0
  Store_X_Buffer:
  for (int i = 0; i < params::Tu; ++i) {
    for (int j = 0; j < params::I / params::Tu; ++j) {
#pragma HLS PIPELINE II=1
      for (int k = 0; k < params::N; ++k) {
        buffers.x[k][i][j] = x_port[k][i * params::I / params::Tu + j];
      }
    }
  }

  // TODO: The Stream_X_Tiles loop is correctly unrolled, but the iterations
  // DON'T run in parallel.
  Stream_X_Tiles:
  for (int ii = 0; ii < params::N; ++ii) {
#pragma HLS UNROLL
    Stream_X_Tiles_inner:
    for (int i = 0; i < num_refinements; ++i) {
      for (int k = 0; k < params::I / params::Tu; ++k) {
  #pragma HLS PIPELINE II=1
        for (int j = 0; j < params::PeU; ++j) {
          for (int g = 0; g < params::G; ++g) {
            if (params::ZTu > 0) { // constexpr
              if (k == 0) {
                tile_idx[ii][g][j] = streams.tile_idx_stream[ii][g][j].read();
              }
              streams.x[ii][g][j].write(buffers.x[ii][tile_idx[ii][g][j]][k]);
            } else {
              streams.x[ii][g][j].write(buffers.x[ii][j][k]);
            }
          }
        }
      }
    }
  }
}

template <typename params>
void V_Dispatcher(const typename params::VPortD v_port[params::R * params::PrunedSizeV],
                  svd::SvdStreams<params> &streams) {
  V_Dispatcher:
  for (int i = 0; i < params::R; ++i) {
    for (int j = 0; j < params::PeV; ++j) {
      for (int k = 0; k < params::PrunedSizeV / params::PeV; ++k) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN
        for (int g = 0; g < params::G; ++g) {
          streams.v[g][j].write(streams.v_dma[g].read());
        }
      }
    }
  }
}

template <typename params>
void SvdInDMA(
    const typename params::ActivationD x_port[params::N][params::I],
    const typename params::UPortD u_port[params::R * params::PrunedSizeU],
    const typename params::SPortD s_port[params::N][params::R],
    const typename params::VPortD v_port[params::R * params::PrunedSizeV],
    const typename params::UnzD nz_u_port[params::R * params::G],
    const typename params::VnzD nz_v_port[params::R * params::G],
    svd::SvdStreams<params> &streams,
    svd::SvdBuffers<params> &buffers) {
#pragma HLS INLINE
#ifndef __VITIS_HLS__
#pragma HLS DATAFLOW
#endif
  S_DMA<params>(s_port, streams);
  U_DMA: StreamSplitter(params::G * params::R * params::PrunedSizeU, u_port, streams.u_dma);
  V_DMA: StreamSplitter(params::G * params::R * params::PrunedSizeV, v_port, streams.v_dma);
  if (params::ZTu > 0) {
    NzDMA<params>(nz_u_port, nz_v_port, streams);
    NzIdxConverter<params>(streams);
  }
  U_Dispatcher<params>(u_port, streams);
  V_Dispatcher<params>(v_port, streams);
  InputDMA<params>(params::R, x_port, streams, buffers);
}

template <typename params>
void SvdOutDMA(
    svd::SvdStreams<params> &streams,
    typename params::ActivationD y_port[params::N][params::G][params::H]) {
  for (int i = 0; i < params::H / params::PeV; ++i) {
    for (int j = 0; j < params::PeV; ++j) {
      for (int k = 0; k < params::N; ++k) {
        for (int g = 0; g < params::G; ++g) {
#pragma HLS PIPELINE II=1
          y_port[k][g][i * params::PeV + j] = streams.xusv[k][g][j].read();
        }
      }
    }
  }
}

template <int NumIter, int NumTiles, int NumGates>
void NZIndex2LstmDMA(const ap_uint<NumTiles> *nz_port,
    hls::stream<ap_uint<NumTiles> > (&nz_stream1_cur)[NumGates / 2],
    hls::stream<ap_uint<NumTiles> > (&nz_stream1_rec)[NumGates / 2],
    hls::stream<ap_uint<NumTiles> > (&nz_stream2_cur)[NumGates / 2],
    hls::stream<ap_uint<NumTiles> > (&nz_stream2_rec)[NumGates / 2]) {
#pragma HLS INLINE
  assert(NumGates % 2 == 0);
  assert(NumTiles % 2 == 0);
  // assert(NumTiles >= 8); // Minimum port size requirement.
  NZIndex_Dma_Iter_Loop:
  for (int i = 0; i < NumIter; ++i) {
#pragma HLS PIPELINE II=1
    NZIndex_Dma_Current_Loop:
    for (int g = 0; g < NumGates / 2; ++g) {
      ap_uint<NumTiles> nz_idx = nz_port[i * NumGates + g];
      nz_stream1_cur[g].write(nz_idx);
      nz_stream2_cur[g].write(nz_idx);
    }
    NZIndex_Dma_Recur_Loop:
    for (int g = 0; g < NumGates / 2; ++g) {
      ap_uint<NumTiles> nz_idx = nz_port[i * NumGates + NumGates / 2 + g];
      nz_stream1_rec[g].write(nz_idx);
      nz_stream2_rec[g].write(nz_idx);
    }
  }
}

template <int NumIter, int NumTiles, int NumGates>
void NZIndexDMA(const ap_uint<NumTiles> *nz_port,
    hls::stream<ap_uint<NumTiles> > (&cur_nz_stream)[NumGates / 2],
    hls::stream<ap_uint<NumTiles> > (&rec_nz_stream)[NumGates / 2]) {
#pragma HLS INLINE
  assert(NumGates % 2 == 0);
  assert(NumTiles % 2 == 0);
  // assert(NumTiles >= 8); // Minimum port size requirement.
  NZIndex_Dma_Iter_Loop:
  for (int i = 0; i < NumIter; ++i) {
#pragma HLS PIPELINE II=1
    NZIndex_Dma_Current_Loop:
    for (int g = 0; g < NumGates / 2; ++g) {
      cur_nz_stream[g].write(nz_port[i * NumGates + g]);
    }
    NZIndex_Dma_Recur_Loop:
    for (int g = 0; g < NumGates / 2; ++g) {
      rec_nz_stream[g].write(nz_port[i * NumGates + NumGates / 2 + g]);
    }
  }
}

/**
 * @brief      Dispatches an LSTM input to the proper MAC units of the U-Unit.
 *
 * @param[in]  x_dmem        The port from which to read the data from
 * @param      comb_stream   The stream of non-zero tiles
 * @param      x_streams     The output streams
 *
 * @tparam     VectLength    The LSTM input size
 * @tparam     NumTiles      The number of tiles
 * @tparam     NumZeroTiles  The number of pruned tiles
 * @tparam     NumGates      The number of LSTM gates (usually 4)
 * @tparam     NumIter       The number of refinement steps
 */
template <int VectLength, int NumTiles, int NumZeroTiles, int NumGates, int NumIter>
void InputDMA(const svd::ActivationD *x_dmem,
    hls::stream<ap_uint<NumTiles> > *comb_stream,
    svd::ActivationStream (&x_streams)[NumGates][NumTiles-NumZeroTiles]) {
  assert(VectLength % NumTiles == 0);
  assert(NumTiles - NumZeroTiles > 0);
  // ===========================================================================
  // Store the input onto an on-chip buffer for data reuse. The buffer is shared
  // by the LSTM gates and their U-units (which contain T - ZT MAC units each).
  // ===========================================================================
#ifdef __VITIS_HLS__
#pragma HLS INLINE
#endif
#pragma HLS DATAFLOW
  const int kTileSize = VectLength / NumTiles;
  const int kNumPEs = NumTiles - NumZeroTiles;
  svd::ActivationD x_buffer[NumTiles][kTileSize];
#pragma HLS ARRAY_PARTITION variable=x_buffer complete dim=1
  Write_Buffer:
  for (int i = 0; i < NumTiles; ++i) {
    for (int j = 0; j < kTileSize; ++j) {
#pragma HLS PIPELINE II=1
      x_buffer[i][j] = x_dmem[i * kTileSize + j];
    }
  }
  hls::stream<ap_uint<hlsutils::log2<NumTiles>::value> > tile_idx_stream[NumGates][kNumPEs];
#pragma HLS ARRAY_PARTITION variable=tile_idx_stream complete dim=0
  NZ_to_Idx:
  for (int i = 0; i < NumIter; ++i) {
    ap_uint<NumTiles> nz_idx[NumGates];
#pragma HLS ARRAY_PARTITION variable=nz_idx complete
    for (int j = 0; j < kNumPEs; ++j) {
#pragma HLS PIPELINE II=1
      for (int k = 0; k < NumGates; ++k) {
        if (j == 0) {
          nz_idx[k] = comb_stream[k].read();
        }
        int set_idx = hlsutils::PriorityEncoderLSB<NumTiles>(nz_idx[k]);
        assert(set_idx < NumTiles);
        tile_idx_stream[k][j].write(set_idx);
        nz_idx[k][set_idx] = 0;
      }
    }
  }
  Stream_Tiles:
  for (int i = 0; i < NumIter; ++i) {
    for (int k = 0; k < kTileSize; ++k) {
#pragma HLS PIPELINE II=1
      ap_uint<hlsutils::log2<NumTiles>::value> tile_idx[NumGates][kNumPEs];
#pragma HLS ARRAY_PARTITION variable=tile_idx complete dim=0
      for (int j = 0; j < kNumPEs; ++j) {
        for (int g = 0; g < NumGates; ++g) {
          if (k == 0) {
            tile_idx[g][j] = tile_idx_stream[g][j].read();
          }
          x_streams[g][j].write(x_buffer[tile_idx[g][j]][k]);
        }
      }
    }
  }
}

template <typename Din, typename Dout, int InWidth, int OutWidth, int OutputSize>
void ArraySplitter(const Din *x,
    Dout (&y)[InWidth / OutWidth][OutputSize / (InWidth / OutWidth)]) {
#pragma HLS INLINE
#ifndef __VITIS_HLS__
#pragma HLS ARRAY_PARTITION variable=y complete dim=1
#endif
  const int kDivider = InWidth / OutWidth;
  const int kInputSize = OutputSize / kDivider;
  assert(InWidth % OutWidth == 0);
  assert(InWidth >= OutWidth);
  assert(OutputSize % kDivider == 0);
  DMA_Loop:
  for (int i = 0; i < kInputSize; ++i) {
#pragma HLS PIPELINE II=1
    Parallel_Write_Loop:
    for (int j = 0; j < kDivider; ++j) {
      const int kHi = (j + 1) * OutWidth - 1;
      const int kLo = j * OutWidth;
      ap_uint<OutWidth> x_val = x[i].range(kHi, kLo);
      y[j][i] = *((Dout*)&x_val);
    }
  }
}

template <typename Din, typename Dout, int InWidth, int OutWidth>
void StreamSplitter(const int output_size,
    const Din *x,
    hls::stream<Dout> (&y)[InWidth / OutWidth]) {
#pragma HLS INLINE
#ifndef __VITIS_HLS__
#pragma HLS ARRAY_PARTITION variable=y complete dim=1
#endif
  const int kDivider = InWidth / OutWidth;
  const int kInputSize = output_size / kDivider;
  assert(InWidth % OutWidth == 0);
  assert(InWidth >= OutWidth);
  assert(output_size % kDivider == 0);
  DMA_Loop:
  for (int i = 0; i < kInputSize; ++i) {
#pragma HLS PIPELINE II=1
    Parallel_Write_Loop:
    for (int j = 0; j < kDivider; ++j) {
      const int kHi = (j + 1) * OutWidth - 1;
      const int kLo = j * OutWidth;
      ap_uint<OutWidth> x_val = x[i].range(kHi, kLo);
      y[j].write(*((Dout*)&x_val));
    }
  }
}

/**
 * @brief      Dispatch weight values to PEs. Work on one port for one LSTM
 *             gate. The gate has shape (I, NZ, E).
 *
 * @param[in]  use_nz_dim       If true, there are #num_non_zero_tiles different
 *                              PEs (for the U-unit), else there are
 *                              #num_elems_per_tile different PEs (for the V-unit).
 * @param[in]  gate_port        The gate port
 * @param      gate_stream      The gate PEs stream
 *
 * @tparam     NumIter          Number of refinement steps.
 * @tparam     num_non_zero_tiles  Number of non pruned tiles.
 * @tparam     num_elems_per_tile  Number of elements per tile.
 */
template <typename T>
void DispatchGateFromArray(const bool use_nz_dim, const int num_iter,
    const int num_non_zero_tiles, const int num_elems_per_tile,
    const T* gate_port, hls::stream<T>* gate_streams) {
#pragma HLS INLINE
#pragma HLS FUNCTION_INSTANTIATE variable=num_iter
#pragma HLS FUNCTION_INSTANTIATE variable=num_non_zero_tiles
#pragma HLS FUNCTION_INSTANTIATE variable=num_elems_per_tile
  const int kI = num_iter;
  const int kNZ = num_non_zero_tiles;
  const int kE = num_elems_per_tile;
  I : for (int i = 0; i < kI; ++i) {
    Z : for (int z = 0; z < kNZ; ++z) {
      E : for (int e = 0; e < kE; ++e) {
#pragma HLS PIPELINE II=1
        const int g_idx = i * kNZ * kE + z * kE + e;
        if (use_nz_dim) {
          gate_streams[z].write(gate_port[g_idx]); // for U weights
        } else {
          gate_streams[e].write(gate_port[g_idx]); // for V weights
        }
      }
    }
  }
}

template <typename T>
void DispatchGateFromStream(const bool use_nz_dim, const int num_iter,
    const int num_non_zero_tiles, const int num_elems_per_tile,
    hls::stream<T>& gate_port, hls::stream<T>* gate_streams) {
#pragma HLS INLINE
#pragma HLS FUNCTION_INSTANTIATE variable=num_iter
#pragma HLS FUNCTION_INSTANTIATE variable=num_non_zero_tiles
#pragma HLS FUNCTION_INSTANTIATE variable=num_elems_per_tile
  const int kI = num_iter;
  const int kNZ = num_non_zero_tiles;
  const int kE = num_elems_per_tile;
  I : for (int i = 0; i < kI; ++i) {
    Z : for (int z = 0; z < kNZ; ++z) {
      E : for (int e = 0; e < kE; ++e) {
#pragma HLS PIPELINE II=1
        if (use_nz_dim) {
          gate_streams[z].write(gate_port.read()); // for U weights
        } else {
          gate_streams[e].write(gate_port.read()); // for V weights
        }
      }
    }
  }
}

/**
 * @brief      Dispatch input elements to several PEs. All PEs receive the same
 *             amount of elements and in the same clock cycle.
 *
 * @param[in]  input_size  The input size, i.e. amount of elements in the input
 *                         stream
 * @param      x           The input stream
 * @param      y           The output stream
 * @param[in]  verbose     If true, print the state of the iternal registers.
 *                         Default: 0
 *
 * @tparam     BitWidth    The number of bits used by the stream types
 * @tparam     NumPE       The number of output PEs
 */
template <int BitWidth, int NumPE>
void PipelinedDispatcher(const int input_size,
                         hls::stream<ap_uint<BitWidth> > &x,
                         hls::stream<ap_uint<BitWidth> > y[NumPE],
                         const int verbose = 0) {
  ap_uint<BitWidth> shift_reg[NumPE][NumPE];
#pragma HLS ARRAY_PARTITION variable=shift_reg complete dim=0
#ifndef __SYNTHESIS__
  for (int i = 0; i < NumPE; ++i) {
    for (int j = 0; j < NumPE; ++j) {
      shift_reg[i][j] = 0;
    }
  }
#endif

  for (int i = 0; i < input_size; ++i) {
#pragma HLS PIPELINE II=1
    ap_uint<BitWidth> x_elem = x.read();
    shift_reg[i % NumPE][i % NumPE] = x_elem;

    for (int j = NumPE - 1; j >= 1; --j) {
      for (int k = j - 1; k >= 0; --k) {
        shift_reg[k][j] = shift_reg[k][j - 1];
      }
    }
#ifndef __SYNTHESIS__
    if (verbose) {
      // Print out
      for (int j = 0; j < NumPE; j++) {
        for (int k = 0; k < NumPE; k++) {
          if (j > k) {
            std::cout << "0" << "\t";
          } else {
            std::cout << shift_reg[j][k] << "\t";
          }
        }
        std::cout << "\n";
      }
    }
#endif
    if (i % NumPE == 0 && i > 0) {
      for (int j = 0; j < NumPE; ++j) {
        y[j].write(shift_reg[j][NumPE - 1]);
      }
    }
  }
  for (int j = 0; j < NumPE; ++j) {
#pragma HLS UNROLL
    y[j].write(shift_reg[j][NumPE - 1]);
  }
}

template <int BitWidth, int NumPE>
void PipelinedDispatcher(const int input_size,
                         const ap_uint<BitWidth> *x,
                         hls::stream<ap_uint<BitWidth> > y[NumPE],
                         const int verbose = 0) {
  ap_uint<BitWidth> shift_reg[NumPE][NumPE];
#pragma HLS ARRAY_PARTITION variable=shift_reg complete dim=0
#ifndef __SYNTHESIS__
  for (int i = 0; i < NumPE; ++i) {
    for (int j = 0; j < NumPE; ++j) {
      shift_reg[i][j] = 0;
    }
  }
#endif

  for (int i = 0; i < input_size; ++i) {
#pragma HLS PIPELINE II=1
    ap_uint<BitWidth> x_elem = x[i];
    shift_reg[i % NumPE][i % NumPE] = x_elem;

    for (int j = NumPE - 1; j >= 1; --j) {
      for (int k = j - 1; k >= 0; --k) {
        shift_reg[k][j] = shift_reg[k][j - 1];
      }
    }
#ifndef __SYNTHESIS__
    if (verbose) {
      // Print out
      for (int j = 0; j < NumPE; j++) {
        for (int k = 0; k < NumPE; k++) {
          if (j > k) {
            std::cout << "0" << "\t";
          } else {
            std::cout << shift_reg[j][k] << "\t";
          }
        }
        std::cout << "\n";
      }
    }
#endif
    if (i % NumPE == 0 && i > 0) {
      for (int j = 0; j < NumPE; ++j) {
        y[j].write(shift_reg[j][NumPE - 1]);
      }
    }
  }
  for (int j = 0; j < NumPE; ++j) {
#pragma HLS UNROLL
    y[j].write(shift_reg[j][NumPE - 1]);
  }
}


#ifdef __VITIS_HLS__
template <typename params>
void VectorizedInputDMA(const int R,
    hls::stream<typename params::VectTuAxiPacketType>& x_port,
    hls::stream<typename params::VectTuType> x_streams[params::N]) {

  typedef typename params::ActivationD ActivationType;
  const int kNumTilesU = params::I / params::Tu;
  svd::AxiStreamPort<params::VectTuAxiWidth> x_axis = svd::AxiStreamPort<params::VectTuAxiWidth>(x_port);
  typename params::VectTuType x_buffer[params::N][kNumTilesU];
#pragma HLS ARRAY_PARTITION variable=x_buffer complete dim=1

  Store_X_Buffer:
  for (int i = 0; i < params::N; ++i) {
    for (int j = 0; j < kNumTilesU; ++j) {
#pragma HLS LOOP_FLATTEN
#pragma HLS PIPELINE II=1
      // x_buffer[i][j] = x_axis.PopVector<ActivationType, params::Tu>();
    }
  }
  Stream_X_Tiles:
  for (int i = 0; i < R; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=params::R max=params::R
    for (int j = 0; j < kNumTilesU; ++j) {
#pragma HLS PIPELINE II=1
      for (int k = 0; k < params::N; ++k) {
        x_streams[k] << x_buffer[k][j];
      }
    }
  }
}
#endif

} // end namespace svd

#endif // end DMA_SVD_DMA_H_