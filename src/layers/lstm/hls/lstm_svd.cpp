#include "layers/lstm/hls/lstm_svd.h"
#include "layers/dense/hls/dense_svd.h"
#include "svd_params.h"
#include "dma/svd_dma.h"
#include "kernel/u_kernel.h"
#include "kernel/s_kernel.h"
#include "kernel/v_kernel.h"
#include "math_utils/activation_functions.h"
#include "hls_utils/hls_debugging.h"

#include "hls_stream.h"
#include "ap_int.h"
#include "assert.h"

#include <string>

namespace svd {

void SvdModel2LstmSDSoCV2(
    const svd::ActivationD x1_port[INPUT_SIZE],
    const svd::ActivationD x2_port[INPUT_SIZE],
    const svd::ActivationD h_t1_prev_port[HIDDEN_SIZE],
    const svd::ActivationD h_t2_prev_port[HIDDEN_SIZE],
    const svd::ActivationD c_t1_prev_port[HIDDEN_SIZE],
    const svd::ActivationD c_t2_prev_port[HIDDEN_SIZE],
    const ap_uint<FIX_WIDTH * 4> *u_cur_port, // [NUM_ITERATIONS*4*INPUT_SIZE / NUM_TILES_U * (NUM_TILES_U - NUM_ZERO_TILES_U)],
    const ap_uint<FIX_WIDTH * 4> *u_rec_port, // [NUM_ITERATIONS*4*HIDDEN_SIZE / NUM_TILES_U * (NUM_TILES_U - NUM_ZERO_TILES_U)],
    const ap_uint<FIX_WIDTH * 8> *v_port, // [NUM_ITERATIONS*4*2*HIDDEN_SIZE / NUM_TILES_V * (NUM_TILES_V - NUM_ZERO_TILES_V)],
    const ap_uint<FIX_WIDTH * 8> *s1_port, // [NUM_ITERATIONS*8],
    const ap_uint<FIX_WIDTH * 8> *s2_port, // [NUM_ITERATIONS*8],
    const svd::WeightD bias1_port[4 * HIDDEN_SIZE],
    const svd::WeightD bias2_port[4 * HIDDEN_SIZE],
    const ap_uint<NUM_TILES_V> nz_v_port[NUM_ITERATIONS * 8],
    const ap_uint<NUM_TILES_U> nz_u_port[NUM_ITERATIONS * 8],
    svd::ActivationD h_t1_curr_port[HIDDEN_SIZE],
    svd::ActivationD h_t2_curr_port[HIDDEN_SIZE],
    svd::ActivationD c_t1_curr_port[HIDDEN_SIZE],
    svd::ActivationD c_t2_curr_port[HIDDEN_SIZE]
#ifdef DEBUG_FIFOS
    ,
    svd::CounterD *counters_port,
    svd::CounterD *clk_count_port
#endif
    ) {
  hlsutils::Log(0, "[INFO] Running SvdModel2LstmSDSoCV2.");
  const int kNumGates = 8;
  const int kNumCurGates = 4;
  const int kNumRecGates = 4;
  const int kInputLength = INPUT_SIZE;
  const int kOutputLength = HIDDEN_SIZE;
  const int kNumTilesU = NUM_TILES_U;
  const int kNumTilesV = NUM_TILES_V;
  const int kNumZeroTilesU = NUM_ZERO_TILES_U;
  const int kNumZeroTilesV = NUM_ZERO_TILES_V;
  const int kNumIter = NUM_ITERATIONS;
  const int kNumTimesteps = NUM_TIMESTEPS;
  const int kNumNonZeroTilesU = kNumTilesU - kNumZeroTilesU;
  const int kNumNonZeroTilesV = kNumTilesV - kNumZeroTilesV;
  const int kTileSizeUCurrent = kInputLength / kNumTilesU;
  const int kTileSizeURecur = kOutputLength / kNumTilesU;
  const int kTileSizeV = kOutputLength / kNumTilesV;
  assert(kNumTilesU % 2 == 0);
  assert(kNumTilesV % 2 == 0);
  // assert(kNumZeroTilesU % 2 == 0);
  // assert(kNumZeroTilesV % 2 == 0);
  // assert(kNumIter % 2 == 0);
  hlsutils::Log(0, "[INFO] asserts passed.");

  const int kTileSizeU = kInputLength / kNumTilesU;
  const int kPrunedLengthU = kInputLength - kNumZeroTilesU * kTileSizeU;
  const int kPrunedLengthV = kOutputLength - kNumZeroTilesV * kTileSizeV;
  const int kInputLengthPruned = kInputLength - kTileSizeU * kNumZeroTilesU;
  const int kOutputLengthPrunedU = kOutputLength - kOutputLength / kNumTilesU * kNumZeroTilesU;
  const int kOutputLengthPrunedV = kOutputLength - kOutputLength / kNumTilesV * kNumZeroTilesV;
  const int kNumSamples = NUM_SAMPLES; // Used for cosimulation only
  const int kNumReadsR = 8 * kNumIter;
  const int kNumReadsC = 8 * kNumIter;
  const int kAxiDepthR = kInputLength;
  const int kAxiPortDepthX = (kInputLength * kNumTimesteps) * kNumSamples;
  const int kAxiDepthU = (kNumIter * 8 * kPrunedLengthU) * kNumSamples;
  const int kAxiDepthV = (kNumIter * 8 * kPrunedLengthV) * kNumSamples;
  const int kAxiDepthS = (kNumIter * 8 * 2) * kNumSamples;
  const int kAxiDepthCombinationsR = kNumReadsR * kNumSamples;
  const int kAxiDepthCombinationsC = kNumReadsC * kNumSamples;
#ifndef SDS_DESIGN
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl

  const int kUSize = kNumIter*(kNumCurGates * kInputLengthPruned + kNumRecGates * kOutputLengthPrunedU);
  const int kVSize = kNumIter*(kNumCurGates * kOutputLengthPrunedV + kNumRecGates * kOutputLengthPrunedV);
  const int kSSize = kNumIter * 2 * (kNumCurGates + kNumRecGates);

  const int kUCurSize = kNumIter * kNumCurGates * kInputLengthPruned;
  const int kURecSize = kNumIter * kNumRecGates * kOutputLengthPrunedU;

  const int kUcurPortDepth = kUCurSize;
  const int kUrecPortDepth = kURecSize;
  const int kVportDepth = kVSize;
  const int kS1portDepth = kSSize / 2;
  const int kS2portDepth = kSSize / 2;
// #pragma HLS INTERFACE m_axi port=u_cur_port offset=slave depth=kUcurPortDepth bundle=u_cur_dmem
// #pragma HLS INTERFACE m_axi port=u_rec_port offset=slave depth=kUrecPortDepth bundle=u_rec_dmem
// #pragma HLS INTERFACE m_axi port=v_port offset=slave depth=kVportDepth bundle=v_dmem
// #pragma HLS INTERFACE m_axi port=s1_port offset=slave depth=kS1portDepth bundle=s1_dmem
// #pragma HLS INTERFACE m_axi port=s2_port offset=slave depth=kS2portDepth bundle=s2_dmem
#pragma HLS INTERFACE axis port=u_cur_port
#pragma HLS INTERFACE axis port=u_rec_port
#pragma HLS INTERFACE axis port=v_port
#pragma HLS INTERFACE axis port=s1_port
#pragma HLS INTERFACE axis port=s2_port

#pragma HLS INTERFACE axis port=x1_port
#pragma HLS INTERFACE axis port=x2_port
#pragma HLS INTERFACE axis port=bias1_port
#pragma HLS INTERFACE axis port=bias2_port
#pragma HLS INTERFACE axis port=nz_v_port
#pragma HLS INTERFACE axis port=nz_u_port
#pragma HLS INTERFACE axis port=h_t1_prev_port
#pragma HLS INTERFACE axis port=h_t2_prev_port
#pragma HLS INTERFACE axis port=h_t1_curr_port
#pragma HLS INTERFACE axis port=h_t2_curr_port
#pragma HLS INTERFACE axis port=c_t1_prev_port
#pragma HLS INTERFACE axis port=c_t2_prev_port
#pragma HLS INTERFACE axis port=c_t1_curr_port
#pragma HLS INTERFACE axis port=c_t2_curr_port
#endif // SDS_DESIGN

#pragma HLS DATAFLOW
  // ===========================================================================
  // Streams Depth Sizing
  // ===========================================================================
  // NOTE: We divide the FIFO depths by a certain factor to save BRAMs. Be aware
  // that a wrong factor could lead to deadlocks!
  const int kFIFOdepthDivider = 8;
  const int kStreamDepthIter = kNumIter / kFIFOdepthDivider;
  const int kFIFOdepthFactor = kNumIter * 2;
  const int kStreamDepthUCurrent = kNumIter * kTileSizeUCurrent / kFIFOdepthFactor == 0 ? 2 : kNumIter * kTileSizeUCurrent / kFIFOdepthFactor;
  const int kStreamDepthURecurrent = kNumIter * kTileSizeURecur / kFIFOdepthFactor == 0 ? 2 : kNumIter * kTileSizeURecur / kFIFOdepthFactor;
  const int kStreamDepthV = kNumIter * kNumTilesV / kFIFOdepthFactor == 0 ? 2 : kNumIter * kNumTilesV / kFIFOdepthFactor;
  const int kTileAccStreamDepth = 2;
  const int kOutStreamDepth = 2; // kNumIter * kTileSizeV;
  // ===========================================================================
  // Current streams
  // ===========================================================================
  svd::WeightStream cur_u_streams[kNumCurGates][kNumNonZeroTilesU];
  svd::WeightStream cur_v_streams[kNumCurGates][kTileSizeV]; // [kNumNonZeroTilesV];
  svd::ActivationStream cur_dot1_streams[kNumCurGates];
  svd::ActivationStream cur_dot2_streams[kNumCurGates];
  svd::ActivationStream cur_out1_streams[kNumCurGates][kNumNonZeroTilesV];
  svd::ActivationStream cur_out2_streams[kNumCurGates][kNumNonZeroTilesV];
  svd::ActivationStream cur_acc1_streams[kNumCurGates][kTileSizeV]; // [kNumTilesV];
  svd::ActivationStream cur_acc2_streams[kNumCurGates][kTileSizeV]; // [kNumTilesV];
  // ===========================================================================
  // Recur streams
  // ===========================================================================
  svd::WeightStream rec_u_streams[kNumRecGates][kNumNonZeroTilesU];
  svd::WeightStream rec_v_streams[kNumRecGates][kTileSizeV]; // [kNumNonZeroTilesV];
  svd::ActivationStream rec_dot1_streams[kNumRecGates];
  svd::ActivationStream rec_dot2_streams[kNumRecGates];
  svd::ActivationStream rec_out1_streams[kNumRecGates][kNumNonZeroTilesV];
  svd::ActivationStream rec_out2_streams[kNumRecGates][kNumNonZeroTilesV];
  svd::ActivationStream rec_acc1_streams[kNumRecGates][kTileSizeV]; // [kNumTilesV];
  svd::ActivationStream rec_acc2_streams[kNumRecGates][kTileSizeV]; // [kNumTilesV];
  // ===========================================================================
  // Scalar streams
  // ===========================================================================
  svd::WeightStream gates_s1_streams[kNumGates]; // used for both curr and recur
  svd::WeightStream gates_s2_streams[kNumGates]; // used for both curr and recur
  // ===========================================================================
  // Current input streams
  // ===========================================================================
  svd::ActivationStream x1_streams[kNumCurGates][kNumNonZeroTilesU];
  svd::ActivationStream x2_streams[kNumCurGates][kNumNonZeroTilesU];
  // ===========================================================================
  // Recurrent input streams
  // ===========================================================================
  svd::ActivationStream h1_streams[kNumRecGates][kNumNonZeroTilesU];
  svd::ActivationStream h2_streams[kNumRecGates][kNumNonZeroTilesU];
  // ===========================================================================
  // Zero Combinations DMA
  // ===========================================================================
  // NOTE: We divide the FIFO depths by a certain factor to save BRAMs. Be aware
  // that a wrong factor could lead to deadlocks!
  hls::stream<ap_uint<kNumTilesV> > nz_v_stream1_cur[kNumCurGates];
  hls::stream<ap_uint<kNumTilesV> > nz_v_stream1_rec[kNumRecGates];
  hls::stream<ap_uint<kNumTilesV> > nz_v_stream2_cur[kNumCurGates];
  hls::stream<ap_uint<kNumTilesV> > nz_v_stream2_rec[kNumRecGates];
  hls::stream<ap_uint<kNumTilesU> > nz_u_stream1_cur[kNumCurGates];
  hls::stream<ap_uint<kNumTilesU> > nz_u_stream1_rec[kNumRecGates];
  hls::stream<ap_uint<kNumTilesU> > nz_u_stream2_cur[kNumCurGates];
  hls::stream<ap_uint<kNumTilesU> > nz_u_stream2_rec[kNumRecGates];
#pragma HLS STREAM depth=kStreamDepthIter variable=nz_v_stream1_cur
#pragma HLS STREAM depth=kStreamDepthIter variable=nz_v_stream1_rec
#pragma HLS STREAM depth=kStreamDepthIter variable=nz_v_stream2_cur
#pragma HLS STREAM depth=kStreamDepthIter variable=nz_v_stream2_rec
#pragma HLS STREAM depth=kStreamDepthIter variable=nz_u_stream1_cur
#pragma HLS STREAM depth=kStreamDepthIter variable=nz_u_stream1_rec
#pragma HLS STREAM depth=kStreamDepthIter variable=nz_u_stream2_cur
#pragma HLS STREAM depth=kStreamDepthIter variable=nz_u_stream2_rec

#pragma HLS STREAM variable=x1_streams depth=kStreamDepthUCurrent // dim=2
#pragma HLS STREAM variable=x2_streams depth=kStreamDepthUCurrent // dim=2
#pragma HLS STREAM variable=h1_streams depth=kStreamDepthURecurrent // dim=2
#pragma HLS STREAM variable=h2_streams depth=kStreamDepthURecurrent // dim=2

#pragma HLS STREAM variable=cur_u_streams depth=kStreamDepthUCurrent // dim=2
#pragma HLS STREAM variable=rec_u_streams depth=kStreamDepthURecurrent // dim=2
#pragma HLS STREAM variable=cur_v_streams depth=kStreamDepthV // dim=2
#pragma HLS STREAM variable=rec_v_streams depth=kStreamDepthV // dim=2

#pragma HLS STREAM variable=gates_s1_streams depth=kStreamDepthIter
#pragma HLS STREAM variable=gates_s2_streams depth=kStreamDepthIter

#pragma HLS STREAM variable=cur_dot1_streams depth=kStreamDepthIter
#pragma HLS STREAM variable=cur_dot2_streams depth=kStreamDepthIter
#pragma HLS STREAM variable=rec_dot1_streams depth=kStreamDepthIter
#pragma HLS STREAM variable=rec_dot2_streams depth=kStreamDepthIter
#pragma HLS STREAM variable=cur_acc1_streams depth=kTileAccStreamDepth // dim=2
#pragma HLS STREAM variable=cur_acc2_streams depth=kTileAccStreamDepth // dim=2
#pragma HLS STREAM variable=rec_acc1_streams depth=kTileAccStreamDepth // dim=2
#pragma HLS STREAM variable=rec_acc2_streams depth=kTileAccStreamDepth // dim=2

#pragma HLS STREAM variable=cur_out1_streams depth=kOutStreamDepth // dim=2
#pragma HLS STREAM variable=cur_out2_streams depth=kOutStreamDepth // dim=2
#pragma HLS STREAM variable=rec_out1_streams depth=kOutStreamDepth // dim=2
#pragma HLS STREAM variable=rec_out2_streams depth=kOutStreamDepth // dim=2
  // ===========================================================================
  // Partitioning
  // ===========================================================================
#ifndef __VITIS_HLS__
#pragma HLS ARRAY_PARTITION variable=cur_u_streams complete dim=0
#pragma HLS ARRAY_PARTITION variable=cur_v_streams complete dim=0
#pragma HLS ARRAY_PARTITION variable=cur_dot1_streams complete dim=0
#pragma HLS ARRAY_PARTITION variable=cur_dot2_streams complete dim=0
#pragma HLS ARRAY_PARTITION variable=cur_out1_streams complete dim=0
#pragma HLS ARRAY_PARTITION variable=cur_out2_streams complete dim=0
#pragma HLS ARRAY_PARTITION variable=cur_acc1_streams complete dim=0
#pragma HLS ARRAY_PARTITION variable=cur_acc2_streams complete dim=0
#pragma HLS ARRAY_PARTITION variable=rec_u_streams complete dim=0
#pragma HLS ARRAY_PARTITION variable=rec_v_streams complete dim=0
#pragma HLS ARRAY_PARTITION variable=rec_dot1_streams complete dim=0
#pragma HLS ARRAY_PARTITION variable=rec_dot2_streams complete dim=0
#pragma HLS ARRAY_PARTITION variable=rec_out1_streams complete dim=0
#pragma HLS ARRAY_PARTITION variable=rec_out2_streams complete dim=0
#pragma HLS ARRAY_PARTITION variable=rec_acc1_streams complete dim=0
#pragma HLS ARRAY_PARTITION variable=rec_acc2_streams complete dim=0
#pragma HLS ARRAY_PARTITION variable=gates_s1_streams complete dim=0
#pragma HLS ARRAY_PARTITION variable=gates_s2_streams complete dim=0
#pragma HLS ARRAY_PARTITION variable=x1_streams complete dim=0
#pragma HLS ARRAY_PARTITION variable=x2_streams complete dim=0
#pragma HLS ARRAY_PARTITION variable=h1_streams complete dim=0
#pragma HLS ARRAY_PARTITION variable=h2_streams complete dim=0
#pragma HLS ARRAY_PARTITION variable=nz_v_stream1_cur complete dim=0
#pragma HLS ARRAY_PARTITION variable=nz_v_stream1_rec complete dim=0
#pragma HLS ARRAY_PARTITION variable=nz_v_stream2_cur complete dim=0
#pragma HLS ARRAY_PARTITION variable=nz_v_stream2_rec complete dim=0
#pragma HLS ARRAY_PARTITION variable=nz_u_stream1_cur complete dim=0
#pragma HLS ARRAY_PARTITION variable=nz_u_stream1_rec complete dim=0
#pragma HLS ARRAY_PARTITION variable=nz_u_stream2_cur complete dim=0
#pragma HLS ARRAY_PARTITION variable=nz_u_stream2_rec complete dim=0
#endif
  hlsutils::Log(0, "Starting ZeroTileCombinationDMA");
  svd::NZIndex2LstmDMA<kNumIter, kNumTilesU, kNumGates>(nz_u_port,
    nz_u_stream1_cur, nz_u_stream1_rec, nz_u_stream2_cur,
    nz_u_stream2_rec);
  svd::NZIndexDMA<kNumIter, kNumTilesV, kNumGates>(nz_v_port,
    nz_v_stream1_cur, nz_v_stream1_rec);
  // ===========================================================================
  // Current Input DMA
  // ===========================================================================
  hlsutils::Log(0, "Starting InputDMA");
  svd::InputDMA<kInputLength, kNumTilesU, kNumZeroTilesU, kNumCurGates, kNumIter>(
    x1_port, nz_u_stream1_cur, x1_streams);
  svd::InputDMA<kInputLength, kNumTilesU, kNumZeroTilesU, kNumCurGates, kNumIter>(
    x2_port, nz_u_stream2_cur, x2_streams);
  // ===========================================================================
  // Recurrent Input DMA
  // ===========================================================================
  svd::InputDMA<kOutputLength, kNumTilesU, kNumZeroTilesU, kNumCurGates, kNumIter>(
    h_t1_prev_port, nz_u_stream1_rec, h1_streams);
  svd::InputDMA<kOutputLength, kNumTilesU, kNumZeroTilesU, kNumCurGates, kNumIter>(
    h_t2_prev_port, nz_u_stream2_rec, h2_streams);
  // ===========================================================================
  // Gates DMA
  // ===========================================================================
  const int kUcurSize = kNumGates / 2 * kNumIter * kPrunedLengthU;
  const int kUrecSize = kNumGates / 2 * kNumIter * kOutputLengthPrunedU;
  const int kSsize = kNumGates * kNumIter;
  const int kVsize = kNumGates * kNumIter * kPrunedLengthV;
  const int kBitWidthU = FIX_WIDTH * 4;
  const int kBitWidthV = FIX_WIDTH * 8;
  const int kBitWidthS = FIX_WIDTH * 8;
#ifndef __VITIS_HLS__
  svd::WeightD u_cur_gate_streams[kNumGates / 2][kNumIter * kPrunedLengthU];
  svd::WeightD u_rec_gate_streams[kNumGates / 2][kNumIter * kOutputLengthPrunedU];
  svd::WeightD v_gate_streams[kNumGates][kNumIter * kPrunedLengthV];
#pragma HLS ARRAY_PARTITION variable=u_cur_gate_streams complete dim=1
#pragma HLS ARRAY_PARTITION variable=u_rec_gate_streams complete dim=1
#pragma HLS ARRAY_PARTITION variable=v_gate_streams complete dim=1
#pragma HLS STREAM variable=u_cur_gate_streams depth=1 dim=1
#pragma HLS STREAM variable=u_rec_gate_streams depth=1 dim=1
#pragma HLS STREAM variable=v_gate_streams depth=1 dim=1
  hlsutils::Log(0, "Starting ArraySplitter");
  svd::ArraySplitter<ap_uint<kBitWidthU>, svd::WeightD, kBitWidthU, FIX_WIDTH, kUcurSize>(
    u_cur_port, u_cur_gate_streams);
  svd::ArraySplitter<ap_uint<kBitWidthU>, svd::WeightD, kBitWidthU, FIX_WIDTH, kUrecSize>(
    u_rec_port, u_rec_gate_streams);
  svd::ArraySplitter<ap_uint<kBitWidthV>, svd::WeightD, kBitWidthV, FIX_WIDTH, kVsize>(
    v_port, v_gate_streams);
#else
  hls::stream<svd::WeightD> u_cur_gate_streams[kNumGates / 2];
  hls::stream<svd::WeightD> u_rec_gate_streams[kNumGates / 2];
  hls::stream<svd::WeightD> v_gate_streams[kNumGates];
#pragma HLS ARRAY_PARTITION variable=u_cur_gate_streams complete dim=0
#pragma HLS ARRAY_PARTITION variable=u_rec_gate_streams complete dim=0
#pragma HLS ARRAY_PARTITION variable=v_gate_streams complete dim=0
#pragma HLS STREAM variable=u_cur_gate_streams depth=2
#pragma HLS STREAM variable=u_rec_gate_streams depth=2
#pragma HLS STREAM variable=v_gate_streams depth=2
  hlsutils::Log(0, "Starting ArraySplitter");
  svd::StreamSplitter<ap_uint<kBitWidthU>, svd::WeightD, kBitWidthU, FIX_WIDTH>(
    kUcurSize, u_cur_port, u_cur_gate_streams);
  svd::StreamSplitter<ap_uint<kBitWidthU>, svd::WeightD, kBitWidthU, FIX_WIDTH>(
    kUrecSize, u_rec_port, u_rec_gate_streams);
  svd::StreamSplitter<ap_uint<kBitWidthV>, svd::WeightD, kBitWidthV, FIX_WIDTH>(
    kVsize, v_port, v_gate_streams);
#endif
  svd::StreamSplitter<ap_uint<kBitWidthS>, svd::WeightD, kBitWidthS, FIX_WIDTH>(
    kSsize, s1_port, gates_s1_streams);
  svd::StreamSplitter<ap_uint<kBitWidthS>, svd::WeightD, kBitWidthS, FIX_WIDTH>(
    kSsize, s2_port, gates_s2_streams);
  const bool kUweights = true;
  // ===========================================================================
  // Current Dot Product Unit
  // ===========================================================================
#ifndef __VITIS_HLS__
  Current_SVD_Kernels:
  for (int g = 0; g < kNumCurGates; ++g) {
#pragma HLS UNROLL
    svd::DispatchGateFromArray(kUweights, kNumIter, kNumNonZeroTilesU,
      kTileSizeUCurrent, u_cur_gate_streams[g], cur_u_streams[g]);
    svd::DispatchGateFromArray(!kUweights, kNumIter, kNumNonZeroTilesV, kTileSizeV,
      v_gate_streams[g], cur_v_streams[g]);
  }
#else
  svd::DispatchGateFromStream<svd::WeightD, kNumCurGates, kNumNonZeroTilesU>(
    kUweights, kNumIter, kNumNonZeroTilesU, kTileSizeUCurrent,
    u_cur_gate_streams, cur_u_streams);
  svd::DispatchGateFromStream<svd::WeightD, kNumCurGates, kTileSizeV>(
    !kUweights, kNumIter, kNumNonZeroTilesV, kTileSizeV,
    v_gate_streams, cur_v_streams);
#endif
  svd::UDotUnit2Lstm<kInputLength, kNumTilesU, kNumZeroTilesU, kNumIter, kNumCurGates>(
    x1_streams, x2_streams, cur_u_streams, cur_dot1_streams,
    cur_dot2_streams);
  svd::VDotUnit2LstmV2<kOutputLength, kNumTilesV, kNumZeroTilesV, kNumIter, kNumCurGates>(
    false, nullptr, nullptr,
    cur_dot1_streams, cur_dot2_streams,
    gates_s1_streams, gates_s2_streams,
    cur_v_streams, nz_v_stream1_cur,
    cur_acc1_streams, cur_acc2_streams);
  // ===========================================================================
  // Recur Dot Product Unit
  // ===========================================================================
#ifndef __VITIS_HLS__
  Recurrent_SVD_Kernels:
  for (int g = 0; g < kNumRecGates; ++g) {
#pragma HLS UNROLL
    svd::DispatchGateFromArray(kUweights, kNumIter, kNumNonZeroTilesU,
      kTileSizeURecur, u_rec_gate_streams[g], rec_u_streams[g]);
    svd::DispatchGateFromArray(!kUweights, kNumIter, kNumNonZeroTilesV, kTileSizeV,
      v_gate_streams[kNumCurGates + g], rec_v_streams[g]);
  }
#else
  svd::DispatchGateFromStream<svd::WeightD, kNumRecGates, kNumNonZeroTilesU>(
    kUweights, kNumIter, kNumNonZeroTilesU, kTileSizeURecur,
    u_rec_gate_streams, rec_u_streams);
  svd::DispatchGateFromStream<svd::WeightD, kNumRecGates, kTileSizeV>(
    !kUweights, kNumIter, kNumNonZeroTilesV, kTileSizeV,
    &v_gate_streams[kNumRecGates], rec_v_streams);
#endif
  svd::UDotUnit2Lstm<kOutputLength, kNumTilesU, kNumZeroTilesU, kNumIter, kNumRecGates>(
    h1_streams, h2_streams, rec_u_streams, rec_dot1_streams,
    rec_dot2_streams);
  svd::VDotUnit2LstmV2<kOutputLength, kNumTilesV, kNumZeroTilesV, kNumIter, kNumRecGates>(
    false, nullptr, nullptr, rec_dot1_streams, rec_dot2_streams,
    &gates_s1_streams[kNumRecGates], &gates_s2_streams[kNumRecGates],
    rec_v_streams, nz_v_stream1_rec, rec_acc1_streams, rec_acc2_streams);
  // ===========================================================================
  // Output Non-Linearities
  // ===========================================================================
  svd::NonLinearityUnit<kOutputLength, kNumTilesV, kNumCurGates>(c_t1_prev_port,
    cur_acc1_streams, rec_acc1_streams, h_t1_curr_port, c_t1_curr_port, true,
    bias1_port);
  svd::NonLinearityUnit<kOutputLength, kNumTilesV, kNumCurGates>(c_t2_prev_port,
    cur_acc2_streams, rec_acc2_streams, h_t2_curr_port, c_t2_curr_port, true,
    bias2_port);
}

} // svd

#ifndef __VITIS_HLS__
#else
void HlsLstmSvd(const int num_active_inputs,
    const int input_size,
    const int output_size,
    const int num_refinements[svd::lstm_params::N],
    // const hls::vector<int, svd::lstm_params::N> num_refinements,
    // Current Gates
    hls::stream<typename svd::lstm_params::VectTuAxiPacketType>& x_port,
    hls::stream<typename svd::lstm_params::VectTuAxiPacketType>& u_cur_port,
    hls::stream<typename svd::lstm_params::VectG_AxiPacketType>& s_cur_port,
    hls::stream<typename svd::lstm_params::VectTvAxiPacketType>& v_cur_port,
    // Recurrent Gates
    hls::stream<typename svd::lstm_params::VectTuAxiPacketType>& h_prev_port,
    hls::stream<typename svd::lstm_params::VectTuAxiPacketType>& u_rec_port,
    hls::stream<typename svd::lstm_params::VectG_AxiPacketType>& s_rec_port,
    hls::stream<typename svd::lstm_params::VectTvAxiPacketType>& v_rec_port,
    // Non-Linearities
    hls::stream<typename svd::lstm_params::VectGTvAxiPacketType>& bias_port,
    hls::stream<typename svd::lstm_params::VectTvAxiPacketType>& c_prev_port,
    hls::stream<typename svd::lstm_params::VectTvAxiPacketType>& h_curr_port,
    hls::stream<typename svd::lstm_params::VectTvAxiPacketType>& c_curr_port) {
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl
#pragma HLS INTERFACE s_axilite port=num_active_inputs bundle=ctrl
#pragma HLS INTERFACE s_axilite port=input_size bundle=ctrl
#pragma HLS INTERFACE s_axilite port=num_refinements bundle=ctrl
#pragma HLS INTERFACE s_axilite port=output_size bundle=ctrl
#pragma HLS INTERFACE axis port=x_port
#pragma HLS INTERFACE axis port=u_cur_port
#pragma HLS INTERFACE axis port=s_cur_port
#pragma HLS INTERFACE axis port=v_cur_port
#pragma HLS INTERFACE axis port=h_prev_port
#pragma HLS INTERFACE axis port=u_rec_port
#pragma HLS INTERFACE axis port=s_rec_port
#pragma HLS INTERFACE axis port=v_rec_port
#pragma HLS INTERFACE axis port=bias_port
#pragma HLS INTERFACE axis port=c_prev_port
#pragma HLS INTERFACE axis port=h_curr_port
#pragma HLS INTERFACE axis port=c_curr_port
  svd::LstmSvdKernel<svd::lstm_params>(num_active_inputs, input_size,
    output_size, num_refinements, x_port, u_cur_port, s_cur_port, v_cur_port,
    h_prev_port, u_rec_port, s_rec_port, v_rec_port, bias_port, c_prev_port,
    h_curr_port, c_curr_port);
}
#endif // __VITIS_HLS__

/**
 * @brief      HLS Wrapper that calls a DenseSvd accelerator.
 *
 *             Useful in Cosimulation.
 *
 * @param[in]  num_active_inputs  The number of active inputs
 * @param[in]  input_size         The input size
 * @param[in]  output_size        The output size
 * @param[in]  num_refinements    The number of refinements
 * @param[in]  x                  The input array. Shape: (N, I)
 * @param[in]  u                  The u array. Shape: (R, I, G)
 * @param[in]  s                  The s array. Shape: (R, N, G)
 * @param[in]  v                  The v array. Shape: (R, H, G)
 * @param[in]  bias               The bias array. Shape: (N, G, H)
 * @param      y                  The y array. Shape: (H / Tv, N, Tv)
 */
void HlsWrapperLstmSvd(
    const int num_active_inputs,
    const int input_size,
    const int output_size,
    const int num_refinements[svd::lstm_params::N],
    // Current Gates
    const typename svd::lstm_params::ActivationD* x,
    const typename svd::lstm_params::ActivationD* u_cur,
    const typename svd::lstm_params::ActivationD* s_cur,
    const typename svd::lstm_params::ActivationD* v_cur,
    // Recurrent Gates
    const typename svd::lstm_params::ActivationD* h,
    const typename svd::lstm_params::ActivationD* u_rec,
    const typename svd::lstm_params::ActivationD* s_rec,
    const typename svd::lstm_params::ActivationD* v_rec,
    // Non-Linearities
    const typename svd::lstm_params::ActivationD* bias,
    const typename svd::lstm_params::ActivationD* c_prev,
    typename svd::lstm_params::ActivationD* h_curr,
    typename svd::lstm_params::ActivationD* c_curr) {
#ifdef __VITIS_HLS__
  // Current Gates
  hls::stream<typename svd::lstm_params::VectTuAxiPacketType> x_port;
  hls::stream<typename svd::lstm_params::VectTuAxiPacketType> u_cur_port;
  hls::stream<typename svd::lstm_params::VectG_AxiPacketType> s_cur_port;
  hls::stream<typename svd::lstm_params::VectTvAxiPacketType> v_cur_port;
  // Recurrent Gates
  hls::stream<typename svd::lstm_params::VectTuAxiPacketType> h_prev_port;
  hls::stream<typename svd::lstm_params::VectTuAxiPacketType> u_rec_port;
  hls::stream<typename svd::lstm_params::VectG_AxiPacketType> s_rec_port;
  hls::stream<typename svd::lstm_params::VectTvAxiPacketType> v_rec_port;
  // Non-Linearities
  hls::stream<typename svd::lstm_params::VectGTvAxiPacketType> bias_port;
  hls::stream<typename svd::lstm_params::VectTvAxiPacketType> c_prev_port;
  hls::stream<typename svd::lstm_params::VectTvAxiPacketType> h_curr_port;
  hls::stream<typename svd::lstm_params::VectTvAxiPacketType> c_curr_port;
  svd::SetLstmSvdInputs<svd::lstm_params>(
    num_active_inputs, input_size, output_size, num_refinements,
    x, u_cur, s_cur, v_cur, h, u_rec, s_rec, v_rec, bias, c_prev,
    x_port, u_cur_port, s_cur_port, v_cur_port,
    h_prev_port, u_rec_port, s_rec_port, v_rec_port, bias_port, c_prev_port);
  HlsLstmSvd(num_active_inputs, input_size, output_size, num_refinements,
    x_port, u_cur_port, s_cur_port, v_cur_port,
    h_prev_port, u_rec_port, s_rec_port, v_rec_port,
    bias_port, c_prev_port, h_curr_port, c_curr_port);
  svd::GetLstmSvdOutputs<svd::lstm_params>(num_active_inputs,
    output_size, h_curr, c_curr, h_curr_port, c_curr_port);
#endif // __VITIS_HLS__
}