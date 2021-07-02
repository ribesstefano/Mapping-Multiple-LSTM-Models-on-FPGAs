#include "lstm/hls/lstm_svd.h"
#include "svd_params.h"
#include "dma/svd_dma.h"
#include "kernel/u_kernel.h"
#include "kernel/s_kernel.h"
#include "kernel/v_kernel.h"
#include "math_utils/activation_functions.h"
#include "hls_utils/hls_debugging.h"

#include "hls_stream.h"
#include "ap_int.h"

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
    const ap_uint<NUM_TILES_V> nz_v_port[NUM_ITERATIONS * 8], // non-zero indexes
    const ap_uint<NUM_TILES_U> nz_u_port[NUM_ITERATIONS * 8], // non-zero indexes
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
  hls_utils::Log(0, "[INFO] Running SvdModel2LstmSDSoCV2.");
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
  const int kNumElemsTileUCurrent = kInputLength / kNumTilesU;
  const int kNumElemsTileURecur = kOutputLength / kNumTilesU;
  const int kNumElemsTileV = kOutputLength / kNumTilesV;
  assert(kNumTilesU % 2 == 0);
  assert(kNumTilesV % 2 == 0);
  // assert(kNumZeroTilesU % 2 == 0);
  // assert(kNumZeroTilesV % 2 == 0);
  assert(kNumIter % 2 == 0);
  hls_utils::Log(0, "[INFO] assert passed.");

  const int kNumElemsTileU = kInputLength / kNumTilesU;
  const int kPrunedLengthU = kInputLength - kNumZeroTilesU * kNumElemsTileU;
  const int kPrunedLengthV = kOutputLength - kNumZeroTilesV * kNumElemsTileV;
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

  const int kInputLengthPruned = kInputLength - kInputLength / kNumTilesU * kNumZeroTilesU;
  const int kOutputLengthPrunedU = kOutputLength - kOutputLength / kNumTilesU * kNumZeroTilesU;
  const int kOutputLengthPrunedV = kOutputLength - kOutputLength / kNumTilesV * kNumZeroTilesV;
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
#pragma HLS INTERFACE m_axi port=u_cur_port offset=slave depth=kUcurPortDepth bundle=u_cur_dmem
#pragma HLS INTERFACE m_axi port=u_rec_port offset=slave depth=kUrecPortDepth bundle=u_rec_dmem
#pragma HLS INTERFACE m_axi port=v_port offset=slave depth=kVportDepth bundle=v_dmem
#pragma HLS INTERFACE m_axi port=s1_port offset=slave depth=kS1portDepth bundle=s1_dmem
#pragma HLS INTERFACE m_axi port=s2_port offset=slave depth=kS2portDepth bundle=s2_dmem

#pragma HLS INTERFACE ap_fifo port=x1_port
#pragma HLS INTERFACE ap_fifo port=x2_port
#pragma HLS INTERFACE ap_fifo port=bias1_port
#pragma HLS INTERFACE ap_fifo port=bias2_port
#pragma HLS INTERFACE ap_fifo port=nz_v_port
#pragma HLS INTERFACE ap_fifo port=nz_u_port
#pragma HLS INTERFACE ap_fifo port=h_t1_prev_port
#pragma HLS INTERFACE ap_fifo port=h_t2_prev_port
#pragma HLS INTERFACE ap_fifo port=h_t1_curr_port
#pragma HLS INTERFACE ap_fifo port=h_t2_curr_port
#pragma HLS INTERFACE ap_fifo port=c_t1_prev_port
#pragma HLS INTERFACE ap_fifo port=c_t2_prev_port
#pragma HLS INTERFACE ap_fifo port=c_t1_curr_port
#pragma HLS INTERFACE ap_fifo port=c_t2_curr_port
#endif // SDS_DESIGN

#pragma HLS DATAFLOW
  hls_utils::Log(0, "[INFO] DATAFLOW passed.");

  // ===========================================================================
  // Current streams
  // ===========================================================================
  svd::WeightStream cur_u_streams[kNumCurGates][kNumNonZeroTilesU];
  svd::WeightStream cur_v_streams[kNumCurGates][kNumElemsTileV]; // [kNumNonZeroTilesV];
  svd::ActivationStream cur_dot1_streams[kNumCurGates];
  svd::ActivationStream cur_dot2_streams[kNumCurGates];
  svd::ActivationStream cur_out1_streams[kNumCurGates][kNumNonZeroTilesV];
  svd::ActivationStream cur_out2_streams[kNumCurGates][kNumNonZeroTilesV];
  svd::ActivationStream cur_acc1_streams[kNumCurGates][kNumElemsTileV]; // [kNumTilesV];
  svd::ActivationStream cur_acc2_streams[kNumCurGates][kNumElemsTileV]; // [kNumTilesV];
#pragma HLS ARRAY_PARTITION variable=cur_u_streams complete dim=0
#pragma HLS ARRAY_PARTITION variable=cur_v_streams complete dim=0
#pragma HLS ARRAY_PARTITION variable=cur_dot1_streams complete dim=0
#pragma HLS ARRAY_PARTITION variable=cur_dot2_streams complete dim=0
#pragma HLS ARRAY_PARTITION variable=cur_out1_streams complete dim=0
#pragma HLS ARRAY_PARTITION variable=cur_out2_streams complete dim=0
#pragma HLS ARRAY_PARTITION variable=cur_acc1_streams complete dim=0
#pragma HLS ARRAY_PARTITION variable=cur_acc2_streams complete dim=0
  // ===========================================================================
  // Recur streams
  // ===========================================================================
  svd::WeightStream rec_u_streams[kNumRecGates][kNumNonZeroTilesU];
  svd::WeightStream rec_v_streams[kNumRecGates][kNumElemsTileV]; // [kNumNonZeroTilesV];
  svd::ActivationStream rec_dot1_streams[kNumRecGates];
  svd::ActivationStream rec_dot2_streams[kNumRecGates];
  svd::ActivationStream rec_out1_streams[kNumRecGates][kNumNonZeroTilesV];
  svd::ActivationStream rec_out2_streams[kNumRecGates][kNumNonZeroTilesV];
  svd::ActivationStream rec_acc1_streams[kNumRecGates][kNumElemsTileV]; // [kNumTilesV];
  svd::ActivationStream rec_acc2_streams[kNumRecGates][kNumElemsTileV]; // [kNumTilesV];
#pragma HLS ARRAY_PARTITION variable=rec_u_streams complete dim=0
#pragma HLS ARRAY_PARTITION variable=rec_v_streams complete dim=0
#pragma HLS ARRAY_PARTITION variable=rec_dot1_streams complete dim=0
#pragma HLS ARRAY_PARTITION variable=rec_dot2_streams complete dim=0
#pragma HLS ARRAY_PARTITION variable=rec_out1_streams complete dim=0
#pragma HLS ARRAY_PARTITION variable=rec_out2_streams complete dim=0
#pragma HLS ARRAY_PARTITION variable=rec_acc1_streams complete dim=0
#pragma HLS ARRAY_PARTITION variable=rec_acc2_streams complete dim=0
  // ===========================================================================
  // Scalar streams
  // ===========================================================================
  svd::WeightStream gates_s1_streams[kNumGates]; // used for both curr and recur
  svd::WeightStream gates_s2_streams[kNumGates]; // used for both curr and recur
#pragma HLS ARRAY_PARTITION variable=gates_s1_streams complete dim=0
#pragma HLS ARRAY_PARTITION variable=gates_s2_streams complete dim=0
  // ===========================================================================
  // Current input streams
  // ===========================================================================
  svd::ActivationStream x1_streams[kNumCurGates][kNumNonZeroTilesU];
  svd::ActivationStream x2_streams[kNumCurGates][kNumNonZeroTilesU];
#pragma HLS ARRAY_PARTITION variable=x1_streams complete dim=0
#pragma HLS ARRAY_PARTITION variable=x2_streams complete dim=0
  // ===========================================================================
  // Recurrent input streams
  // ===========================================================================
  svd::ActivationStream h1_streams[kNumRecGates][kNumNonZeroTilesU];
  svd::ActivationStream h2_streams[kNumRecGates][kNumNonZeroTilesU];
#pragma HLS ARRAY_PARTITION variable=h1_streams complete dim=0
#pragma HLS ARRAY_PARTITION variable=h2_streams complete dim=0
  // ===========================================================================
  // Streams Depth Sizing
  // ===========================================================================
  // NOTE: We divide the FIFO depths by a certain factor to save BRAMs. Be aware
  // that a wrong factor could lead to deadlocks!
  const int kFIFOdepthFactor = kNumIter * 2;
  const int kStreamDepthUCurrent = kNumIter * kNumElemsTileUCurrent / kFIFOdepthFactor == 0 ? 2 : kNumIter * kNumElemsTileUCurrent / kFIFOdepthFactor;
  const int kStreamDepthURecurrent = kNumIter * kNumElemsTileURecur / kFIFOdepthFactor == 0 ? 2 : kNumIter * kNumElemsTileURecur / kFIFOdepthFactor;
  const int kStreamDepthV = kNumIter * kNumTilesV / kFIFOdepthFactor == 0 ? 2 : kNumIter * kNumTilesV / kFIFOdepthFactor;
  const int kTileAccStreamDepth = 2;
#pragma HLS STREAM variable=x1_streams depth=kStreamDepthUCurrent dim=2
#pragma HLS STREAM variable=x2_streams depth=kStreamDepthUCurrent dim=2
#pragma HLS STREAM variable=h1_streams depth=kStreamDepthURecurrent dim=2
#pragma HLS STREAM variable=h2_streams depth=kStreamDepthURecurrent dim=2

#pragma HLS STREAM variable=cur_u_streams depth=kStreamDepthUCurrent dim=2
#pragma HLS STREAM variable=rec_u_streams depth=kStreamDepthURecurrent dim=2
#pragma HLS STREAM variable=cur_v_streams depth=kStreamDepthV dim=2
#pragma HLS STREAM variable=rec_v_streams depth=kStreamDepthV dim=2

#pragma HLS STREAM variable=gates_s1_streams depth=kStreamDepthIter
#pragma HLS STREAM variable=gates_s2_streams depth=kStreamDepthIter

#pragma HLS STREAM variable=cur_dot1_streams depth=kStreamDepthIter
#pragma HLS STREAM variable=cur_dot2_streams depth=kStreamDepthIter
#pragma HLS STREAM variable=rec_dot1_streams depth=kStreamDepthIter
#pragma HLS STREAM variable=rec_dot2_streams depth=kStreamDepthIter
#pragma HLS STREAM variable=cur_acc1_streams depth=kTileAccStreamDepth dim=2
#pragma HLS STREAM variable=cur_acc2_streams depth=kTileAccStreamDepth dim=2
#pragma HLS STREAM variable=rec_acc1_streams depth=kTileAccStreamDepth dim=2
#pragma HLS STREAM variable=rec_acc2_streams depth=kTileAccStreamDepth dim=2

#pragma HLS STREAM variable=cur_out1_streams depth=kOutStreamDepth dim=2
#pragma HLS STREAM variable=cur_out2_streams depth=kOutStreamDepth dim=2
#pragma HLS STREAM variable=rec_out1_streams depth=kOutStreamDepth dim=2
#pragma HLS STREAM variable=rec_out2_streams depth=kOutStreamDepth dim=2
  hls_utils::Log(0, "[INFO] Depth sizing passed.");

  // ===========================================================================
  // Zero Combinations DMA
  // ===========================================================================
  // NOTE: We divide the FIFO depths by a certain factor to save BRAMs. Be aware
  // that a wrong factor could lead to deadlocks!
  const int kFIFOdepthDivider = 8;
  const int kStreamDepthIter = kNumIter / kFIFOdepthDivider;
  hls_utils::Log(0, "[INFO] DATAFLOW passed.");
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
#pragma HLS ARRAY_PARTITION variable=nz_v_stream1_cur complete
#pragma HLS ARRAY_PARTITION variable=nz_v_stream1_rec complete
#pragma HLS ARRAY_PARTITION variable=nz_v_stream2_cur complete
#pragma HLS ARRAY_PARTITION variable=nz_v_stream2_rec complete
#pragma HLS ARRAY_PARTITION variable=nz_u_stream1_cur complete
#pragma HLS ARRAY_PARTITION variable=nz_u_stream1_rec complete
#pragma HLS ARRAY_PARTITION variable=nz_u_stream2_cur complete
#pragma HLS ARRAY_PARTITION variable=nz_u_stream2_rec complete

  hls_utils::Log(0, "Starting ZeroTileCombinationDMA");
  hls_utils::Log(0, "Starting ZeroTileCombinationDMA");
  svd::ZeroTileCombination2LstmDMA<kNumIter, kNumTilesU, kNumGates>(nz_u_port,
    nz_u_stream1_cur, nz_u_stream1_rec, nz_u_stream2_cur,
    nz_u_stream2_rec);
  svd::ZeroTileCombinationDMA<kNumIter, kNumTilesV, kNumGates>(nz_v_port,
    nz_v_stream1_cur, nz_v_stream1_rec);
  // ===========================================================================
  // Current Input DMA
  // ===========================================================================
  hls_utils::Log(0, "Starting InputDMA");
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
  svd::WeightD u_cur_gate_streams[kNumGates / 2][kNumIter * kInputLength / kNumTilesU * (kNumTilesU - kNumZeroTilesU)];
  svd::WeightD u_rec_gate_streams[kNumGates / 2][kNumIter * kOutputLength / kNumTilesU * (kNumTilesU - kNumZeroTilesU)];
  svd::WeightD v_gate_streams[kNumGates][kNumIter * kOutputLength / kNumTilesV * (kNumTilesV - kNumZeroTilesV)];
#pragma HLS STREAM variable=u_cur_gate_streams depth=1 dim=1
#pragma HLS STREAM variable=u_rec_gate_streams depth=1 dim=1
#pragma HLS STREAM variable=v_gate_streams depth=1 dim=1
#pragma HLS ARRAY_PARTITION variable=u_cur_gate_streams complete dim=1
#pragma HLS ARRAY_PARTITION variable=u_rec_gate_streams complete dim=1
#pragma HLS ARRAY_PARTITION variable=v_gate_streams complete dim=1

  const int kUcurSize = kNumGates / 2 * kNumIter * kInputLength / kNumTilesU * (kNumTilesU - kNumZeroTilesU);
  const int kUrecSize = kNumGates / 2 * kNumIter * kOutputLength / kNumTilesU * (kNumTilesU - kNumZeroTilesU);
  const int kSsize = kNumGates * kNumIter;
  const int kVsize = kNumGates * kNumIter * kOutputLength / kNumTilesV * (kNumTilesV - kNumZeroTilesV);
  const int kBitWidthU = FIX_WIDTH * 4;
  const int kBitWidthV = FIX_WIDTH * 8;
  const int kBitWidthS = FIX_WIDTH * 8;
  hls_utils::Log(0, "Starting ArraySplitter");
  svd::ArraySplitter<ap_uint<kBitWidthU>, svd::WeightD, kBitWidthU, FIX_WIDTH, kUcurSize>(u_cur_port, u_cur_gate_streams);
  svd::ArraySplitter<ap_uint<kBitWidthU>, svd::WeightD, kBitWidthU, FIX_WIDTH, kUrecSize>(u_rec_port, u_rec_gate_streams);
  svd::ArraySplitter<ap_uint<kBitWidthV>, svd::WeightD, kBitWidthV, FIX_WIDTH, kVsize>(v_port, v_gate_streams);
  svd::StreamSplitter<ap_uint<kBitWidthS>, svd::WeightD, kBitWidthS, FIX_WIDTH>(kSsize, s1_port, gates_s1_streams);
  svd::StreamSplitter<ap_uint<kBitWidthS>, svd::WeightD, kBitWidthS, FIX_WIDTH>(kSsize, s2_port, gates_s2_streams);
  const bool kUweights = true;
  // ===========================================================================
  // Current Dot Product Unit
  // ===========================================================================
  Current_Gates_Dot_Product_Loop:
  for (int g = 0; g < kNumCurGates; ++g) {
#pragma HLS UNROLL
    hls_utils::Log(0, std::string("Starting Cur Gate n.") + std::to_string(g));
    svd::GateDMA<svd::WeightD>(kUweights, kNumIter, kNumNonZeroTilesU, kNumElemsTileUCurrent, u_cur_gate_streams[g], cur_u_streams[g]);
    svd::GateDMA<svd::WeightD>(!kUweights, kNumIter, kNumNonZeroTilesV, kNumElemsTileV, v_gate_streams[g], cur_v_streams[g]);
    svd::UDotUnit2Lstm<kInputLength, kNumTilesU, kNumZeroTilesU, kNumIter, 1>(x1_streams[g],
      x2_streams[g], cur_u_streams[g],
      cur_dot1_streams[g], cur_dot2_streams[g]);
    svd::VDotUnit2LstmV2<kOutputLength, kNumTilesV, kNumZeroTilesV, kNumIter, 1>(
      false,
      nullptr,
      nullptr,
      cur_dot1_streams[g],
      cur_dot2_streams[g],
      gates_s1_streams[g],
      gates_s2_streams[g],
      cur_v_streams[g],
      nz_v_stream1_cur[g],
      cur_acc1_streams[g],
      cur_acc2_streams[g]);
  }
  // ===========================================================================
  // Recur Dot Product Unit
  // ===========================================================================
  Recur_Gates_Dot_Product_Loop:
  for (int g = 0; g < kNumRecGates; ++g) {
#pragma HLS UNROLL
    hls_utils::Log(0, std::string("Starting Rec Gate n.") + std::to_string(g));
    svd::GateDMA<svd::WeightD>(kUweights, kNumIter, kNumNonZeroTilesU, kNumElemsTileURecur, u_rec_gate_streams[g], rec_u_streams[g]);
    svd::GateDMA<svd::WeightD>(!kUweights, kNumIter, kNumNonZeroTilesV, kNumElemsTileV, v_gate_streams[kNumCurGates + g], rec_v_streams[g]);
    svd::UDotUnit2Lstm<kOutputLength, kNumTilesU, kNumZeroTilesU, kNumIter, 1>(h1_streams[g],
      h2_streams[g], rec_u_streams[g],
      rec_dot1_streams[g], rec_dot2_streams[g]);
    svd::VDotUnit2LstmV2<kOutputLength, kNumTilesV, kNumZeroTilesV, kNumIter, 1>(
      false,
      nullptr,
      nullptr,
      rec_dot1_streams[g],
      rec_dot2_streams[g],
      gates_s1_streams[kNumCurGates + g],
      gates_s2_streams[kNumCurGates + g],
      rec_v_streams[g],
      nz_v_stream1_rec[g],
      rec_acc1_streams[g],
      rec_acc2_streams[g]);
  }
  // ===========================================================================
  // Output Non-Linearities
  // ===========================================================================
  // NOTE: The output FIFOs in NonLinearityUnit have been resized! Check for deadlocks!
  svd::NonLinearityUnit<kOutputLength, kNumTilesV, kNumCurGates>(c_t1_prev_port,
    cur_acc1_streams, rec_acc1_streams, h_t1_curr_port, c_t1_curr_port, true,
    bias1_port);
  svd::NonLinearityUnit<kOutputLength, kNumTilesV, kNumCurGates>(c_t2_prev_port,
    cur_acc2_streams, rec_acc2_streams, h_t2_curr_port, c_t2_curr_port, true,
    bias2_port);

#ifdef DEBUG_FIFOS

  const int kNumPEsU = NUM_TILES_U - NUM_ZERO_TILES_U;
  const int kNumPEsVCur = INPUT_SIZE / NUM_TILES_V;
  const int kNumPEsVRec = HIDDEN_SIZE / NUM_TILES_V;
  const int kNumUprobes = kNumGates * kNumPEsU * 3; // one for each: x1, x2, u streams
  const int kNumVprobes = kNumGates / 2 * (kNumPEsVCur + kNumPEsVRec); // one for v streams
  const int kNumProbes = kNumUprobes + kNumVprobes;
  svd::ProbeStream stop_ctrl;
  svd::ProbeStream probe_ctrl[kNumUprobes];

  svd::ClockCounter<svd::CounterD, kNumProbes>(probe_ctrl, stop_ctrl, counters_port, clk_count_port);
#endif
}