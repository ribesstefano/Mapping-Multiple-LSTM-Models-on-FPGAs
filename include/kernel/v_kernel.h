#ifndef KERNEL_V_KERNEL_H_
#define KERNEL_V_KERNEL_H_

#include "svd_params.h"

template <typename params>
void KernelV(svd::SvdStreams<params> &streams) {
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
    for (int i = 0; i < params::R; ++i) {
      for (int j = 0; j < params::PrunedSizeV / params::R / params::PeV; ++j) {
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
  for (int i = 0; i < params::R; ++i) {
    for (int j = 0; j < params::PrunedSizeV / params::R / params::PeV; ++j) {
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


namespace svd {

template <int VectLength, int NumTiles, int NumZeroTiles, int NumIter, int NumTimesteps>
void VDotUnit2LstmV2(const bool has_bias,
                   svd::WeightStream *bias1,
                   svd::WeightStream *bias2,
                   svd::ActivationStream &gate_dot1_streams,
                   svd::ActivationStream &gate_dot2_streams,
                   svd::WeightStream &gate_s1_streams,
                   svd::WeightStream &gate_s2_streams,
                   svd::WeightStream (&gate_v_streams)[VectLength / NumTiles],
                   hls::stream<ap_uint<NumTiles> > &comb_stream_port,
                   svd::ActivationStream (&gate_out1_streams)[VectLength / NumTiles],
                   svd::ActivationStream (&gate_out2_streams)[VectLength / NumTiles],
                   bool debug_module = false,
                   ProbeStream *probe_ctrl = nullptr) {
#pragma HLS INLINE
#pragma HLS DATAFLOW

  assert(VectLength % NumTiles == 0);
  assert(NumTiles > NumZeroTiles);
  assert(NumTiles % 2 == 0);
  assert(NumIter % 2 == 0);

  const int kFifoResizeFactor = 4;
  const int kNonZeroTiles = NumTiles - NumZeroTiles;
  const int kNumTileElems = VectLength / NumTiles;
  // NOTE: By the time the dot products are available at the ports, the weight
  // values s1, s2 and v should be already at the FIFO ports.
  const int kStreamDepth = NumIter / kFifoResizeFactor;
  hls::stream<svd::MultD> xs1_streams[kNumTileElems];
  hls::stream<svd::MultD> xs2_streams[kNumTileElems];
#pragma HLS STREAM variable=xs1_streams depth=kStreamDepth dim=1
#pragma HLS STREAM variable=xs2_streams depth=kStreamDepth dim=1

  // svd::MultD xs1_val = 0;
  // svd::MultD xs2_val = 0;

  ScalarMul:
  for (int i = 0; i < NumIter; ++i) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
    auto xs1_val = gate_s1_streams.read() * gate_dot1_streams.read();
    auto xs2_val = gate_s2_streams.read() * gate_dot2_streams.read();
#pragma HLS RESOURCE variable=xs1_val core=DSP48 latency=3
#pragma HLS RESOURCE variable=xs2_val core=DSP48 latency=3
    ScalarMulDispatcher:
    for (int j = 0; j < kNumTileElems; ++j) {
      xs1_streams[j].write(xs1_val);
      xs2_streams[j].write(xs2_val);
    }
  }

  svd::WeightStream bias1_streams[kNumTileElems];
  svd::WeightStream bias2_streams[kNumTileElems];
  if (has_bias) {
#pragma HLS STREAM variable=bias1_streams depth=NumTiles dim=1
#pragma HLS STREAM variable=bias2_streams depth=NumTiles dim=1
#pragma HLS ARRAY_PARTITION variable=bias1_streams complete dim=1
#pragma HLS ARRAY_PARTITION variable=bias2_streams complete dim=1
  }

  if (has_bias) {
    BiasDispatcher_tiles:
    for (int i = 0; i < NumTiles; ++i) {
#pragma HLS INLINE off
      BiasDispatcher_elems:
      for (int j = 0; j < kNumTileElems; ++j) {
#pragma HLS PIPELINE II=1
        bias1_streams[j].write(bias1->read());
        bias2_streams[j].write(bias2->read());
      }
    }
  }

  const int kCombStreamDepth = NumIter * NumZeroTiles / (kFifoResizeFactor * 2);
  const int kNzBitLength = hls_utils::log2<NumTiles>::value;
  hls::stream<ap_uint<kNzBitLength> > nz_idx_streams[kNumTileElems];
#pragma HLS STREAM variable=nz_idx_streams depth=kCombStreamDepth dim=1
#pragma HLS RESOURCE variable=nz_idx_streams core=FIFO_SRL

#if 0 // USE_PRIORITY_ENCODER
  // ===========================================================================
  // NOTE: The critical path is HUGE here. So we go for the other solution.
  // ===========================================================================
  ap_uint<NumTiles> zero_comb = 0;

  Convert_Iter:
  for (int i = 0; i < NumIter; ++i) {
    for (int j = 0; j < NumTiles - NumZeroTiles; ++j) {
#pragma HLS PIPELINE II=1
      if (j == 0) {
        zero_comb = comb_stream_port.read();
        int set_idx = PriorityEncoderLSB<NumTiles>(zero_comb);
        assert(set_idx < NumTiles);
        for (int k = 0; k < kNumTileElems; ++k) {
          nz_idx_streams[k].write(set_idx);
        }
        zero_comb[set_idx] = 0;
      } else {
        int set_idx = PriorityEncoderLSB<NumTiles>(zero_comb);
        assert(set_idx < NumTiles);
        for (int k = 0; k < kNumTileElems; ++k) {
          nz_idx_streams[k].write(set_idx);
        }
        zero_comb[set_idx] = 0;
      }
    }
  }
#else
  ap_uint<NumTiles> c;
  int nz_cnt = 0;
  assert(nz_cnt < kNonZeroTiles);

  CombConverter_iter:
  for (int i = 0; i < NumIter; ++i) {
#pragma HLS INLINE off
    CombConverter_tiles:
    for (int j = 0; j < NumTiles; ++j) {
#pragma HLS LOOP_FLATTEN
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=kNonZeroTiles max=kNonZeroTiles
      if (j == 0) {
        c = comb_stream_port.read();
        // std::cout << "nz[" << i << "] = " << c.to_string(2, false) << "\n";
        if (c[0] == 1) {
          for (int k = 0; k < kNumTileElems; ++k) {
            nz_idx_streams[k].write(0);
          }
          nz_cnt++;
        }
      } else {
        if (c[j] == 1) {
          CombConverter_elem:
          for (int k = 0; k < kNumTileElems; ++k) {
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

#if 1
  V_Kernel: {
#pragma HLS INLINE off
    if (debug_module) {
      probe_ctrl->write(1);
    }

    svd::AccumD acc_buffer1[kNumTileElems][NumTiles];
    svd::AccumD acc_buffer2[kNumTileElems][NumTiles];
#pragma HLS ARRAY_PARTITION variable=acc_buffer1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc_buffer2 complete dim=1
// #pragma HLS RESOURCE variable=acc_buffer1 core=RAM_T2P_BRAM latency=1
// #pragma HLS RESOURCE variable=acc_buffer2 core=RAM_T2P_BRAM latency=1
// #pragma HLS RESOURCE variable=acc_buffer1 core=RAM_T2P_URAM
// #pragma HLS RESOURCE variable=acc_buffer2 core=RAM_T2P_URAM

// #pragma HLS RESOURCE variable=acc_buffer1 core=XPM_MEMORY uram
// #pragma HLS RESOURCE variable=acc_buffer2 core=XPM_MEMORY uram


    Init_buffer:
    for (int i = 0; i < NumTiles; ++i) {
#pragma HLS PIPELINE II=1
      for (int j = 0; j < kNumTileElems; ++j) {
        acc_buffer1[j][i] = 0;
        acc_buffer2[j][i] = 0;
      }
    }

    ap_uint<kNzBitLength> nz_idx[kNumTileElems];
    svd::AccumD xs1[kNumTileElems];
    svd::AccumD xs2[kNumTileElems];
#pragma HLS ARRAY_PARTITION variable=nz_idx complete
#pragma HLS ARRAY_PARTITION variable=xs1 complete
#pragma HLS ARRAY_PARTITION variable=xs2 complete
    svd::AccumD mac_1[kNumTileElems];
    svd::AccumD mac_2[kNumTileElems];
#pragma HLS ARRAY_PARTITION variable=mac_1 complete
#pragma HLS ARRAY_PARTITION variable=mac_2 complete
    svd::AccumD acc_1[kNumTileElems];
    svd::AccumD acc_2[kNumTileElems];
#pragma HLS ARRAY_PARTITION variable=acc_1 complete
#pragma HLS ARRAY_PARTITION variable=acc_2 complete

    svd::WeightD v[kNumTileElems];
#pragma HLS ARRAY_PARTITION variable=v complete

#ifndef __SYNTHESIS__
    int nz_idx_buf[kNumTileElems][NumIter * kNonZeroTiles] = {-1};
    const int kPipelineDepth = 8;
    const bool printout = false;
#endif

    for (int i = 0; i < NumIter; ++i) {
      for (int k = 0; k < kNonZeroTiles; ++k) {
#pragma HLS LOOP_FLATTEN
#pragma HLS PIPELINE II=1
        for (int j = 0; j < kNumTileElems; ++j) {
          if (k == 0) {
            xs1[j] = xs1_streams[j].read();
            xs2[j] = xs2_streams[j].read();
          }
          nz_idx[j] = nz_idx_streams[j].read();

#ifndef __SYNTHESIS__
          // ===================================================================
          // Naive dependency detection mechanism:
          // ===================================================================
          // std::cout << "\tPE n." << j << " accessing acc_buffer[" << j << "][" << nz_idx[j] << "]\n";
          const int idx = i * kNonZeroTiles + k;
          nz_idx_buf[j][idx] = nz_idx[j];

          if (idx > 0) {
            int dependency_idx = -1;
            if (j == 0 && printout) {
              std::cout << "curr_idx: " << nz_idx[j] << "\nprev_idx: ";
            }
            for (int q = idx - 1; q > 0; --q) {
              if (nz_idx_buf[j][idx] == nz_idx_buf[j][q]) {
                dependency_idx = q;
                break;
              }
            }
            if (j == 0 && printout) {
              for (int q = idx; q > 0; --q) {
                if (q - 1 == dependency_idx && idx - dependency_idx < kPipelineDepth) {
                  std::cout << "<((";
                }
                std::cout << nz_idx_buf[j][q - 1];
                if (q - 1 == dependency_idx && idx - dependency_idx < kPipelineDepth) {
                  std::cout << "))>";
                }
                std::cout << " ";
                if ((q - 1) % kNonZeroTiles == 0) {
                  std::cout << "| ";
                }
              }
              std::cout << "\ndistance: " << idx - dependency_idx << "\n";
            }
            // NOTE: The dependency will be the same for all PEs, i.e. j indexes.
            if (j == 0 && dependency_idx != -1 && idx - dependency_idx < kPipelineDepth) { 
              //num_raw_hazards++;
              if (printout) {
                std::cout << "[WARNING] Possible dependecy detected: nz[" << idx << "] = " << nz_idx_buf[j][idx]
                          << " -> nz[" << dependency_idx << "] = " << nz_idx_buf[j][dependency_idx]
                          << ", distance: " << idx - dependency_idx << "\n";
              }
            }
            if (j == 0 && printout) {
              std::cout << "\n";
            }
          }
#endif
          v[j] = gate_v_streams[j].read();

          mac_1[j] = (xs1[j] * v[j]) + acc_buffer1[j][nz_idx[j]];
          mac_2[j] = (xs2[j] * v[j]) + acc_buffer2[j][nz_idx[j]];
#pragma HLS RESOURCE variable=mac_1[j] core=DSP48 latency=3
#pragma HLS RESOURCE variable=mac_2[j] core=DSP48 latency=3
#pragma HLS DEPENDENCE variable=acc_buffer1 RAW false inter
#pragma HLS DEPENDENCE variable=acc_buffer2 RAW false inter

          acc_buffer1[j][nz_idx[j]] = mac_1[j];
          acc_buffer2[j][nz_idx[j]] = mac_2[j];
        } // end kNumTileElems
      } // end kNonZeroTiles
    } // end NumIter

    WriteBack_tiles:
    for (int i = 0; i < NumTiles; ++i) {
#pragma HLS PIPELINE II=1
      WriteBack_elems:
      for (int j = 0; j < kNumTileElems; ++j) {
        if (has_bias) {
          auto acc_1 = acc_buffer1[j][i] + bias1_streams[j].read();
          auto acc_2 = acc_buffer2[j][i] + bias2_streams[j].read();
#pragma HLS RESOURCE variable=acc_1 core=AddSub_DSP
#pragma HLS RESOURCE variable=acc_2 core=AddSub_DSP
          gate_out1_streams[j].write(acc_1);
          gate_out2_streams[j].write(acc_2);
        } else {
          gate_out1_streams[j].write(acc_buffer1[j][i]);
          gate_out2_streams[j].write(acc_buffer2[j][i]);
        }
      }
    }

    if (debug_module) {
      probe_ctrl->write(0);
    }
#else

  svd::AccumD acc_buffer1[kNumTileElems][NumTiles];
  svd::AccumD acc_buffer2[kNumTileElems][NumTiles];
#pragma HLS ARRAY_PARTITION variable=acc_buffer1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc_buffer2 complete dim=1
#pragma HLS RESOURCE variable=acc_buffer1 core=RAM_T2P_BRAM latency=1
#pragma HLS RESOURCE variable=acc_buffer2 core=RAM_T2P_BRAM latency=1
// #pragma HLS STREAM variable=acc_buffer1 depth=1
// #pragma HLS STREAM variable=acc_buffer2 depth=1

  ap_uint<kNzBitLength> nz_idx[kNumTileElems];
  svd::AccumD xs1[kNumTileElems];
  svd::AccumD xs2[kNumTileElems];
#pragma HLS ARRAY_PARTITION variable=nz_idx complete
#pragma HLS ARRAY_PARTITION variable=xs1 complete
#pragma HLS ARRAY_PARTITION variable=xs2 complete
  svd::AccumD mac_1[kNumTileElems];
  svd::AccumD mac_2[kNumTileElems];
#pragma HLS ARRAY_PARTITION variable=mac_1 complete
#pragma HLS ARRAY_PARTITION variable=mac_2 complete
  svd::AccumD acc_1[kNumTileElems];
  svd::AccumD acc_2[kNumTileElems];
#pragma HLS ARRAY_PARTITION variable=acc_1 complete
#pragma HLS ARRAY_PARTITION variable=acc_2 complete

  svd::WeightD v[kNumTileElems];
#pragma HLS ARRAY_PARTITION variable=v complete

  hls::stream<svd::AccumD> xsv1_streams[kNumTileElems];
  hls::stream<svd::AccumD> xsv2_streams[kNumTileElems];
#pragma HLS ARRAY_PARTITION variable=xsv1_streams complete
#pragma HLS ARRAY_PARTITION variable=xsv2_streams complete
#pragma HLS STREAM variable=xsv1_streams depth=kNonZeroTiles
#pragma HLS STREAM variable=xsv2_streams depth=kNonZeroTiles
  hls::stream<svd::AccumD> acc1_streams[kNumTileElems];
  hls::stream<svd::AccumD> acc2_streams[kNumTileElems];
#pragma HLS ARRAY_PARTITION variable=acc1_streams complete
#pragma HLS ARRAY_PARTITION variable=acc2_streams complete
  const int kStreamDepthAcc = NumTiles;
#pragma HLS STREAM variable=acc1_streams depth=kStreamDepthAcc
#pragma HLS STREAM variable=acc2_streams depth=kStreamDepthAcc

  for (int n = 0; n < NumIter; ++n) {
    for (int nz = 0; nz < kNonZeroTiles; ++nz) {
#pragma HLS LOOP_FLATTEN
#pragma HLS PIPELINE II=1
      for (int i = 0; i < kNumTileElems; ++i) {
        if (nz == 0) {
          xs1[i] = xs1_streams[i].read();
          xs2[i] = xs2_streams[i].read();
        }
        svd::WeightD v_val = gate_v_streams[i].read();
        svd::AccumD xvs1_val = xs1[i] * v_val;
        svd::AccumD xvs2_val = xs2[i] * v_val;
#pragma HLS RESOURCE variable=xvs1_val core=DSP48 latency=3
#pragma HLS RESOURCE variable=xvs2_val core=DSP48 latency=3
        xsv1_streams[i].write(xvs1_val);
        xsv2_streams[i].write(xvs2_val);
      }
    }
  }
  V_Kernel: {
#pragma HLS INLINE off

    Init_buffer:
    for (int i = 0; i < NumTiles; ++i) {
  #pragma HLS PIPELINE II=1
      for (int j = 0; j < kNumTileElems; ++j) {
        acc_buffer1[j][i] = 0;
        acc_buffer2[j][i] = 0;
      }
    }
    for (int n = 0; n < NumIter; ++n) {
      for (int t = 0; t < kNonZeroTiles; ++t) {
#pragma HLS LOOP_FLATTEN
#pragma HLS PIPELINE II=1
        for (int i = 0; i < kNumTileElems; ++i) {
          ActivationD operand_prev1 = 0;
          ActivationD operand_curr1 = 0;
          ActivationD operand_prev2 = 0;
          ActivationD operand_curr2 = 0;
          // ===================================================================
          // Setup the internal stream
          // ===================================================================
          nz_idx[i] = nz_idx_streams[i].read();
          operand_prev1 = acc_buffer1[i][nz_idx[i]];
          operand_prev2 = acc_buffer2[i][nz_idx[i]];
          operand_curr1 = xsv1_streams[i].read();
          operand_curr2 = xsv2_streams[i].read();
          // ===================================================================
          // Accumulate the incoming streams
          // ===================================================================
          svd::AccumD sum1 = operand_prev1 + operand_curr1;
          svd::AccumD sum2 = operand_prev2 + operand_curr2;
#pragma HLS RESOURCE variable=sum1 core=AddSub_DSP
#pragma HLS RESOURCE variable=sum2 core=AddSub_DSP
          acc_buffer1[i][nz_idx[i]] = sum1;
          acc_buffer2[i][nz_idx[i]] = sum2;
        }
      }
    }
    // ===================================================================
    // Write the results to the output streams
    // ===================================================================
    WriteBack_tiles:
    for (int i = 0; i < NumTiles; ++i) {
#pragma HLS PIPELINE II=1
      WriteBack_elems:
      for (int j = 0; j < kNumTileElems; ++j) {
        if (has_bias) {
          auto acc_1 = acc_buffer1[j][i] + bias1_streams[j].read();
          auto acc_2 = acc_buffer2[j][i] + bias2_streams[j].read();
#pragma HLS RESOURCE variable=acc_1 core=AddSub_DSP
#pragma HLS RESOURCE variable=acc_2 core=AddSub_DSP
          gate_out1_streams[j].write(acc_1);
          gate_out2_streams[j].write(acc_2);
        } else {
          gate_out1_streams[j].write(acc_buffer1[j][i]);
          gate_out2_streams[j].write(acc_buffer2[j][i]);
        }
      }
    }
#endif
  } // end V_Function
}

} // svd

#endif // end KERNEL_V_KERNEL_H_