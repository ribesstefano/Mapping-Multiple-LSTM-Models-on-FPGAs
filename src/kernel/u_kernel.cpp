#include "kernel/u_kernel.h"
#include "kernel/gemv_kernel.h"
#include "hls_utils/adder_tree.h"
#include "dma/svd_dma.h"
#include "dma/axis_lib.h"

#include "assert.h"
#include "ap_axi_sdata.h"
#include "hls_stream.h"
#ifdef __VITIS_HLS__
#include "hls_vector.h"
#endif

#ifndef __VITIS_HLS__
void HlsKernelU(const int num_refinements,
  const typename testu::params::ActivationD x_port[testu::params::N][testu::params::I],
  const typename testu::params::UPortD u_port[testu::params::R * testu::params::PrunedSizeU],
  typename testu::params::ActivationD xu_port[testu::params::N][testu::params::G * testu::params::R]) {
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl
#pragma HLS INTERFACE s_axilite port=num_refinements bundle=ctrl
#pragma HLS INTERFACE m_axi port=x_port offset=slave depth=testu::params::I
#pragma HLS INTERFACE m_axi port=u_port offset=slave depth=testu::params::R*testu::params::PrunedSizeU
#pragma HLS INTERFACE m_axi port=xu_port offset=slave depth=testu::params::R
#pragma HLS DATAFLOW
  svd::SvdStreams<testu::params> streams;
  svd::SvdBuffers<testu::params> buffers;
  svd::InputDMA<testu::params>(num_refinements, x_port, streams, buffers);
  svd::StreamSplitter(num_refinements * testu::params::G * testu::params::PrunedSizeU, u_port, streams.u_dma);
  U_Dispatcher:
  for (int i = 0; i < num_refinements; ++i) {
    for (int j = 0; j < testu::params::PeU; ++j) {
      for (int k = 0; k < testu::params::PrunedSizeU / testu::params::PeU; ++k) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN
        for (int g = 0; g < testu::params::G; ++g) {
          streams.u[g][j].write(streams.u_dma[g].read());
        }
      }
    }
  }
  svd::KernelU<testu::params>(num_refinements, streams);
  for (int i = 0; i < num_refinements; ++i) {
    for (int j = 0; j < testu::params::N; ++j) {
#pragma HLS PIPELINE II=1
      for (int k = 0; k < testu::params::G; ++k) {
        auto tmp = hlsutils::adder_tree<testu::params::AccumulationD, testu::params::PeU>(streams.xu[j][k]);
        xu_port[j][k * num_refinements + i] = tmp;
      }
    }
  }
}
#else
void HlsVectorKernelU(const int num_refinements,
  hls::stream<hls::vector<typename testu::params::ActivationD, testu::params::Tu> > &x_port,
  hls::stream<hls::vector<typename testu::params::ActivationD, testu::params::Tu> > &u_port,
  hls::stream<hls::vector<typename testu::params::ActivationD, testu::params::N> > &xu_port) {
  const int R_test = num_refinements;
  const int kNumTilesU = testu::params::I / testu::params::Tu;
  const int kDepth_X = testu::params::N * kNumTilesU;
  const int kDepth_U = num_refinements * kNumTilesU * testu::params::G;
  const int kDepth_XU = num_refinements * testu::params::G;

// #pragma HLS INTERFACE m_axi port=x_port bundle=x offset=slave
// #pragma HLS INTERFACE m_axi port=u_port bundle=u offset=slave
// #pragma HLS INTERFACE m_axi port=xu_port bundle=xu offset=slave
// #pragma HLS INTERFACE s_axilite port=x_port
// #pragma HLS INTERFACE s_axilite port=u_port
// #pragma HLS INTERFACE s_axilite port=xu_port

#pragma HLS INTERFACE axis port=x_port bundle=x_dmem
#pragma HLS INTERFACE axis port=u_port bundle=u_dmem
#pragma HLS INTERFACE axis port=xu_port bundle=xu_dmem

#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE s_axilite port=num_refinements
#pragma HLS DATAFLOW
  typedef typename testu::params::ActivationD ActivationType;
  typedef hls::vector<ActivationType, testu::params::N> VectN_Type;

  hls::stream<testu::params::VectTuType> x_streams[testu::params::N];
  hls::stream<testu::params::VectTuType> u_streams[testu::params::G];
  hls::stream<testu::params::VectTuType> xu_streams[testu::params::N][testu::params::G];
  testu::params::VectTuType x_buffer[testu::params::N][kNumTilesU];
  testu::params::VectTuType xu[testu::params::N][testu::params::G];
#pragma HLS STREAM variable=x_streams depth=2
#pragma HLS STREAM variable=u_streams depth=2
#pragma HLS STREAM variable=xu_streams depth=2
#pragma HLS ARRAY_PARTITION variable=x_streams complete dim=0
#pragma HLS ARRAY_PARTITION variable=u_streams complete dim=0
#pragma HLS ARRAY_PARTITION variable=xu_streams complete dim=0
#pragma HLS ARRAY_PARTITION variable=x_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=xu complete dim=0
  
  Store_X_Buffer:
  for (int i = 0; i < testu::params::N; ++i) {
    for (int j = 0; j < kNumTilesU; ++j) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN
      x_buffer[i][j] = x_port.read(); // [i * kNumTilesU + j];
    }
  }
  Stream_X_Tiles:
  for (int i = 0; i < R_test; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=testu::params::R max=testu::params::R
    for (int j = 0; j < kNumTilesU; ++j) {
#pragma HLS PIPELINE II=1
      for (int k = 0; k < testu::params::N; ++k) {
        x_streams[k] << x_buffer[k][j];
      }
    }
  }
  U_DMA:
  for (int i = 0; i < R_test; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=testu::params::R max=testu::params::R
    for (int j = 0; j < kNumTilesU; ++j) {
      for (int k = 0; k < testu::params::G; ++k) {
#pragma HLS PIPELINE II=1
        int u_idx = i * kNumTilesU * testu::params::G + j * testu::params::G + k;
        u_streams[k] << u_port.read(); // [u_idx];
      }
    }
  }
  U_Kernel:
  for (int i = 0; i < R_test; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=testu::params::R max=testu::params::R
    for (int j = 0; j < kNumTilesU; ++j) {
#pragma HLS PIPELINE II=1
      testu::params::VectTuType x[testu::params::N];
#pragma HLS ARRAY_PARTITION variable=x complete dim=0
      for (int ii = 0; ii < testu::params::N; ++ii) {
        x[ii] = x_streams[ii].read();
      }
      for (int k = 0; k < testu::params::G; ++k) {
        testu::params::VectTuType u = u_streams[k].read();
        for (int ii = 0; ii < testu::params::N; ++ii) {
          if (j == 0) {
            xu[ii][k] = testu::params::VectTuType(0);
          }
          xu[ii][k] += u * x[ii];
          if (j == kNumTilesU - 1) {
            xu_streams[ii][k] << xu[ii][k];
          }
        }
      }
    }
  }
  XU_DMA:
  for (int i = 0; i < R_test; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=testu::params::R max=testu::params::R
#pragma HLS PIPELINE II=1
    for (int j = 0; j < testu::params::G; ++j) {
      VectN_Type xu_out;
      for (int k = 0; k < testu::params::N; ++k) {
        xu_out[k] = xu_streams[k][j].read().reduce_add();
      }
      // xu_port[i * testu::params::G + j] = xu_out;
      xu_port << xu_out;
    }
  }
}

void HlsAxisKernelU(const int num_refinements,
  hls::stream<typename testu::VectTuAxiType>& x_port,
  hls::stream<typename testu::VectTuAxiType>& u_port,
  hls::stream<typename testu::VectGN_AxiType>& xu_port) {
  const int R_test = num_refinements;
  const int kNumTilesU = testu::params::I / testu::params::Tu;
  const int kStreamDepth_X = 2 + kNumTilesU * testu::params::N;
  const int kStreamDepth_U = 8 + kNumTilesU * testu::params::N;
  const int kStreamDepth_XU = 2 + testu::params::G;
#pragma HLS INTERFACE axis port=x_port
#pragma HLS INTERFACE axis port=u_port
#pragma HLS INTERFACE axis port=xu_port
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE s_axilite port=num_refinements
#pragma HLS DATAFLOW
  typedef typename testu::params::ActivationD ActivationType;

  auto x_axis = svd::AxiStreamInterface<testu::VectTuAxiBitwidth>(x_port);
  auto u_axis = svd::AxiStreamInterface<testu::VectTuAxiBitwidth>(u_port);
  auto xu_axis = svd::AxiStreamInterface<testu::VectGN_AxiBitwidth>(xu_port);

  hls::stream<testu::params::VectTuType> x_streams[testu::params::N];
  hls::stream<testu::params::VectTuType> u_streams[testu::params::G];
  hls::stream<testu::params::VectTuType> xu_streams[testu::params::N][testu::params::G];
  testu::params::VectTuType x_buffer[testu::params::N][kNumTilesU];
#pragma HLS STREAM variable=x_streams depth=kStreamDepth_X
#pragma HLS STREAM variable=u_streams depth=kStreamDepth_U
#pragma HLS STREAM variable=xu_streams depth=kStreamDepth_XU
#pragma HLS ARRAY_PARTITION variable=x_streams complete dim=1
#pragma HLS ARRAY_PARTITION variable=u_streams complete dim=1
#pragma HLS ARRAY_PARTITION variable=xu_streams complete dim=1
#pragma HLS ARRAY_PARTITION variable=xu_streams complete dim=2
#pragma HLS ARRAY_PARTITION variable=x_buffer complete dim=1
  
  Store_X_Buffer:
  for (int i = 0; i < testu::params::N; ++i) {
    for (int j = 0; j < kNumTilesU; ++j) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN
      x_buffer[i][j] = x_axis.PopVector<ActivationType, testu::params::Tu>();
    }
  }
  Stream_X_Tiles:
  for (int i = 0; i < R_test; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=testu::params::R max=testu::params::R
    for (int j = 0; j < kNumTilesU; ++j) {
#pragma HLS PIPELINE II=1
      for (int k = 0; k < testu::params::N; ++k) {
        x_streams[k] << x_buffer[k][j];
      }
    }
  }

  U_DMA:
  for (int i = 0; i < R_test; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=testu::params::R max=testu::params::R
    for (int j = 0; j < kNumTilesU; ++j) {
      for (int k = 0; k < testu::params::G; ++k) {
#pragma HLS PIPELINE II=1
        u_streams[k] << u_axis.PopVector<ActivationType, testu::params::Tu>();
      }
    }
  }

  U_Kernel:
  for (int i = 0; i < R_test; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=testu::params::R max=testu::params::R
    for (int j = 0; j < kNumTilesU; ++j) {
#pragma HLS PIPELINE II=1
      testu::params::VectTuType x[testu::params::N];
      testu::params::VectTuType xu[testu::params::N][testu::params::G];
#pragma HLS ARRAY_PARTITION variable=x complete dim=0
#pragma HLS ARRAY_PARTITION variable=xu complete dim=0
      for (int ii = 0; ii < testu::params::N; ++ii) {
        x[ii] = x_streams[ii].read();
      }
      for (int k = 0; k < testu::params::G; ++k) {
        testu::params::VectTuType u = u_streams[k].read();
        for (int ii = 0; ii < testu::params::N; ++ii) {
          if (j == 0) {
            xu[ii][k] = testu::params::VectTuType(0);
          }
          xu[ii][k] += u * x[ii];
          if (j == kNumTilesU - 1) {
            xu_streams[ii][k] << xu[ii][k];
          }
        }
      }
    }
  }
  XU_DMA:
  for (int i = 0; i < R_test; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=testu::params::R max=testu::params::R
#pragma HLS PIPELINE II=1
    testu::params::VectGN_Type xu_out;
    for (int j = 0; j < testu::params::G; ++j) {
      for (int k = 0; k < testu::params::N; ++k) {
        xu_out[j * testu::params::N + k] = xu_streams[k][j].read().reduce_add();
      }
    }
    const bool kIsLast = (i == R_test - 1) ? true : false;
    xu_axis.PushVector<ActivationType, testu::params::G * testu::params::N>(xu_out, kIsLast);
  }
}


void HlsKernelU_ManySampling(const bool pad_output,
  const hls::vector<int, testu::params::N> num_refinements,
  hls::stream<typename testu::params::VectTuAxiType>& x_port,
  hls::stream<typename testu::params::VectTuAxiType>& u_port,
  hls::stream<typename testu::params::VectG_AxiType>& xu_port) {
#pragma HLS INTERFACE axis port=x_port
#pragma HLS INTERFACE axis port=u_port
#pragma HLS INTERFACE axis port=xu_port
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE s_axilite port=pad_output
#pragma HLS INTERFACE s_axilite port=num_refinements
#pragma HLS DATAFLOW
  int R_max = num_refinements[testu::params::N - 1];
  int R_total = num_refinements[0] * testu::params::N; // Total elements.
  Get_Total_R:
  for (int i = 1; i < testu::params::N; ++i) {
#pragma HLS PIPELINE II=1
    if (num_refinements[i] > R_max) {
      R_max = num_refinements[i];
    }
    R_total += (num_refinements[i] - num_refinements[i - 1]) * (testu::params::N - i);
  }
  // std::cout << "\tR_max:   " << R_max << std::endl;
  // std::cout << "\tR_total: " << R_total << std::endl;

  /*
   Ideally, if Rs are ordered, it would be: R0 * N + (R1-R0) * (N-1) + (R2-R1) * (N-2)

   Imagine we have: R0 = 2, R1 = 3, R2 = 6
  
   This means:
   - till refinement 2 we have input 0 to process
   - till refinement 3 we have input 1 to process
   - till refinement 6 we have input 2 to process

  So it would become:

    R_total = 2 * 3 + (3-2) * (3-1) + (6-3) * (3-2)

  */
  const int kNumTilesU = testu::params::I / testu::params::Tu;
  const int kStreamDepth_X = 2 + kNumTilesU * testu::params::N;
  const int kStreamDepth_U = 8 + kNumTilesU * testu::params::N;
  const int kStreamDepth_XU = 2 + testu::params::G;
  typedef typename testu::params::ActivationD ActivationType;

  auto x_axis = svd::AxiStreamInterface<testu::params::VectTuAxiWidth>(x_port);
  auto u_axis = svd::AxiStreamInterface<testu::params::VectTuAxiWidth>(u_port);
  auto xu_axis = svd::AxiStreamInterface<testu::params::VectG_AxiWidth>(xu_port);

  hls::stream<testu::params::VectTuType> x_stream("x_stream");
  hls::stream<testu::params::VectTuType> u_streams[testu::params::G];
  hls::stream<ActivationType> xu_streams[testu::params::G];
  testu::params::VectTuType x_buffer[testu::params::N][kNumTilesU] = {0};
#pragma HLS STREAM variable=x_stream depth=kStreamDepth_X
#pragma HLS STREAM variable=u_streams depth=kStreamDepth_U
#pragma HLS STREAM variable=xu_streams depth=kStreamDepth_XU
#pragma HLS ARRAY_PARTITION variable=u_streams complete dim=1
#pragma HLS ARRAY_PARTITION variable=x_buffer complete dim=1
#pragma HLS BIND_STORAGE variable=x_buffer type=ram_t2p impl=bram latency=2
  
  assert(num_refinements >= 1);

  int R_prev = 0;
  X_DMA:
  for (int ii = 0; ii < testu::params::N; ++ii) {
    Stream_X_Tiles:
    for (int i = 0; i < num_refinements[ii] - R_prev; ++i) {
      assert(num_refinements[ii] - R_prev >= 1);
      for (int j = 0; j < kNumTilesU; ++j) {
        for (int k = 0; k < testu::params::N - ii; ++k) {
#pragma HLS PIPELINE II=1
          if (ii == 0 && i == 0) {
            auto x_val = x_axis.PopVector<ActivationType, testu::params::Tu>();
#pragma HLS AGGREGATE variable=x_val
            x_buffer[k][j] = x_val;
            x_stream << x_val;
          } else {
            assert(k + ii < testu::params::N);
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
#pragma HLS LOOP_TRIPCOUNT min=testu::params::R max=testu::params::R
    for (int j = 0; j < kNumTilesU; ++j) {
      for (int k = 0; k < testu::params::G; ++k) {
        auto u_val = u_axis.PopVector<ActivationType, testu::params::Tu>();
        for (int ii = 0; ii < testu::params::N; ++ii) {
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
      for (int k = 0; k < testu::params::G; ++k) {
        xu_streams[k] << hlsutils::adder_tree<ActivationType, testu::params::Tu>(x_val * u_streams[k].read());
        // xu_streams[k] << (x_val * u_streams[k].read()).reduce_add();
      }
    }
  }
  int iter_cnt = 0;
  XU_DMA:
  for (int i = 0; i < R_max; ++i) {
    testu::params::VectG_Type xu_out[testu::params::N] = {testu::params::VectG_Type(0)};
#pragma HLS ARRAY_PARTITION variable=xu_out complete dim=1
    for (int j = 0; j < kNumTilesU; ++j) {
      for (int k = 0; k < testu::params::N; ++k) {
#pragma HLS PIPELINE II=1
        for (int ii = 0; ii < testu::params::G; ++ii) {
          if (i < num_refinements[k]) {
            xu_out[k][ii] += xu_streams[ii].read();
#pragma HLS BIND_OP variable=xu_out[k][ii] op=add impl=dsp
            // std::cout << "\t[KernelU] Reading xu[R." << i << "][G." << ii << "][N." << k << "][T." << j << "]" << std::endl;
          }
        }
        if (i < num_refinements[k] && j == kNumTilesU - 1) {
          const bool kIsLast = (iter_cnt == R_total - 1 && !pad_output) ? true : false;
          xu_axis.PushVector<ActivationType, testu::params::G>(xu_out[k], kIsLast);
          // std::cout << "\t[KernelU] Sending xu[R." << i << "][N." << k << "]" << std::endl;
          ++iter_cnt;
        } else if (pad_output) {
          const bool last_condition = i == R_max - 1 && j == kNumTilesU - 1 && k == testu::params::N - 1; 
          const bool kIsLast = (last_condition) ? true : false;
          xu_axis.PushVector<ActivationType, testu::params::G>(xu_out[k], kIsLast);
          ++iter_cnt;
        }
      }
    }
  }
  // std::cout << "[KernelU] iter_cnt: " << iter_cnt << std::endl;
}

#endif

namespace svd {

} // svd