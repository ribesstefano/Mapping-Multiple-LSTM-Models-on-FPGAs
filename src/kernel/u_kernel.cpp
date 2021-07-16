#include "kernel/u_kernel.h"
#include "hls_utils/adder_tree.h"
#include "dma/svd_dma.h"

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
void HlsKernelU(const int num_refinements,
  hls::vector<typename testu::params::ActivationD, testu::params::N>* x_port,
  hls::vector<typename testu::params::ActivationD, testu::params::G>* u_port,
  hls::vector<typename testu::params::ActivationD, testu::params::N>* xu_port) {
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl
#pragma HLS INTERFACE s_axilite port=num_refinements bundle=ctrl
#pragma HLS INTERFACE m_axi port=x_port offset=slave bundle=x_dmem depth=testu::params::I
#pragma HLS INTERFACE m_axi port=u_port offset=slave bundle=u_dmem depth=num_refinements*testu::params::PrunedSizeU
#pragma HLS INTERFACE m_axi port=xu_port offset=slave bundle=xu_dmem depth=num_refinements*testu::params::G
#pragma HLS DATAFLOW
  typedef hls::vector<typename testu::params::ActivationD, testu::params::N> VectN_Type;
  typedef hls::vector<typename testu::params::ActivationD, testu::params::G> VectG_Type;
  typedef hls::vector<VectG_Type, testu::params::N> VectNG_Type;
  typedef hls::vector<VectN_Type, testu::params::G> VectGN_Type;
  typedef hls::vector<typename testu::params::AccumulationD, testu::params::PeU> VectPe_Type;
  typedef hls::vector<VectNG_Type, testu::params::PeU> VectPeNG_Type;
  typedef hls::vector<VectGN_Type, testu::params::PeU> VectPeGN_Type;
  const int R_test = 8;
  svd::SvdStreams<testu::params> streams;

  VectN_Type x_buffer[testu::params::Tu][testu::params::TuElems];
#pragma HLS ARRAY_PARTITION variable=x_buffer complete dim=1
  
  Store_X_Buffer:
  for (int i = 0; i < testu::params::Tu; ++i) {
    for (int j = 0; j < testu::params::TuElems; ++j) {
#pragma HLS PIPELINE II=1
      x_buffer[i][j] = x_port[i * testu::params::TuElems + j];
    }
  }

// #define U_KERNEL_VECTOR_DESIGN
#ifndef U_KERNEL_VECTOR_DESIGN
  Stream_X_Tiles:
  for (int i = 0; i < R_test; ++i) {
    for (int k = 0; k < testu::params::TuElems; ++k) {
#pragma HLS PIPELINE II=1
      for (int j = 0; j < testu::params::PeU; ++j) {
        for (int g = 0; g < testu::params::G; ++g) {
          auto x_val = x_buffer[j][k];
          for (int ii = 0; ii < testu::params::N; ++ii) {
            streams.x[ii][g][j].write(x_val[ii]);
          }
        }
      }
    }
  }
  U_Dispatcher:
  for (int i = 0; i < R_test; ++i) {
    for (int j = 0; j < testu::params::PeU; ++j) {
      for (int k = 0; k < testu::params::PrunedSizeU / testu::params::PeU; ++k) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN
        auto u_val = u_port[i * testu::params::PrunedSizeU + j * testu::params::PrunedSizeU / testu::params::PeU + k];
        for (int g = 0; g < testu::params::G; ++g) {
          streams.u[g][j].write(u_val[g]);
        }
      }
    }
  }
  svd::KernelU<testu::params>(R_test, streams);
//   typename testu::params::AccumulationD xu[testu::params::N][testu::params::G][testu::params::PeU];
// #pragma HLS ARRAY_PARTITION variable=xu complete dim=0
//   U_Kernel:
//   for (int i = 0; i < R_test; ++i) {
//     for (int j = 0; j < testu::params::PrunedSizeU / testu::params::PeU; ++j) {
// #pragma HLS PIPELINE II=1
//       for (int k = 0; k < testu::params::PeU; ++k) {
//         for (int g = 0; g < testu::params::G; ++g) {
//           auto u = streams.u[g][k].read();
//           for (int ii = 0; ii < testu::params::N; ++ii) {
//             if (j == 0) {
//               xu[ii][g][k] = 0;
//             }
//             xu[ii][g][k] += u * streams.x[ii][g][k].read();
// #pragma HLS RESOURCE variable=xu[ii][g][k] core=DSP48 latency=3
//             if (j == testu::params::PrunedSizeU / testu::params::PeU - 1) {
//               streams.xu[ii][g][k].write(xu[ii][g][k]);
//             }
//           }
//         }
//       }
//     }
//   }
  XU_DMA:
  for (int i = 0; i < R_test; ++i) {
    for (int j = 0; j < testu::params::G; ++j) {
#pragma HLS PIPELINE II=1
      VectN_Type out;
      for (int k = 0; k < testu::params::N; ++k) {
        VectPe_Type xu;
        for (int ii = 0; ii < testu::params::PeU; ++ii) {
          xu[ii] = streams.xu[k][j][ii].read();
        }
        out[k] = testu::params::ActivationD(xu.reduce_add());
      }
      xu_port[i * testu::params::G + j] = out;
    }
  }
#else // U_KERNEL_VECTOR_DESIGN
  // TODO: Rewrite the internal streams in terms of hls::vector<PE>.
  hls::stream<VectN_Type> x_streams[testu::params::PeU];
  Stream_X_Tiles:
  for (int i = 0; i < R_test; ++i) {
    for (int k = 0; k < testu::params::TuElems; ++k) {
#pragma HLS PIPELINE II=1
      for (int j = 0; j < testu::params::PeU; ++j) {
        x_streams[j] << x_buffer[j][k];
      }
    }
  }

  hls::stream<VectG_Type> u_streams[testu::params::PeU];
  U_Dispatcher:
  for (int i = 0; i < R_test; ++i) {
    for (int j = 0; j < testu::params::PeU; ++j) {
      for (int k = 0; k < testu::params::PrunedSizeU / testu::params::PeU; ++k) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN
        u_streams[j] << u_port[i * testu::params::PrunedSizeU + j * testu::params::PrunedSizeU / testu::params::PeU + k];
      }
    }
  }
  hls::stream<VectGN_Type> xu_streams;
  U_Kernel:
  for (int i = 0; i < R_test; ++i) {
#pragma HLS LOOP_MERGE force
    typename testu::params::AccumulationD xu[testu::params::N][testu::params::G][testu::params::PeU] = {0};
#pragma HLS ARRAY_PARTITION variable=xu complete dim=0
    for (int j = 0; j < testu::params::PrunedSizeU / testu::params::PeU; ++j) {
#pragma HLS PIPELINE II=1
      for (int k = 0; k < testu::params::PeU; ++k) {
        VectG_Type u = u_streams[k].read();
        VectN_Type x = x_streams[k].read();
        for (int ii = 0; ii < testu::params::G; ++ii) {
          for (int jj = 0; jj < testu::params::N; ++jj) {
            xu[jj][ii][k] += x[jj] * u[ii];
          }
        }
      }
    }
    VectGN_Type xu_tmp;
    for (int ii = 0; ii < testu::params::G; ++ii) {
      for (int jj = 0; jj < testu::params::N; ++jj) {
#pragma HLS PIPELINE II=1
        typename testu::params::AccumulationD xu_val = 0;
        for (int k = 0; k < testu::params::PeU; ++k) {
          xu_val += xu[jj][ii][k];
        }
        xu_tmp[ii][jj] = xu_val;
      }
    }
    xu_streams << xu_tmp;
  }
  XU_DMA:
  for (int i = 0; i < R_test; ++i) {
#pragma HLS PIPELINE II=1
    VectGN_Type xu = xu_streams.read();
    for (int j = 0; j < testu::params::G; ++j) {
      xu_port[i * testu::params::G + j] = xu[j];
    }
  }
#endif

}
#endif

namespace svd {

} // svd