#include "kernel/u_kernel.h"
#include "kernel/gemv_kernel.h"
#include "hls_utils/adder_tree.h"
#include "dma/svd_dma.h"
#include "dma/axis_lib.h"

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


void HlsManySamplingsKernelU(const hls::vector<int, testu::params::N> num_refinements,
  hls::stream<typename testu::VectTuAxiType>& x_port,
  hls::stream<typename testu::VectTuAxiType>& u_port,
  hls::stream<typename testu::VectN_AxiType>& xu_port) {
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
  // NOTE: The number of refinements and the inputs must be ordered.
  const int max_R = num_refinements[testu::params::N - 1]; // -1;
  const int min_R = num_refinements[0]; // 1 << 31;
  auto get_current_R = [&](const int idx) {
    int R = 0;
    for (int i = 0; i < testu::params::N; ++i) {
      R += (idx < num_refinements[i] ? 1 : 0);
    }
    return R;
  };
  int total_R = 0;
  Get_Total_R:
  for (int i = 0; i < max_R; ++i) {
#pragma HLS PIPELINE II=1
    const int R = get_current_R(i);
    int tmp = (kNumTilesU * R + testu::params::N - 1) / testu::params::N; // Ceil 
    std::cout << "(for) cnt: " << tmp << std::endl;
    total_R += tmp;
  }
  std::cout << "total_R: " << total_R << std::endl;
  for (int j = 0; j < testu::params::N; ++j) {
    std::cout << j << ") num_refinements: " << num_refinements[j] << std::endl;
  }

  typedef typename testu::params::ActivationD ActivationType;
  auto x_axis = svd::AxiStreamInterface<testu::VectTuAxiBitwidth>(x_port);
  auto u_axis = svd::AxiStreamInterface<testu::VectTuAxiBitwidth>(u_port);
  auto xu_axis = svd::AxiStreamInterface<testu::VectN_AxiBitwidth>(xu_port);

  hls::stream<testu::params::VectTuType> x_streams[testu::params::N];
  hls::stream<testu::params::VectTuType> u_streams[testu::params::N];
  hls::stream<testu::params::ActivationD> xu_streams[testu::params::N];
  testu::params::VectTuType x_buffer[testu::params::N][kNumTilesU];
#pragma HLS STREAM variable=x_streams depth=kStreamDepth_X
#pragma HLS STREAM variable=u_streams depth=kStreamDepth_U
#pragma HLS STREAM variable=xu_streams depth=kStreamDepth_XU
#pragma HLS ARRAY_PARTITION variable=x_streams complete dim=1
#pragma HLS ARRAY_PARTITION variable=u_streams complete dim=1
#pragma HLS ARRAY_PARTITION variable=xu_streams complete dim=1
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
  for (int g = 0; g < testu::params::G; ++g) {
    for (int i = 0; i < max_R; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=testu::params::R max=testu::params::R
      std::cout << "----------------------------" << std::endl;

      int max_idx = testu::params::N - 1;
      hls::vector<int, testu::params::N> tile_cnt = hls::vector<int, testu::params::N>(0);

//       const int R = (kNumTilesU * get_current_R(i) + testu::params::N - 1) / testu::params::N; // Ceil

//       X_Dispatcher:
//       for (int j = 0; j < R; ++j) {
//         for (int k = 0; k < testu::params::N; ++k) {
// #pragma HLS PIPELINE II=1
//           if (i < num_refinements[k]) { 
//             x_streams[k] << x_buffer[k][tile_cnt[k]];
//             ++tile_cnt[k];
//           } else {
//             tile_cnt[k] = kNumTilesU;
//             if (tile_cnt[max_idx] < kNumTilesU) {
//               x_streams[k] << x_buffer[k][tile_cnt[max_idx]];
//               tile_cnt[max_idx] = (tile_cnt[max_idx] == kNumTilesU) ? kNumTilesU : ++tile_cnt[max_idx];
//             } else {
//               if (max_idx == 0) {
//                 x_streams[j] << testu::params::VectTuType(0);
//               } else {
//                 --max_idx; 
//                 x_streams[k] << x_buffer[k][tile_cnt[max_idx]];
//               }
//             }
//           }
//         }
//       }

/*
 
----------------------------
0) [0]:0 [1]:0 [2]:0 [3]:0              1 1 1 1
0) [0]:1 [1]:1 [2]:1 [3]:1              2 2 2 2
0) [0]:2 [1]:2 [2]:2 [3]:2              3 3 3 3
0) [0]:3 [1]:3 [2]:3 [3]:3              4 4 4 4
(while) cnt: 4
----------------------------
1) [0]:0 [1]:0 [2]:0 [3]:0              1 1 1 1
1) [0]:1 [1]:1 [2]:1 [3]:1              2 2 2 2
1) [0]:2 [1]:2 [2]:2 [3]:2              3 3 3 3
1) [0]:3 [1]:3 [2]:3 [3]:3              4 4 4 4
(while) cnt: 4
----------------------------
2) [3]:0 [1]:0 [2]:0 [3]:1              4 1 1 2
2) [3]:2 [1]:1 [2]:1 [3]:3              4 2 2 4
2) [2]:2 [1]:2 [2]:3 [3]:X <-- ERROR    4 3 4 4
2) [1]:3 [1]:X [2]:X [3]:X              4 4 4 4
(while) cnt: 4
----------------------------
3) [3]:0 [3]:1 [2]:0 [3]:2              4 4 1 3
3) [3]:3 [2]:1 [2]:2 [3]:X              4 4 3 4
3) [2]:3 [1]:X [2]:X [3]:X              4 4 4 4
(while) cnt: 3
----------------------------
4) [3]:0 [3]:1 [3]:2 [3]:3              4 4 4 4
(while) cnt: 1

 */



      int cnt = 0;
      X_Dispatcher:
      while(tile_cnt != hls::vector<int, testu::params::N>(kNumTilesU)) {
      // for (int ii = 0; ii < R; ++ii) {
        ++cnt;
        std::cout << i << ") ";
        for (int j = 0; j < testu::params::N; ++j) {
#pragma HLS PIPELINE II=1

          int curr_idx;
          if (i >= num_refinements[j]) {
            tile_cnt[j] = kNumTilesU;
            if (tile_cnt[max_idx] >= kNumTilesU) {
              --max_idx;
            }
            curr_idx = max_idx;
          } else {
            curr_idx = j;
          }
          std::cout << "[" << curr_idx << "]:" << tile_cnt[curr_idx] << " ";
          auto x_out = (curr_idx >= 0) ? x_buffer[curr_idx][tile_cnt[curr_idx]] : testu::params::VectTuType(0);
          x_streams[j] << x_out;
          tile_cnt[curr_idx] = (tile_cnt[curr_idx] == kNumTilesU) ? tile_cnt[curr_idx] : ++tile_cnt[curr_idx];

          // int curr_idx = j;
          // if (i >= num_refinements[j]) {
          //   tile_cnt[j] = kNumTilesU;
          //   curr_idx = max_idx;
          // }
          // if (tile_cnt[curr_idx] < kNumTilesU) {
          //   std::cout << "[" << curr_idx << "]:" << tile_cnt[curr_idx] << " ";
          //   x_streams[j] << x_buffer[curr_idx][tile_cnt[curr_idx]];
          //   ++tile_cnt[curr_idx];
          // } else {
          //   if (tile_cnt[max_idx] >= kNumTilesU) {
          //     tile_cnt[max_idx] = kNumTilesU;
          //     --max_idx;
          //   }
          //   if (max_idx != -1) {
          //     std::cout << "[" << max_idx << "]:" << tile_cnt[max_idx] << " ";
          //     x_streams[j] << x_buffer[max_idx][tile_cnt[max_idx]];
          //     ++tile_cnt[max_idx];
          //   } else {
          //     std::cout << "[" << max_idx << "]:" << "X ";
          //     x_streams[j] << testu::params::VectTuType(0);
          //   }
          // }
        }
        std::cout << "\t\t";
        for (int j = 0; j < testu::params::N; ++j) {
          std::cout << tile_cnt[j] << " ";
        }
        std::cout << std::endl;
      }
      std::cout << "(while) cnt: " << cnt << std::endl;

    }
  }


//   U_DMA:
//   for (int i = 0; i < max_R; ++i) {
// #pragma HLS LOOP_TRIPCOUNT min=testu::params::R max=testu::params::R
//     for (int j = 0; j < kNumTilesU; ++j) {
//       for (int k = 0; k < testu::params::G; ++k) {
// #pragma HLS PIPELINE II=1
//         auto u_val = u_axis.PopVector<ActivationType, testu::params::Tu>();
//         for (int ii = 0; ii < testu::params::N; ++ii) {
//           u_streams[ii] << u_val;
//         }
//       }
//     }
//   }


  U_DMA:
  for (int g = 0; g < testu::params::G; ++g) {
    for (int i = 0; i < max_R; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=testu::params::R max=testu::params::R
      std::cout << "----------------------------" << std::endl;

      testu::params::VectTuType u_buffer[kNumTilesU];
#pragma HLS ARRAY_PARTITION variable=u_buffer complete dim=1

      Store_U:
      for (int j = 0; j < kNumTilesU; ++j) {
#pragma HLS PIPELINE II=1
        u_buffer[j] = u_axis.PopVector<ActivationType, testu::params::Tu>();
      }


      int max_idx = testu::params::N - 1;
      hls::vector<int, testu::params::N> tile_cnt = hls::vector<int, testu::params::N>(0);

//       U_Dispatcher:
//       while(tile_cnt != hls::vector<int, testu::params::N>(kNumTilesU)) {
//         // std::cout << i << ") ";
//         for (int j = 0; j < testu::params::N; ++j) {
// #pragma HLS PIPELINE II=1
//           int curr_idx = j;
//           if (i >= num_refinements[j]) {
//             tile_cnt[j] = kNumTilesU;
//             curr_idx = max_idx;
//           }
//           if (tile_cnt[curr_idx] < kNumTilesU) {
//             // std::cout << "[" << curr_idx << "]:" << tile_cnt[curr_idx] << " ";
//             u_streams[j] << u_buffer[tile_cnt[curr_idx]];
//             ++tile_cnt[curr_idx];
//           } else {
//             // std::cout << "[" << curr_idx << "]:" << "X ";
//             u_streams[j] << testu::params::VectTuType(0);
//           }
//           // if (tile_cnt[max_idx] >= kNumTilesU) {
//           //   tile_cnt[max_idx] = kNumTilesU;
//           //   max_idx = (max_idx > 0) ? --max_idx : max_idx;
//           // }
//           while (tile_cnt[max_idx] >= kNumTilesU && max_idx > 0) {
//             tile_cnt[max_idx] = kNumTilesU;
//             --max_idx;
//           }
//         }
//         // std::cout << "\t\t";
//         // for (int j = 0; j < testu::params::N; ++j) {
//         //   std::cout << tile_cnt[j] << " ";
//         // }
//         // std::cout << std::endl;
//       }


      int cnt = 0;
      U_Dispatcher:
      while(tile_cnt != hls::vector<int, testu::params::N>(kNumTilesU)) {
      // for (int ii = 0; ii < R; ++ii) {
        ++cnt;
        std::cout << i << ") max: " << max_idx << " | ";
        for (int j = 0; j < testu::params::N; ++j) {
#pragma HLS PIPELINE II=1
          if (tile_cnt[max_idx] >= kNumTilesU) {
            --max_idx;
          }
          int curr_idx = j;
          if (i >= num_refinements[curr_idx]) {
            tile_cnt[curr_idx] = kNumTilesU;
            curr_idx = max_idx;
          }
          std::cout << "[" << curr_idx << "]:" << ((curr_idx >= 0) ? tile_cnt[curr_idx] : -1) << " ";
          auto u_out = (curr_idx >= 0) ? u_buffer[tile_cnt[curr_idx]] : testu::params::VectTuType(0);
          u_streams[j] << u_out;
          tile_cnt[curr_idx] = (tile_cnt[curr_idx] == kNumTilesU) ? tile_cnt[curr_idx] : ++tile_cnt[curr_idx];

          // int curr_idx = j;
          // if (i >= num_refinements[j]) {
          //   tile_cnt[j] = kNumTilesU;
          //   curr_idx = max_idx;
          // }
          // if (tile_cnt[curr_idx] < kNumTilesU) {
          //   std::cout << "[" << curr_idx << "]:" << tile_cnt[curr_idx] << " ";
          //   u_streams[j] << u_buffer[tile_cnt[curr_idx]];
          //   ++tile_cnt[curr_idx];
          // } else {
          //   if (tile_cnt[max_idx] >= kNumTilesU) {
          //     tile_cnt[max_idx] = kNumTilesU;
          //     --max_idx;
          //   }
          //   if (max_idx != -1) {
          //     std::cout << "[" << max_idx << "]:" << tile_cnt[max_idx] << " ";
          //     u_streams[j] << u_buffer[tile_cnt[max_idx]];
          //     ++tile_cnt[max_idx];
          //   } else {
          //     std::cout << "[" << max_idx << "]:" << "X ";
          //     u_streams[j] << testu::params::VectTuType(0);
          //   }
          // }
        }
        std::cout << "\t\t";
        for (int j = 0; j < testu::params::N; ++j) {
          std::cout << tile_cnt[j] << " ";
        }
        std::cout << std::endl;
      }
      std::cout << "(while) cnt: " << cnt << std::endl;



    }
  }

  for (int j = 0; j < testu::params::N; ++j) {
    // std::cout << j << ") x_streams.size: " << x_streams[j].size() << std::endl;
    // std::cout << j << ") u_streams.size: " << u_streams[j].size() << std::endl;
  }

  // std::cout << "total_R: " << total_R << std::endl;
  
  Kernel:
  for (int i = 0; i < testu::params::G * total_R; ++i) {
#pragma HLS PIPELINE II=1    
    for (int j = 0; j < testu::params::N; ++j) {
      xu_streams[j] << (x_streams[j].read() * u_streams[j].read()).reduce_add();
    }
  }

  for (int j = 0; j < testu::params::N; ++j) {
    // std::cout << j << ") xu_streams.size: " << xu_streams[j].size() << std::endl;
  }

  XU_DMA:
  for (int g = 0; g < testu::params::G; ++g) {
    for (int i = 0; i < max_R; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=testu::params::R max=testu::params::R
      // std::cout << "----------------------------" << std::endl;

      typename testu::params::VectN_Type xu_out = testu::params::VectN_Type(0);

      int max_idx = testu::params::N - 1;
      hls::vector<int, testu::params::N> tile_cnt = hls::vector<int, testu::params::N>(0);

//       while(tile_cnt != hls::vector<int, testu::params::N>(kNumTilesU)) {
// #pragma HLS PIPELINE II=1
//         // std::cout << i << ") ";
//         for (int j = 0; j < testu::params::N; ++j) {
//           int curr_idx = j;
//           if (i >= num_refinements[j]) {
//             tile_cnt[j] = kNumTilesU;
//             curr_idx = max_idx;
//           }
//           xu_out[curr_idx] += xu_streams[j].read();
//           if (tile_cnt[curr_idx] < kNumTilesU) {
//             // std::cout << "[" << curr_idx << "]:" << tile_cnt[curr_idx] << " ";
//             ++tile_cnt[curr_idx];
//           }
//           // if (tile_cnt[max_idx] >= kNumTilesU) {
//           //   tile_cnt[max_idx] = kNumTilesU;
//           //   max_idx = (max_idx > 0) ? --max_idx : max_idx;
//           // }
//           while (tile_cnt[max_idx] >= kNumTilesU && max_idx > 0) {
//             tile_cnt[max_idx] = kNumTilesU;
//             --max_idx;
//           }
//         }
//       }


      int cnt = 0;
      XU_Fetcher:
      while(tile_cnt != hls::vector<int, testu::params::N>(kNumTilesU)) {
      // for (int ii = 0; ii < R; ++ii) {
        ++cnt;
        std::cout << i << ") ";
        for (int j = 0; j < testu::params::N; ++j) {
#pragma HLS PIPELINE II=1
          auto xu_val = xu_streams[j].read();

          int curr_idx = j;
          if (i >= num_refinements[j]) {
            tile_cnt[j] = kNumTilesU;
            curr_idx = max_idx;
          }
          if (tile_cnt[curr_idx] < kNumTilesU) {
            std::cout << "[" << curr_idx << "]:" << tile_cnt[curr_idx] << " ";
            // u_streams[j] << u_buffer[tile_cnt[curr_idx]];
            xu_out[tile_cnt[curr_idx]] += xu_val;
            ++tile_cnt[curr_idx];
          } else {
            if (tile_cnt[max_idx] >= kNumTilesU) {
              tile_cnt[max_idx] = kNumTilesU;
              --max_idx;
            }
            if (max_idx != -1) {
              std::cout << "[" << max_idx << "]:" << tile_cnt[max_idx] << " ";
              xu_out[tile_cnt[max_idx]] += xu_val;
              // u_streams[j] << u_buffer[tile_cnt[max_idx]];
              ++tile_cnt[max_idx];
            } else {
              xu_out[tile_cnt[max_idx]] += xu_val;
              std::cout << "[" << max_idx << "]:" << "X ";
              // u_streams[j] << testu::params::VectTuType(0);
            }
          }
        }
        std::cout << "\t\t";
        for (int j = 0; j < testu::params::N; ++j) {
          std::cout << tile_cnt[j] << " ";
        }
        std::cout << std::endl;
      }
      std::cout << "(while) cnt: " << cnt << std::endl;



      const bool kIsLast = (i == max_R * testu::params::G - 1) ? true : false;
      xu_axis.PushVector<ActivationType, testu::params::N>(xu_out, kIsLast);
    }
  }
}

#endif

namespace svd {

} // svd