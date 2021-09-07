/******************************************************************************
*  Copyright (c) 2019 Stefano Ribes.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
*  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
*  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
*  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
*  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
*  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
*  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
*  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
*  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
*  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
*  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
*****************************************************************************/
/******************************************************************************
*
*
* @file        lstm_hardware.cpp
* 
* @author      Stefano Ribes
*
* Library of templated HLS functions for BNN deployment. 
* This file lists a set of functions to access memory mapped values into 
* streams 
*
*****************************************************************************/
#include "svd_params.h"
#include "layers/lstm/hls/lstm_hardware.h"
#include "dma/svd_dma.h"

#include "hls_math.h"
#include "hls_half.h"
#include "assert.h"

#include <iostream>

template <int BramRows, int BramCols>
void print_matrix(const svd::ActivationD matrix[BramRows][BramCols]) {
#ifndef __SYNTHESIS__
  for (int i = 0; i < BramRows; i++) {
    std::cout << "|\t";
    for(int j = 0; j < BramCols; j++) {
      std::cout << matrix[i][j] << "\t";
    }
    std::cout << "|\n";
  }
  std::cout << "-------------------------\n";
#endif
}

namespace svd {

template <int BramRows, int BramCols>
void load_bram(const bool execute, const int row_size, const int col_size,
               const int row_offset, const int col_offset,
               const svd::AxiD *dmem, WeightD bram[BramRows][BramCols]) {
  if (execute) {
    const int kDivider = svd::AxiD::width / FIX_WIDTH;
    for (int i = 0; i < BramRows; ++i) {
      const int dmem_idx = ((row_offset + i) * col_size + col_offset) / kDivider;
      Mem2MemDataWidthConverter<ActivationD, svd::AxiD::width, FIX_WIDTH>(BramCols,
        &dmem[dmem_idx], bram[i]);
    }
    // print_matrix<BramRows, BramCols>(bram);
  }
}

template <int BramRows, int BramCols>
void load_bram(const bool execute, const int row_size, const int col_size,
               const int row_offset, const int col_offset,
               const svd::WeightD *dmem, svd::WeightD bram[BramRows][BramCols]) {
  if (execute) {
    for (int i = 0; i < BramRows; ++i) {
      if (BramCols == 1) {
// NOTE: If we are dealing with a vector, the next pipeline directive will be
// ignored and no burst will be generated.
#pragma HLS PIPELINE II=1
      }
      const int dmem_idx = (row_offset + i) * col_size + col_offset;
      for (int j = 0; j < BramCols; ++j) {
#pragma HLS PIPELINE II=1
        bram[i][j] = dmem[dmem_idx + j];
      }
    }
    // print_matrix<BramRows, BramCols>(bram);
  }
}

template <int BramRows, int BramCols>
void store_bram(const bool execute, const int row_size, const int col_size,
                const int row_offset, const int col_offset,
                const svd::WeightD bram[BramRows][BramCols], AxiD *dmem) {
  if (execute) {
    const int kDivider = AxiD::width / FIX_WIDTH;
    for (int i = 0; i < BramRows; ++i) {
      const int dmem_idx = ((row_offset + i) * col_size + col_offset) / kDivider;
      Mem2MemDataWidthConverter<ActivationD, FIX_WIDTH, AxiD::width>(BramCols,
        bram[i], &dmem[dmem_idx]);
    }
    print_matrix<BramRows, BramCols>(bram);
//     AxiD val_tmp = 0;
//     int write_counter = 0;
//     int write_idx = 0;
// #if USE_FIX
//     const int kDivider = AxiD::width / WeightD::width;
//     assert(kDivider % 2 == 0);
// #endif
//     for (int i = 0; i < BramRows; ++i) {
//       for (int j = 0; j < BramCols; ++j) {
// #pragma HLS PIPELINE II=1
// #if USE_FIX
//         const int kHi = ((write_counter  % kDivider) + 1) * WeightD::width - 1;
//         const int kLo = (write_counter % kDivider) * WeightD::width;
//         val_tmp.range(kHi, kLo) = bram[i][j].range();
//         write_counter++;
//         if (write_counter % kDivider == 0) {
//           const int dmem_idx = write_idx;
//           dmem[dmem_idx] = val_tmp;
//           write_idx++;
//         }
// #endif
//       }
//     }
  }
}

template <int BramRows, int BramCols>
void store_bram(const bool execute, const int row_size, const int col_size,
                const int row_offset, const int col_offset,
                const WeightD bram[BramRows][BramCols], WeightD *dmem) {
  if (execute) {
    for (int i = 0; i < BramRows; ++i) {
      if (BramCols == 1) {
// NOTE: If we are dealing with a vector, the next pipeline directive will be
// ignored and no burst will be generated.
#pragma HLS PIPELINE II=1
      }
      const int dmem_idx = (row_offset + i) * col_size + col_offset;
      for (int j = 0; j < BramCols; ++j) {
#pragma HLS PIPELINE II=1
        dmem[dmem_idx + j] = bram[i][j];
        // std::cout << dmem_idx + j << " ";
      }
    }
    // std::cout << "\n";
    // print_matrix<BramRows, BramCols>(bram);
  }
}

template <int BramRows, int BramCols>
void store_fifo(const bool execute, const bool refill_fifo, const int row_size,
                const int col_size, const int row_offset, const int col_offset,
                hls::stream<WeightD> *fifo, WeightD *dmem) {
  if (execute) {
    for (int i = 0; i < BramRows; ++i) {
      if (BramCols == 1) {
// NOTE: If we are dealing with a vector, the next pipeline directive will be
// ignored and no burst will be generated.
#pragma HLS PIPELINE II=1
      }
      const int dmem_idx = (row_offset + i) * col_size + col_offset;
      for (int j = 0; j < BramCols; ++j) {
#pragma HLS PIPELINE II=1
        auto fifo_elem = fifo[i].read();
        dmem[dmem_idx + j] = fifo_elem;
        if (refill_fifo) {
          fifo[i].write(fifo_elem);
        }
      }
    }
  }
}

template <typename DataType, int M, int N>
void accum_bram(const ActivationD c_in[M][N],
                ActivationD c_out[M][N]) {
// #pragma HLS ARRAY_PARTITION variable=c_in complete dim=2
// #pragma HLS ARRAY_PARTITION variable=c_out complete dim=2
#pragma HLS RESOURCE variable=c_out core=RAM_T2P_BRAM
#pragma HLS RESOURCE variable=c_in core=RAM_T2P_BRAM

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
#pragma HLS PIPELINE II=1
      c_out[i][j] += c_in[i][j];
#pragma HLS DEPENDENCE variable=c_out[i][j] intra false
#pragma HLS DEPENDENCE variable=c_out[i][j] inter false
// #pragma HLS RESOURCE variable=c_out[i][j] core=DSP48 latency=3
    }
  }
}

template <typename DataType, typename AccumType, int M, int N, int K>
void gemm_kernel(const bool execute,
                 const bool accumulate,
                 const DataType a[M][K],
                 const DataType b[K][N],
                 DataType c[M][N],
                 const bool use_stream = false,
                 hls::stream<DataType> *c_stream = nullptr,
                 const int architecture_type = 0) {
  if (execute) {
    switch (architecture_type) {
      case 0: {

#if 1
#pragma HLS ARRAY_PARTITION variable=a complete dim=2
#pragma HLS ARRAY_PARTITION variable=b complete dim=1
// #pragma HLS RESOURCE variable=c core=RAM_T2P_BRAM
          m_kernel: for(int i = 0; i < M; ++i) {
            n_kernel: for(int j = 0; j < N; ++j) {
#pragma HLS PIPELINE II=1
              AccumType c_acc = 0;
              k_kernel: for(int k = 0; k < K; ++k) {
                c_acc += a[i][k] * b[k][j];
#pragma HLS RESOURCE variable=c_acc core=DSP48 latency=3
              }
              if (use_stream) {
                if (accumulate) {
                  c_acc += c_stream->read();
                  c_stream->write(c_acc);
                } else {
                  c_stream->write(c_acc);
                }
              } else {
                if (accumulate) {
                  c[i][j] += c_acc;
                } else {
                  c[i][j] = c_acc;
                }
              }
            }
          }
#else
#pragma HLS ARRAY_PARTITION variable=a complete dim=2
#pragma HLS ARRAY_PARTITION variable=b complete dim=1
#pragma HLS ARRAY_PARTITION variable=c complete dim=2
// #pragma HLS RESOURCE variable=c core=RAM_T2P_BRAM
          m_kernel: for(int i = 0; i < M; ++i) {
     
            AccumType acc[N];
#pragma HLS ARRAY_PARTITION variable=acc complete
            
            hls::stream<AccumType> acc_stream;
#pragma HLS STREAM variable=acc_stream depth=N

            k_kernel: for(int k = 0; k < K; ++k) {
// #pragma HLS UNROLL
              const auto a_reg = a[i][k];

              n_kernel: for(int j = 0; j < N; ++j) {
#pragma HLS PIPELINE II=1
                auto prev = (k == 0) ? ((accumulate) ? c[i][j] : DataType(0)) : acc[j];
                acc[j] = prev + a_reg * b[k][j];
#pragma HLS DEPENDENCE variable=acc[j] intra false // useless
#pragma HLS DEPENDENCE variable=acc[j] inter false // useless
// #pragma HLS RESOURCE variable=acc[j] core=DSP48 latency=3
//                 auto init_val = (accumulate) ? c[i][j] : DataType(0);
//                 auto prev = (k == 0) ? init_val : acc_stream.read();
//                 auto acc_val = prev + a_reg * b[k][j];
// // #pragma HLS DEPENDENCE variable=prev intra false
// // #pragma HLS DEPENDENCE variable=acc_val intra false
// // #pragma HLS RESOURCE variable=acc_val core=DSP48 latency=3
//                 acc_stream.write(acc_val);
// // #pragma HLS DEPENDENCE variable=acc_stream intra false
// // #pragma HLS DEPENDENCE variable=acc_stream inter false
              }

            }
            n_store_kernel: for (int j = 0; j < N; ++j) {
// #pragma HLS UNROLL
#pragma HLS PIPELINE II=1
              c[i][j] = acc[j];
              // c[i][j] = acc_stream.read();
            }
          }
#endif
        }
        break;
      case 1: {
#ifndef __VITIS_HLS__
          if (accumulate) {
#pragma HLS DATAFLOW
            ActivationD c_tmp[M][N];
            hls::matrix_multiply_top<hls::NoTranspose, hls::NoTranspose,
              M, K, K, N,
              M, N, MatrixConfigFixCurrent, ActivationD,
              ActivationD>(a, b, c_tmp);
              accum_bram<ActivationD, M, N>(c_tmp, c);
          } else {
            hls::matrix_multiply_top<hls::NoTranspose, hls::NoTranspose,
              M, K, K, N,
              M, N, MatrixConfigFixCurrent, ActivationD,
              ActivationD>(a, b, c);
          }
#endif
        }
        break;
      case 2: {
        // =====================================================================
        // Not working yet.
        // =====================================================================
        // Perform systolic matrix multiply
        // local matrices localA and localB have been partitioned in dimensions
        // 1 and 2 respectively. local matrix C has been partitioned completely
        //
        // This partitioning enables to access MAX_SIZE elements in parallel in
        // the local matrices. Because of the mode of access of array elements,
        // we are able to perform MAX_SIZE*MAX_SIZE operations in parallel.
        //
        // Note : i, j and k loops are interchanged.
        //
        // The top loop systolic1 runs only for a_col iterations instead of
        // MAX_SIZE like the inner loops. The inner loops have fixed loop
        // iteration counts to enable complete unroll
        //
        // The following diagram explains how the matrix multiply happens
        //
        //        B_0        B_1        B_2        B_3
        //         |          |          |          |
        //         v          v          v          v
        //        ___        ___        ___        ___
        //       |   |      |   |      |   |      |   |
        //  A0_->|C00| ---->|C01| ---->|C02| ---->|C03|
        //       |___|      |___|      |___|      |___|
        //         |          |          |          |
        //        _v_        _v_        _v_        _v_
        //       |   |      |   |      |   |      |   |
        //  A1_->|C10| ---->|C11| ---->|C12| ---->|C13|
        //       |___|      |___|      |___|      |___|
        //         |          |          |          |
        //        _v_        _v_        _v_        _v_
        //       |   |      |   |      |   |      |   |
        //  A2_->|C20| ---->|C21| ---->|C21| ---->|C21|
        //       |___|      |___|      |___|      |___|
        //         |          |          |          |
        //        _v_        _v_        _v_        _v_
        //       |   |      |   |      |   |      |   |
        //  A3_->|C30| ---->|C31| ---->|C32| ---->|C33|
        //       |___|      |___|      |___|      |___|
        //       
#pragma HLS ARRAY_PARTITION variable=a complete dim=1
#pragma HLS ARRAY_PARTITION variable=b complete dim=2
#pragma HLS ARRAY_PARTITION variable=c complete dim=0

        systolic1:
        for (int k = 0; k < K; k++) {
#pragma HLS PIPELINE II=1
          systolic2:
          for (int i = 0; i < M; i++) {
            systolic3:
            for (int j = 0; j < N; j++) {
              // Get previous sum
              ActivationD last = (k == 0) ? ActivationD(0) : c[i][j];

              // Update current sum
              // Handle boundary conditions
              ActivationD a_val = (i < M && k < K) ? a[i][k] : ActivationD(0);
              ActivationD b_val = (k < K && j < N) ? b[k][j] : ActivationD(0);
              ActivationD result = last + a_val * b_val;

              // Write back results
              if (accumulate) {
                c[i][j] += result;
              } else {
                c[i][j] = result;
              }
            }
          }
        }
      }
      break;
    }
  }
}

void svd_fpga_cur_gemm_axi(const AxiD *a, const AxiD *b, AxiD *c) {
  const int kDepthPortA = NUM_TIMESTEPS * INPUT_SIZE / (AxiD::width / FIX_WIDTH);
  const int kDepthPortB = INPUT_SIZE * HIDDEN_SIZE / (AxiD::width / FIX_WIDTH);
  const int kDepthPortC = NUM_TIMESTEPS * HIDDEN_SIZE / (AxiD::width / FIX_WIDTH);
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl
#pragma HLS INTERFACE m_axi port=a offset=slave depth=kDepthPortA bundle=a_dmem
#pragma HLS INTERFACE m_axi port=b offset=slave depth=kDepthPortB bundle=b_dmem
#pragma HLS INTERFACE m_axi port=c offset=slave depth=kDepthPortC bundle=c_dmem

  const int M = TIMESTEPS_SIZE;
  const int N = HIDDEN_SIZE;
  const int K = INPUT_SIZE;

  const int Tm = TIMESTEPS_TILE_SIZE;
  const int Tn = HIDDEN_TILE_SIZE;
  const int Tk = INPUT_TILE_SIZE;

  assert(M % Tm == 0);
  assert(N % Tn == 0);
  assert(K % Tk == 0);

  ActivationD a_bram[2][Tm][Tk];
  ActivationD b_bram[2][Tk][Tn];
  ActivationD c_bram[2][Tm][Tn];

  m_tile: for(int i = 0; i < M; i += Tm) {
    n_tile: for(int j = 0; j < N; j += Tn) {
      k_tile: for(int k = 0; k < K + Tk; k += Tk) {
        bool load_a = true;
        bool load_b = true;
        bool execute_gemm = true;
        bool accumulate_gemm = true;
        if (k == 0) {
          execute_gemm = false;
        }
        if ((k / Tk) == 1) {
          accumulate_gemm = false; // overwrite c, to avoid re-initialization.
        }
        if (k == K) {
          load_a = false;
          load_b = false;
        }
        if ((k / Tk) % 2 == 0) {
          load_bram<Tm, Tk>(load_a, M, K, i, k, a, a_bram[0]);
          load_bram<Tk, Tn>(load_b, K, N, k, j, b, b_bram[0]);
          // gemm_kernel(execute_gemm, accumulate_gemm, a_bram[1], b_bram[1], c_bram[0]);
        } else {
          load_bram<Tm, Tk>(load_a, M, K, i, k, a, a_bram[1]);
          load_bram<Tk, Tn>(load_b, K, N, k, j, b, b_bram[1]);
          // gemm_kernel(execute_gemm, accumulate_gemm, a_bram[0], b_bram[0], c_bram[0]);
        }
      }
      const bool store_c = true;
      store_bram<Tm, Tn>(store_c, M, N, i, j, c_bram[0], c);
    }
  }
}

template <typename DataType, typename AccumType, int M, int N, int K, int Tm, int Tn, int Tk>
void compute_gemm_kernel(const bool execute, const int i, const int j,
                    const ActivationD *a, const ActivationD *b,
                    ActivationD c_bram[Tm][Tn],
                    const bool use_stream = false,
                    hls::stream<ActivationD> *c_fifo = nullptr) {
  assert(M % Tm == 0);
  assert(N % Tn == 0);
  assert(K % Tk == 0);

  ActivationD a_bram[2][Tm][Tk];
  ActivationD b_bram[2][Tk][Tn];
#pragma HLS ARRAY_PARTITION variable=a_bram complete dim=1
#pragma HLS ARRAY_PARTITION variable=b_bram complete dim=1

  if (execute) {
    k_tile: for(int k = 0; k < K + Tk; k += Tk) {
      bool load_a = true;
      bool load_b = true;
      bool execute_gemm = true;
      bool accumulate_gemm = true;
      if (k == 0) {
        execute_gemm = false;
      }
      if ((k / Tk) == 1) {
        accumulate_gemm = false; // overwrite c, to avoid re-initialization.
      }
      if (k == K) {
        load_a = false;
        load_b = false;
      }
      if ((k / Tk) % 2 == 0) {
        load_bram<Tm, Tk>(load_a, M, K, i, k, a, a_bram[0]);
        load_bram<Tk, Tn>(load_b, K, N, k, j, b, b_bram[0]);
        gemm_kernel<DataType, AccumType, Tm, Tn, Tk>(execute_gemm,
          accumulate_gemm, a_bram[1], b_bram[1], c_bram, use_stream, c_fifo);
      } else {
        load_bram<Tm, Tk>(load_a, M, K, i, k, a, a_bram[1]);
        load_bram<Tk, Tn>(load_b, K, N, k, j, b, b_bram[1]);
        gemm_kernel<DataType, AccumType, Tm, Tn, Tk>(execute_gemm,
          accumulate_gemm, a_bram[0], b_bram[0], c_bram, use_stream, c_fifo);
      }
    }
  }
}

template <typename DataType, typename AccumType, int M, int N, int K, int Tm, int Tn, int Tk>
void gemm(const ActivationD *a, const ActivationD *b, ActivationD *c) {
  assert(M % Tm == 0);
  assert(N % Tn == 0);
  assert(K % Tk == 0);

  // TODO(20/02/2020): The design using the internal FIFO hangs in cosimulation.
  const bool kUseInternalFIFO = false;
  ActivationD c_bram[2][Tm][Tn];
#pragma HLS ARRAY_PARTITION variable=c_bram complete dim=1

  hls::stream<ActivationD> c_fifo[2][Tm];
#pragma HLS ARRAY_PARTITION variable=c_fifo complete dim=0
#pragma HLS STREAM variable=c_fifo depth=Tn dim=2

  if (kUseInternalFIFO) {
    for (int i = 0; i < Tn; ++i) {
#pragma HLS PIPELINE II=1
      for (int j = 0; j < Tm; ++j) {
        for (int k = 0; k < 2; ++k) {
          c_fifo[k][j].write(0);
        }
      }
    }
  }

  m_tile: for(int i = 0; i < M; i += Tm) {
    n_tile: for(int j = 0; j < N + Tn; j += Tn) {
      bool execute_kernel = true;
      bool store_c = true;
      bool refill_fifo = true;
      if (i == M - Tm && (j == N || j == N - Tn)) {
        // Do not refill the internal streams in the last two iterations.
        refill_fifo = false;
      }
      if (j == 0) {
        store_c = false;
      }
      if (j == N) {
        execute_kernel = false;
      }
      if ((j / Tn) % 2 == 0) {
        compute_gemm_kernel<DataType, AccumType, M, N, K, Tm, Tn, Tk>(execute_kernel,
          i, j, a, b, c_bram[0], kUseInternalFIFO, c_fifo[0]);
        if (kUseInternalFIFO) {
          store_fifo<Tm, Tn>(store_c, refill_fifo, M, N, i, j - Tn, c_fifo[1], c);
        } else {
          store_bram<Tm, Tn>(store_c, M, N, i, j - Tn, c_bram[1], c);
        }
      } else {
        compute_gemm_kernel<DataType, AccumType, M, N, K, Tm, Tn, Tk>(execute_kernel,
          i, j, a, b, c_bram[1], kUseInternalFIFO, c_fifo[1]);
        if (kUseInternalFIFO) {
          store_fifo<Tm, Tn>(store_c, refill_fifo, M, N, i, j - Tn, c_fifo[0], c);
        } else {
          store_bram<Tm, Tn>(store_c, M, N, i, j - Tn, c_bram[0], c);
        }
      }
    }
  }
}

template <typename DataType, typename AccumType, int M, int N, int K, int Tm, int Tn, int Tk>
void gemm_summa_kernel(hls::stream<DataType> a[Tk], // [M][K],
                       hls::stream<DataType> b[Tk], // [K][N],
                       hls::stream<DataType> &c, // [M][N],
                       const int architecture_type = 0) {
  hls::stream<DataType> c_fifo("c_fifo");
  DataType a_bram[Tm][Tk];
  DataType b_bram[Tk][Tn];
#pragma HLS ARRAY_PARTITION variable=a_bram complete dim=2
#pragma HLS ARRAY_PARTITION variable=b_bram complete dim=1

  const bool kExecuteKernel = true;
  const bool kAccumulate = true;
  const bool kUseInternalFIFO = true;

  for (int calls = 0; calls < K / Tk; ++calls) {
    a_dma:
    for (int i = 0; i < Tm; ++i) {
#pragma HLS PIPELINE II=1
      for (int j = 0; j < Tk; ++j) {
        a_bram[i][j] = a[j].read();
      }
    }

    b_dma:
    for (int i = 0; i < Tn; ++i) {
#pragma HLS PIPELINE II=1
      for (int j = 0; j < Tk; ++j) {
        b_bram[j][i] = b[j].read();
      }
    }
    gemm_kernel<DataType, AccumType, Tm, Tn, Tk>(kExecuteKernel, (calls == 0 ? false : true),
      a_bram, b_bram, nullptr, kUseInternalFIFO, &c_fifo);
  }

  c_dma:
  for (int i = 0; i < Tm; ++i) {
    for (int j = 0; j < Tn; ++j) {
#pragma HLS PIPELINE II=1
      c.write(c_fifo.read());
    }
  }

//  for (int calls = 0; calls < K / Tk; ++calls) {    
//     m_kernel: for(int i = 0; i < Tm; ++i) {
//       n_kernel: for(int j = 0; j < Tn; ++j) {
// #pragma HLS PIPELINE II=1
//         AccumType c_acc = 0;
//         k_kernel: for(int k = 0; k < Tk; ++k) {
//           c_acc += a[k].read() * b[k].read();
//   #pragma HLS RESOURCE variable=c_acc core=DSP48 latency=3
//           // std::cout << "a/b/c size: " << a[k].size() << "/" << b[k].size() << "/" << c_fifo.size() << "\n";
//         }
//         if (calls == 0) {
//           c_fifo.write(c_acc);
//         } else if (calls == K / Tk - 1) {
//           c.write(c_acc + c_fifo.read());
//         } else {
//           auto c_prev = c_fifo.read();
//           c_fifo.write(c_prev + c_acc);
//         }
//       }
//     }
//   }
}

template <typename DataType, typename AccumType, int M, int N, int K, int Tm, int Tn, int Tk>
void gemm_summa(const ActivationD *a, const ActivationD *b, ActivationD *c) {
  assert(M % Tm == 0);
  assert(N % Tn == 0);
  assert(K % Tk == 0);
#pragma HLS DATAFLOW

  hls::stream<DataType> a_fifo[M / Tm][N / Tn][Tk];
  hls::stream<DataType> b_fifo[M / Tm][N / Tn][Tk];
  hls::stream<DataType> c_fifo[M / Tm][N / Tn];

  std::cout << "starting A transfer\n";
  a_dma:
  for (int i = 0; i < M; i += Tm) {
    for (int j = 0; j < K; j += Tk) {
      for (int ii = 0; ii < Tm; ++ii) {
        for (int jj = 0; jj < Tk; ++jj) {
#pragma HLS PIPELINE II=1
          auto a_val = a[(i / Tm + ii) * K + j + jj];
            std::cout << "writing a[" << (i / Tm + ii) * K + j + jj << "] to fifo\n";

          // Broadcast the value to the PEs (row-wise).
          for (int col = 0; col < N / Tn; ++col) {
            a_fifo[i / Tm][col][jj].write(a_val);
            // std::cout << "writing to a_fifo[" << i / Tm << "][" << col << "][" << jj << "]\n";
          }
        }
      }
    }
  }

  std::cout << "starting B transfer\n";

  b_dma:
  for (int i = 0; i < K; i += Tk) {
    for (int j = 0; j < N; j += Tn) {
      for (int ii = 0; ii < Tk; ++ii) {
        for (int jj = 0; jj < Tn; ++jj) {
#pragma HLS PIPELINE II=1
          auto b_val = b[(i / Tk + ii) * N + j + jj];
            std::cout << "writing b[" << (i / Tk + ii) * N + j + jj << "] to fifo[" << 0 << "]\n";
          // DataType b_val = 0;
          // Broadcast the value to the PEs (column-wise).
          for (int row = 0; row < M / Tm; ++row) {
            b_fifo[row][j / Tn][ii].write(b_val);
            // std::cout << "\twriting to b_fifo[" << row << "][" << j / Tn << "][" << ii << "]\n";
          }
        }
      }
    }
  }
  // for (int i = 0; i < Tm; ++i) {
  //   for (int j = 0; j < Tn; ++j) {
  //     for (int k = 0; k < Tk; ++k) {
  //       std::cout << "a_fifo[" << i << "][" << j << "][" << k << "].size: " << a_fifo[i][j][k].size() << "\n";
  //       std::cout << "b_fifo[" << i << "][" << j << "][" << k << "].size: " << b_fifo[i][j][k].size() << "\n";
  //     }
  //   }
  // }

  std::cout << "starting PEs\n";
  PE:
  for (int i = 0; i < M / Tm; ++i) {
    for (int j = 0; j < N / Tn; ++j) {
#pragma HLS UNROLL
      gemm_summa_kernel<DataType, AccumType, M, N, K, Tm, Tn, Tk>(a_fifo[i][j],
        b_fifo[i][j], c_fifo[i][j]);
    }
  }

  // for (int i = 0; i < Tm; ++i) {
  //   for (int j = 0; j < Tn; ++j) {
  //     std::cout << "c_fifo[" << i << "][" << j << "].size: " << c_fifo[i][j].size() << " (expected: " << Tm * Tn << ")\n";
  //   }
  // }

  std::cout << "starting C transfer\n";
  c_dma:
  for (int i = 0; i < M; i += Tm) {
    for (int j = 0; j < N; j += Tn) {
      for (int ii = 0; ii < Tm; ++ii) {
        for (int jj = 0; jj < Tn; ++jj) {
#pragma HLS PIPELINE II=1
          c[(i / Tm + ii) * N + j + jj] = c_fifo[i / Tm][j / Tn].read();
        }
      }
    }
  }
}


template <typename DataType, typename AccumType, int M, int N, int K, int Tm, int Tn, int Tk>
void gemm_systolic_kernel(const int row_id,
                          const int col_id,
                          hls::stream<DataType> a_in[Tk], // [M][K],
                          hls::stream<DataType> a_out[Tk], // [M][K],
                          hls::stream<DataType> b_in[Tk], // [K][N],
                          hls::stream<DataType> b_out[Tk], // [K][N],
                          hls::stream<DataType> &c) {
}

template <typename DataType, typename AccumType, int M, int N, int K, int Tm, int Tn, int Tk>
void gemm_systolic(const ActivationD *a, const ActivationD *b, ActivationD *c) {
  assert(M % Tm == 0);
  assert(N % Tn == 0);
  assert(K % Tk == 0);
#pragma HLS DATAFLOW

  hls::stream<DataType> a_fifo[M / Tm][N / Tn][Tk];
  hls::stream<DataType> b_fifo[M / Tm][N / Tn][Tk];
  hls::stream<DataType> c_fifo[M / Tm][N / Tn];

  std::cout << "starting A transfer\n";
  a_dma:
  for (int i = 0; i < M; i += Tm) {
    for (int j = 0; j < K; j += Tk) {
      for (int ii = 0; ii < Tm; ++ii) {
        for (int jj = 0; jj < Tk; ++jj) {
#pragma HLS PIPELINE II=1
          auto a_val = a[(i / Tm + ii) * K + j + jj];
            std::cout << "writing a[" << (i / Tm + ii) * K + j + jj << "] to fifo\n";

          // Broadcast the value to the PEs (row-wise).
          for (int col = 0; col < N / Tn; ++col) {
            a_fifo[i / Tm][col][jj].write(a_val);
            // std::cout << "writing to a_fifo[" << i / Tm << "][" << col << "][" << jj << "]\n";
          }
        }
      }
    }
  }

  std::cout << "starting B transfer\n";

  b_dma:
  for (int i = 0; i < K; i += Tk) {
    for (int j = 0; j < N; j += Tn) {
      for (int ii = 0; ii < Tk; ++ii) {
        for (int jj = 0; jj < Tn; ++jj) {
#pragma HLS PIPELINE II=1
          auto b_val = b[(i / Tk + ii) * N + j + jj];
            std::cout << "writing b[" << (i / Tk + ii) * N + j + jj << "] to fifo[" << 0 << "]\n";
          // DataType b_val = 0;
          // Broadcast the value to the PEs (column-wise).
          for (int row = 0; row < M / Tm; ++row) {
            b_fifo[row][j / Tn][ii].write(b_val);
            // std::cout << "\twriting to b_fifo[" << row << "][" << j / Tn << "][" << ii << "]\n";
          }
        }
      }
    }
  }
  // for (int i = 0; i < Tm; ++i) {
  //   for (int j = 0; j < Tn; ++j) {
  //     for (int k = 0; k < Tk; ++k) {
  //       std::cout << "a_fifo[" << i << "][" << j << "][" << k << "].size: " << a_fifo[i][j][k].size() << "\n";
  //       std::cout << "b_fifo[" << i << "][" << j << "][" << k << "].size: " << b_fifo[i][j][k].size() << "\n";
  //     }
  //   }
  // }

  std::cout << "starting PEs\n";
  PE:
  for (int i = 0; i < M / Tm; ++i) {
    for (int j = 0; j < N / Tn; ++j) {
#pragma HLS UNROLL
      gemm_summa_kernel<DataType, AccumType, M, N, K, Tm, Tn, Tk>(a_fifo[i][j],
        b_fifo[i][j], c_fifo[i][j]);
    }
  }

  // for (int i = 0; i < Tm; ++i) {
  //   for (int j = 0; j < Tn; ++j) {
  //     std::cout << "c_fifo[" << i << "][" << j << "].size: " << c_fifo[i][j].size() << " (expected: " << Tm * Tn << ")\n";
  //   }
  // }

  std::cout << "starting C transfer\n";
  c_dma:
  for (int i = 0; i < M; i += Tm) {
    for (int j = 0; j < N; j += Tn) {
      for (int ii = 0; ii < Tm; ++ii) {
        for (int jj = 0; jj < Tn; ++jj) {
#pragma HLS PIPELINE II=1
          c[(i / Tm + ii) * N + j + jj] = c_fifo[i / Tm][j / Tn].read();
        }
      }
    }
  }
}

template <int M, int N>
void gemv_kernel(const bool execute,
                 const bool accumulate,
                 const ActivationD a[M][N],
                 const ActivationD b[N][1],
                 ActivationD c[M][1]) {
  if (execute) {
#if 1

#pragma HLS ARRAY_PARTITION variable=a complete dim=2
#pragma HLS ARRAY_PARTITION variable=b complete dim=0
// #pragma HLS ARRAY_PARTITION variable=c complete dim=0

    ActivationD c_val = 0;
    m_kernel: for(int i = 0; i < M; ++i) {
#pragma HLS PIPELINE II=1
      n_kernel: for(int j = 0; j < N; ++j) {
        if (j == 0) {
          c_val = 0;
        }
        c_val += a[i][j] * b[j][0];
#pragma HLS RESOURCE variable=c_val core=DSP48 latency=3

        if (j == N - 1) {
          if (accumulate) {
            c[i][0] += c_val;
          } else {
            c[i][0] = c_val;
          }
        }
      }
    }
#else
    if (accumulate) {
#pragma HLS DATAFLOW
      ActivationD c_tmp[M][1];
// #pragma HLS RESOURCE variable=c_tmp core=RAM_2P

      hls::matrix_multiply_top<hls::NoTranspose, hls::NoTranspose,
        M, N, N, 1,
        M, 1, MatrixConfigFixRecurrent, ActivationD,
        ActivationD>(a, b, c_tmp);
        accum_bram(c_tmp, c);
    } else {
      hls::matrix_multiply_top<hls::NoTranspose, hls::NoTranspose,
        M, N, N, 1,
        M, 1, MatrixConfigFixCurrent, ActivationD,
        ActivationD>(a, b, c);
    }
#endif
  }
}

template <int M, int N, int Tm, int Tn>
void compute_gemv_kernel(const bool execute, const int i, const ActivationD *a,
    const ActivationD b_bram[N / Tn][Tn][1], ActivationD c_bram[Tm][1]) {
  if (execute) {
    ActivationD a_bram[2][Tm][Tn];
#pragma HLS ARRAY_PARTITION variable=a_bram complete dim=1

    n_tile: for (int j = 0; j < N + Tn; j += Tn) {
      bool load_a = true;
      bool execute_gemv = true;
      bool accumulate_gemv = true;
      if (j == 0) {
        execute_gemv = false;
      }
      if ((j / Tn) == 1) {
        accumulate_gemv = false; // overwrite c, to avoid re-initialization.
      }
      if (j == N) {
        load_a = false;
      }
      if ((j / Tn) % 2 == 0) {
        load_bram<Tm, Tn>(load_a, M, N, i, j, a, a_bram[0]);
        gemv_kernel<Tm, Tn>(execute_gemv, accumulate_gemv, a_bram[1], b_bram[(j - Tn) / Tn], c_bram);
      } else {
        load_bram<Tm, Tn>(load_a, M, N, i, j, a, a_bram[1]);
        gemv_kernel<Tm, Tn>(execute_gemv, accumulate_gemv, a_bram[0], b_bram[(j - Tn) / Tn], c_bram);
      }
    }
  }
}

template <int M, int N, int Tm, int Tn>
void gemv(const ActivationD *a, const ActivationD *b, ActivationD *c,
          const bool writeback_once = false) {
  assert(M % Tm == 0);
  assert(N % Tn == 0);

  ActivationD b_bram[N / Tn][Tn][1];
#pragma HLS ARRAY_PARTITION variable=b_bram complete dim=2
  // NOTE: Before starting computation, load all the vector in local memory.
  const bool kLoadVector = true;
  const int kZeroOffset = 0;
  load_b: for (int i = 0; i < N; i += Tn) {
    load_bram<Tn, 1>(kLoadVector, N, 1, i, kZeroOffset, b, b_bram[i / Tn]);
  }

  if (writeback_once) {
    // =========================================================================
    // Store the WHOLE output vector before writing it back.
    // =========================================================================
    const bool kExecuteKernel = true;
    ActivationD c_bram[M / Tm][Tm][1];

    m: for(int i = 0; i < M; i += Tm) {
      compute_gemv_kernel<M, N, Tm, Tn>(kExecuteKernel, i, a, b_bram, c_bram[i / Tm]);
    }

    writeback_i: for (int i = 0; i < M / Tm; ++i) {
      writeback_j: for (int j = 0; j < Tm; ++j) {
#pragma HLS PIPELINE II=1
        c[i * Tm + j] = c_bram[i][j][0];
      }
    }
  } else {
    // =========================================================================
    // Overlate computation with writebacks.
    // =========================================================================
    ActivationD c_bram[2][Tm][1];
#pragma HLS ARRAY_PARTITION variable=c_bram complete dim=1

    m_tile: for(int i = 0; i < M + Tm; i += Tm) {
      bool execute_kernel = true;
      bool store_c = true;
      if (i == 0) {
        store_c = false;
      }
      if (i == M) {
        execute_kernel = false;
      }
      if ((i / Tm) % 2 == 0) {
        compute_gemv_kernel<M, N, Tm, Tn>(execute_kernel, i, a, b_bram, c_bram[0]);
        store_bram<Tm, 1>(store_c, M, 1, i - Tm, kZeroOffset, c_bram[1], c);
      } else {
        compute_gemv_kernel<M, N, Tm, Tn>(execute_kernel, i, a, b_bram, c_bram[1]);
        store_bram<Tm, 1>(store_c, M, 1, i - Tm, kZeroOffset, c_bram[0], c);
      }
    }
  }
}


template <typename DataType, typename AccumType, int M, int N, int Tm, int Tn>
void gemv_systolic_pe(const int pe_id,
#ifndef __SYNTHESIS__
                      const int i,
#endif
                      hls::stream<ap_uint<DataType::width * Tm> > &a,
                      hls::stream<DataType> &b,
                      hls::stream<ap_uint<DataType::width * Tm> > &c_fwrd_in,
                      hls::stream<ap_uint<DataType::width * Tm> > &c_fwrd_out,
                      hls::stream<ap_uint<DataType::width * Tm> > &c_bwrd_in,
                      hls::stream<ap_uint<DataType::width * Tm> > &c_bwrd_out) {
  // Snake-like architecture: it shows some deadlocks in cosimulation
#if 0
  assert(M % 2 == 0);
  assert(N % 2 == 0);
  assert(M % Tm == 0);
  assert(N % Tn == 0);

  DataType b_reg = 0;

#ifdef __SYNTHESIS__
  hls::stream<ap_uint<DataType::width * Tm> > c_internal("c_internal");
#pragma HLS STREAM variable=c_internal depth=M/Tm
#else
  static hls::stream<ap_uint<DataType::width * Tm> > c_internal[Tn];
#endif

#ifdef __SYNTHESIS__
  for (int i = 0; i < N / Tn; ++i) {
#endif
    for (int j = 0; j < M / Tm; ++j) {
  #pragma HLS PIPELINE II=1
      if (j == 0) {
        b_reg = b.read();
      }
      
      ap_uint<DataType::width * Tm> a_pack = a.read();
      ap_uint<DataType::width * Tm> c_prev_pack = 0;
      ap_uint<DataType::width * Tm> c_curr_pack = 0;

      if (pe_id == 0) {
        if (i != 0) {
          if (i % 2 == 0) {
            // std::cout << "PE n." << pe_id << ": reading from c_internal, step " << i <<"\n";
#ifdef __SYNTHESIS__
            c_prev_pack = c_internal.read();
#else
            c_prev_pack = c_internal[pe_id].read();
#endif
          } else {
            // std::cout << "PE n." << pe_id << ": reading from backward[" << pe_id + 1 << "], step " << i <<"\n";
            c_prev_pack = c_bwrd_in.read();
          }
        }
      } else if (pe_id == Tn - 1) {
        if (i % 2 == 0) {
          // std::cout << "PE n." << pe_id << " (last): reading from forward[" << pe_id << "], step " << i <<"\n";
          c_prev_pack = c_fwrd_in.read();
        } else {
          // std::cout << "PE n." << pe_id << " (last): reading from c_internal, step " << i <<"\n";
#ifdef __SYNTHESIS__
          c_prev_pack = c_internal.read();
#else
          c_prev_pack = c_internal[pe_id].read();
#endif
        }
      } else {
        if (i % 2 == 0) {
          // std::cout << "PE n." << pe_id << ": reading from forward[" << pe_id << "], step " << i <<"\n";
          c_prev_pack = c_fwrd_in.read();
        } else {
          // std::cout << "PE n." << pe_id << ": reading from backward[" << pe_id + 1 << "], step " << i <<"\n";
          c_prev_pack = c_bwrd_in.read();
        }
      }
      for (int k = 0; k < Tm; ++k) {
        const int kHi = DataType::width * (k + 1) - 1;
        const int kLo = DataType::width * k;
        DataType c_prev = 0;
        DataType a_reg = 0;
        c_prev.range() = c_prev_pack.range(kHi, kLo);
        a_reg.range() = a_pack.range(kHi, kLo);
        if (pe_id == 0 || pe_id == Tn - 1) {
          // Since we are writing to the internal FIFO, we need to use a "simpler"
          // fixed point type to avoid II=2.
          ap_fixed<DataType::width, DataType::iwidth> c_acc = c_prev + a_reg * b_reg;
          c_curr_pack.range(kHi, kLo) = c_acc.range();
        } else {
          DataType c_acc = c_prev + a_reg * b_reg;
          c_curr_pack.range(kHi, kLo) = c_acc.range();
        }
      }
      if (pe_id == 0) {
        if (i == N / Tn - 1) {
          // std::cout << "PE n." << pe_id << ": writing to c_backward_out[" << pe_id << "], step " << i <<"\n";
          c_bwrd_out.write(c_curr_pack);
        } else {
          if (i % 2 == 0) {
            // std::cout << "PE n." << pe_id << ": writing to c_forward_out[" << pe_id + 1 << "], step " << i <<"\n";
            c_fwrd_out.write(c_curr_pack);
          } else {
            // std::cout << "PE n." << pe_id << ": writing to c_internal, step " << i <<"\n";
#ifdef __SYNTHESIS__
            c_internal.write(c_curr_pack);
#else
            c_internal[pe_id].write(c_curr_pack);
#endif
          }
        }
      } else if (pe_id == Tn - 1) {
        if (i % 2 == 0) {
          // std::cout << "PE n." << pe_id << " (last): writing to c_internal, step " << i <<"\n";
#ifdef __SYNTHESIS__
          c_internal.write(c_curr_pack);
#else
          c_internal[pe_id].write(c_curr_pack);
#endif
        } else {
          // std::cout << "PE n." << pe_id << " (last): writing to c_backward_out[" << pe_id << "], step " << i <<"\n";
          c_bwrd_out.write(c_curr_pack);
        }
      } else {
        if (i % 2 == 0) {
          // std::cout << "PE n." << pe_id << ": writing to c_forward_out[" << pe_id + 1 << "], step " << i <<"\n";
          c_fwrd_out.write(c_curr_pack);
        } else {
          // std::cout << "PE n." << pe_id << ": writing to c_backward_out[" << pe_id << "], step " << i <<"\n";
          c_bwrd_out.write(c_curr_pack);
        }
      }
    }
#ifdef __SYNTHESIS__
  }
#endif
#endif
}

template <typename DataType, typename AccumType, int M, int N, int Tm, int Tn>
void gemv_systolic_pe(const int pe_id,
#ifndef __SYNTHESIS__
                      const int i,
#endif
                      hls::stream<ap_uint<DataType::width * Tm> > &a,
                      hls::stream<DataType> &b,
                      hls::stream<ap_uint<DataType::width * Tm> > &c_in,
                      hls::stream<ap_uint<DataType::width * Tm> > &c_out,
                      hls::stream<ap_uint<DataType::width * Tm> > &c_final) {
  assert(M % Tm == 0);
  assert(N % Tn == 0);
  
  DataType b_reg;
#ifdef __SYNTHESIS__
  for (int i = 0; i < N / Tn; ++i) {
#endif
    for (int j = 0; j < M / Tm; ++j) {
#pragma HLS PIPELINE II=1
      if (j == 0) {
        b_reg = b.read();
      }
      ap_uint<DataType::width * Tm> a_pack = a.read();
      ap_uint<DataType::width * Tm> c_prev_pack = 0;
      ap_uint<DataType::width * Tm> c_curr_pack = 0;
      if (pe_id != 0) {
        c_prev_pack = c_in.read();
      } else {
        if (i != 0) {
          c_prev_pack = c_in.read();
        }
      }
      for (int k = 0; k < Tm; ++k) {
        const int kHi = DataType::width * (k + 1) - 1;
        const int kLo = DataType::width * k;
        DataType c_prev = 0;
        DataType a_reg = 0;
#if USE_FIX
        c_prev.range() = c_prev_pack.range(kHi, kLo);
        a_reg.range() = a_pack.range(kHi, kLo);
        DataType c_acc = c_prev + a_reg * b_reg;
        c_curr_pack.range(kHi, kLo) = c_acc.range();
#else
        c_prev = c_prev_pack.range(kHi, kLo).to_float();
        a_reg = a_pack.range(kHi, kLo).to_float();
        DataType c_acc = c_prev + a_reg * b_reg;
        ap_uint<DataType::width * Tm> c_acc_fix = *((ap_uint<DataType::width * Tm>*) &c_acc);
        c_curr_pack.range(kHi, kLo) = c_acc_fix.range();
#endif
      }
      if (pe_id == Tn - 1 && i == N / Tn - 1) {
        c_final.write(c_curr_pack);
      } else {
        c_out.write(c_curr_pack);
      }
    }
#ifdef __SYNTHESIS__
  }
#endif
}

template <typename DataType, typename AccumType, int M, int N, int Tm, int Tn>
void gemv_systolic(const ap_uint<DataType::width * Tm> *a_transposed,
                   const ap_uint<DataType::width * Tn> *b,
                   ap_uint<DataType::width * Tm> *c) {
// #ifdef SDS_DESIGN
//   const int kDepthFifoPortA = M * N / (DataType::width / Tm);
//   const int kDepthFifoPortB = N / (DataType::width / Tn);
//   const int kDepthFifoPortC = M / (DataType::width / Tm);
// #pragma HLS INTERFACE ap_fifo port=a_transposed depth=kDepthFifoPortA
// #pragma HLS INTERFACE ap_fifo port=b depth=kDepthFifoPortB
// #pragma HLS INTERFACE ap_fifo port=c depth=kDepthFifoPortC
// #endif
#pragma HLS DATAFLOW
  assert(M % Tm == 0);
  assert(N % Tn == 0);

  const int kNumPE = Tn;

  hls::stream<DataType> b_fifo[kNumPE];
  hls::stream<ap_uint<DataType::width * Tm> > a_fifo[kNumPE];
  hls::stream<ap_uint<DataType::width * Tm> > c_fwrd_fifo[kNumPE + 1]; // forward FIFOs
  hls::stream<ap_uint<DataType::width * Tm> > c_bwrd_fifo[kNumPE + 1]; // backward FIFOs

  hls::stream<ap_uint<DataType::width * Tm> > c_fifo[kNumPE + 2];

#pragma HLS ARRAY_PARTITION variable=b_fifo complete
#pragma HLS ARRAY_PARTITION variable=a_fifo complete
#pragma HLS ARRAY_PARTITION variable=c_fifo complete
#if USE_FIX
  const int kLatencyPE = 7;
#else
  const int kLatencyPE = 16; // TODO: check latency for floating point ops.
#endif
  // NOTE: The feedback c FIFO gets filled after L * PEs cycles. A safer value
  // of the depth is M / Tm.
  const int kDepthFifoA = kNumPE * kLatencyPE;
  const int kDepthFifoB = kNumPE; // (N / kNumPE > 0) ? N / kNumPE : 1;
  const int kDepthFifoC = Tm ; //M / Tm; // (M / Tm > 0) ? M / Tm : 1;
#pragma HLS STREAM variable=a_fifo depth=kDepthFifoA
#pragma HLS STREAM variable=b_fifo depth=kDepthFifoB
#pragma HLS STREAM variable=c_fifo depth=kDepthFifoC

  b_DMA:
  for (int i = 0; i < N / kNumPE; ++i) {
#pragma HLS PIPELINE II=1
    ap_uint<DataType::width * kNumPE> b_pack = b[i];
    for (int j = 0; j < kNumPE; ++j) {
      const int kHi = DataType::width * (j + 1) - 1;
      const int kLo = DataType::width * j;
      DataType b_val;
#if USE_FIX
      b_val.range() = b_pack.range(kHi, kLo);
#else
      b_val = b_pack.range(kHi, kLo).to_float();
#endif
      b_fifo[j].write(b_val);
    }
  }

  a_DMA:
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M / Tm; ++j) {
#pragma HLS PIPELINE II=1
      a_fifo[i % kNumPE].write(a_transposed[i * (M / Tm) + j]);
    }
  }

#ifndef __SYNTHESIS__
  for (int j = 0; j < N / kNumPE; ++j) {
    for (int i = 0; i < kNumPE; ++i) {
      if (i == 0) {
        gemv_systolic_pe<DataType, AccumType, M, N, Tm, Tn>(i, j, a_fifo[i], b_fifo[i],
          c_fifo[kNumPE], c_fifo[i + 1], c_fifo[i + 1]);
      } else if (i == kNumPE - 1) {
        gemv_systolic_pe<DataType, AccumType, M, N, Tm, Tn>(i, j, a_fifo[i], b_fifo[i],
          c_fifo[i], c_fifo[kNumPE], c_fifo[kNumPE + 1]);
      } else {
        gemv_systolic_pe<DataType, AccumType, M, N, Tm, Tn>(i, j, a_fifo[i], b_fifo[i],
          c_fifo[i], c_fifo[i + 1], c_fifo[i + 1]);
      }
    }
  }
#else
  PE:
  for (int i = 0; i < kNumPE; ++i) {
#pragma HLS UNROLL
    if (i == 0) {
      gemv_systolic_pe<DataType, AccumType, M, N, Tm, Tn>(i, a_fifo[i], b_fifo[i],
        c_fifo[kNumPE], c_fifo[i + 1], c_fifo[i + 1]);
    } else if (i == kNumPE - 1) {
      gemv_systolic_pe<DataType, AccumType, M, N, Tm, Tn>(i, a_fifo[i], b_fifo[i],
        c_fifo[i], c_fifo[kNumPE], c_fifo[kNumPE + 1]);
    } else {
      gemv_systolic_pe<DataType, AccumType, M, N, Tm, Tn>(i, a_fifo[i], b_fifo[i],
        c_fifo[i], c_fifo[i + 1], c_fifo[i + 1]);
    }
  }
#endif

// #ifndef __SYNTHESIS__
//   for (int j = 0; j < N / kNumPE; ++j) {
//     if (j % 2 == 0) {
//       for (int i = 0; i < kNumPE; ++i) {
//         // std::cout << "STARTING PE N." << i << "\n";
//         gemv_systolic_pe<DataType, AccumType, M, N, Tm, Tn>(i, j, a_fifo[i], b_fifo[i],
//           c_fwrd_fifo[i], c_fwrd_fifo[i + 1], c_bwrd_fifo[i + 1], c_bwrd_fifo[i]);
//       }
//     } else {
//       for (int i = kNumPE - 1; i >= 0; --i) {
//         // std::cout << "STARTING PE N." << i << "\n";
//         gemv_systolic_pe<DataType, AccumType, M, N, Tm, Tn>(i, j, a_fifo[i], b_fifo[i],
//           c_fwrd_fifo[i], c_fwrd_fifo[i + 1], c_bwrd_fifo[i + 1], c_bwrd_fifo[i]);
//       }
//     }
//     // std::cout << "----------------------------\n";
//   }
// #else
//   PE:
//   for (int i = 0; i < kNumPE; ++i) {
// #pragma HLS UNROLL
//     gemv_systolic_pe<DataType, AccumType, M, N, Tm, Tn>(i, a_fifo[i], b_fifo[i],
//       c_fwrd_fifo[i], c_fwrd_fifo[i + 1], c_bwrd_fifo[i], c_bwrd_fifo[i + 1]);
//   }
// #endif
//   c_DMA:
//   for (int i = 0; i < M / Tm; ++i) {
// #pragma HLS PIPELINE II=1
//     c[i] = c_bwrd_fifo[0].read();
//   }

  c_DMA:
  for (int i = 0; i < M / Tm; ++i) {
#pragma HLS PIPELINE II=1
    c[i] = c_fifo[kNumPE + 1].read();
  }
}

} // end namespace svd


#ifdef SDS_DESIGN
#pragma SDS data zero_copy(a[0:HIDDEN_SIZE * HIDDEN_SIZE / HIDDEN_TILE_SIZE])
#pragma SDS data zero_copy(b[0:HIDDEN_SIZE / HIDDEN_TILE_SIZE])
#pragma SDS data zero_copy(c[0:HIDDEN_SIZE / HIDDEN_TILE_SIZE])
#pragma SDS data access_pattern(a:SEQUENTIAL)
#pragma SDS data access_pattern(b:SEQUENTIAL)
#pragma SDS data access_pattern(c:SEQUENTIAL)
#pragma SDS data mem_attribute(a:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(b:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(c:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#endif
void svd_fpga_rec_gemv_gate_systolic(const ap_uint<FIX_WIDTH * HIDDEN_TILE_SIZE> *a,
                                     const ap_uint<FIX_WIDTH * HIDDEN_TILE_SIZE> *b,
                                     ap_uint<FIX_WIDTH * HIDDEN_TILE_SIZE> *c) {
  const int M = HIDDEN_SIZE;
  const int N = HIDDEN_SIZE;

  const int Tm = HIDDEN_TILE_SIZE;
  const int Tn = HIDDEN_TILE_SIZE;

#ifndef SDS_DESIGN
  const int kDepthPortA = M * N / HIDDEN_TILE_SIZE;
  const int kDepthPortB = N / HIDDEN_TILE_SIZE;
  const int kDepthPortC = M / HIDDEN_TILE_SIZE;
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl
// #pragma HLS INTERFACE m_axi port=a offset=slave depth=kDepthPortA bundle=a_dmem
// #pragma HLS INTERFACE m_axi port=b offset=slave depth=kDepthPortB bundle=b_dmem
// #pragma HLS INTERFACE m_axi port=c offset=slave depth=kDepthPortC bundle=c_dmem
#pragma HLS INTERFACE ap_fifo port=a depth=kDepthPortA
#pragma HLS INTERFACE ap_fifo port=b depth=kDepthPortB
#pragma HLS INTERFACE ap_fifo port=c depth=kDepthPortC
#endif
  // svd::gemv_systolic<svd::ActivationD, svd::ActivationD, M, N, Tm, Tn>(a, b, c);
}

#ifdef SDS_DESIGN
#pragma SDS data zero_copy(a[0:HIDDEN_SIZE * INPUT_SIZE / INPUT_TILE_SIZE])
#pragma SDS data zero_copy(b[0:INPUT_SIZE / INPUT_TILE_SIZE])
#pragma SDS data zero_copy(c[0:HIDDEN_SIZE / INPUT_TILE_SIZE])
#pragma SDS data access_pattern(a:SEQUENTIAL)
#pragma SDS data access_pattern(b:SEQUENTIAL)
#pragma SDS data access_pattern(c:SEQUENTIAL)
#pragma SDS data mem_attribute(a:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(b:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(c:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#endif
void svd_fpga_cur_gemv_gate_systolic(const ap_uint<FIX_WIDTH * INPUT_TILE_SIZE> *a,
                                     const ap_uint<FIX_WIDTH * INPUT_TILE_SIZE> *b,
                                     ap_uint<FIX_WIDTH * INPUT_TILE_SIZE> *c) {
  const int M = HIDDEN_SIZE;
  const int N = INPUT_SIZE;

  const int Tm = INPUT_TILE_SIZE;
  const int Tn = INPUT_TILE_SIZE;

#ifndef SDS_DESIGN
  const int kDepthPortA = M * N / INPUT_TILE_SIZE;
  const int kDepthPortB = N / INPUT_TILE_SIZE;
  const int kDepthPortC = M / INPUT_TILE_SIZE;
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl
// #pragma HLS INTERFACE m_axi port=a offset=slave depth=kDepthPortA bundle=a_dmem
// #pragma HLS INTERFACE m_axi port=b offset=slave depth=kDepthPortB bundle=b_dmem
// #pragma HLS INTERFACE m_axi port=c offset=slave depth=kDepthPortC bundle=c_dmem
#pragma HLS INTERFACE ap_fifo port=a depth=kDepthPortA
#pragma HLS INTERFACE ap_fifo port=b depth=kDepthPortB
#pragma HLS INTERFACE ap_fifo port=c depth=kDepthPortC
#endif
  // svd::gemv_systolic<svd::ActivationD, svd::ActivationD, M, N, Tm, Tn>(a, b, c);
}

/**
 * @brief      Perform the matrix-matrix multiplication of the LSTM input with
 *             the current gate: (num_timesteps, input_size) @ (input_size,
 *             hidden_size).
 *
 * @param[in]  a     The LSTM input matrix. Shape: (num_timesteps, input_size)
 * @param[in]  b     The gate matrix. Shape (input_size, hidden_size)
 * @param      c     The output matrix. Shape: (num_timesteps, hidden_size)
 */
#ifdef SDS_DESIGN
#pragma SDS data zero_copy(a[0:TIMESTEPS_SIZE * INPUT_SIZE])
#pragma SDS data zero_copy(b[0:INPUT_SIZE * HIDDEN_SIZE])
#pragma SDS data zero_copy(c[0:TIMESTEPS_SIZE * HIDDEN_SIZE])
#pragma SDS data mem_attribute(a:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(b:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(c:PHYSICAL_CONTIGUOUS)
#endif
void svd_fpga_cur_gemm_gate(const svd::ActivationD *a, const svd::ActivationD *b, svd::ActivationD *c) {
  const int M = TIMESTEPS_SIZE;
  const int N = HIDDEN_SIZE;
  const int K = INPUT_SIZE;

  const int Tm = M; // TIMESTEPS_TILE_SIZE;
  const int Tn = 32; // HIDDEN_TILE_SIZE;
  const int Tk = 64; // INPUT_TILE_SIZE;

#ifndef SDS_DESIGN
  const int kDepthPortA = M * K;
  const int kDepthPortB = K * N;
  const int kDepthPortC = M * N;
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl
#pragma HLS INTERFACE m_axi port=a offset=slave depth=kDepthPortA bundle=a_dmem
#pragma HLS INTERFACE m_axi port=b offset=slave depth=kDepthPortB bundle=b_dmem
#pragma HLS INTERFACE m_axi port=c offset=slave depth=kDepthPortC bundle=c_dmem
#endif
  svd::gemm<svd::ActivationD, svd::ActivationD, M, N, K, Tm, Tn, Tk>(a, b, c);
}

#ifdef SDS_DESIGN
#pragma SDS data zero_copy(a[0:TIMESTEPS_SIZE * INPUT_SIZE])
#pragma SDS data zero_copy(b[0:INPUT_SIZE * HIDDEN_SIZE])
#pragma SDS data zero_copy(c[0:TIMESTEPS_SIZE * HIDDEN_SIZE])
#pragma SDS data mem_attribute(a:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(b:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(c:PHYSICAL_CONTIGUOUS)
#endif
void svd_fpga_cur_gemm_summa_gate(const svd::ActivationD *a, const svd::ActivationD *b, svd::ActivationD *c) {
  const int M = TIMESTEPS_SIZE;
  const int N = HIDDEN_SIZE;
  const int K = INPUT_SIZE;

  const int Tm = 2; // TIMESTEPS_TILE_SIZE;
  const int Tn = 8; // HIDDEN_TILE_SIZE;
  const int Tk = 4; // INPUT_TILE_SIZE;

#ifndef SDS_DESIGN
  const int kDepthPortA = M * K;
  const int kDepthPortB = K * N;
  const int kDepthPortC = M * N;
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl
#pragma HLS INTERFACE m_axi port=a offset=slave depth=kDepthPortA bundle=a_dmem
#pragma HLS INTERFACE m_axi port=b offset=slave depth=kDepthPortB bundle=b_dmem
#pragma HLS INTERFACE m_axi port=c offset=slave depth=kDepthPortC bundle=c_dmem
#endif
  svd::gemm_summa<svd::ActivationD, svd::ActivationD, M, N, K, Tm, Tn, Tk>(a, b, c);
}

/**
 * @brief      Perform the matrix-matrix multiplication of the LSTM input with
 *             the four current gates: (num_timesteps, input_size) @
 *             (input_size, 4 * hidden_size).
 *
 * @param[in]  a     The LSTM input matrix. Shape: (num_timesteps, input_size)
 * @param[in]  b     The gate matrixes. Shape (input_size, hidden_size * 4)
 * @param      c     The output matrix. Shape: (num_timesteps, hidden_size)
 */
#ifdef SDS_DESIGN
#pragma SDS data zero_copy(a[0:TIMESTEPS_SIZE * INPUT_SIZE])
#pragma SDS data zero_copy(b[0:INPUT_SIZE * HIDDEN_SIZE * 4])
#pragma SDS data zero_copy(c[0:TIMESTEPS_SIZE * HIDDEN_SIZE])
#pragma SDS data mem_attribute(a:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(b:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(c:PHYSICAL_CONTIGUOUS)
#endif
void svd_fpga_cur_gemm(const svd::ActivationD *a, const svd::ActivationD *b, svd::ActivationD *c) {
  const int M = TIMESTEPS_SIZE;
  const int N = HIDDEN_SIZE * 4;
  const int K = INPUT_SIZE;

  const int Tm = M; // TIMESTEPS_TILE_SIZE;
  const int Tn = 64; // HIDDEN_TILE_SIZE;
  const int Tk = 128; // INPUT_TILE_SIZE;

#ifndef SDS_DESIGN
  const int kDepthPortA = M * K;
  const int kDepthPortB = K * N;
  const int kDepthPortC = M * N;
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl
#pragma HLS INTERFACE m_axi port=a offset=slave depth=kDepthPortA bundle=a_dmem
#pragma HLS INTERFACE m_axi port=b offset=slave depth=kDepthPortB bundle=b_dmem
#pragma HLS INTERFACE m_axi port=c offset=slave depth=kDepthPortC bundle=c_dmem
#endif
  svd::gemm<svd::ActivationD, svd::ActivationD, M, N, K, Tm, Tn, Tk>(a, b, c);
}

/**
 * @brief      Perform the matrix-vector multiplication of the recurrent gate
 *             matrix with the LSTM input vector: (hidden_size, hidden_size) @
 *             (hidden_size,)
 *
 * @param[in]  a     The gate matrix. Shape (hidden_size, hidden_size)
 * @param[in]  b     The LSTM input vector. Shape: (hidden_size,)
 * @param      c     The output matrix. Shape: (hidden_size, hidden_size)
 */
#ifdef SDS_DESIGN
#pragma SDS data zero_copy(a[0:HIDDEN_SIZE * HIDDEN_SIZE])
#pragma SDS data zero_copy(b[0:HIDDEN_SIZE])
#pragma SDS data zero_copy(c[0:HIDDEN_SIZE])
#pragma SDS data mem_attribute(a:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(b:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(c:PHYSICAL_CONTIGUOUS)
#endif
void svd_fpga_rec_gemv_gate(const svd::ActivationD *a, const svd::ActivationD *b, svd::ActivationD *c) {
  const int M = HIDDEN_SIZE;
  const int N = HIDDEN_SIZE;

  const int Tm = 32; // HIDDEN_TILE_SIZE;
  const int Tn = 32; // HIDDEN_TILE_SIZE;

#ifndef SDS_DESIGN
  const int kDepthPortA = M * N;
  const int kDepthPortB = N;
  const int kDepthPortC = M;
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl
#pragma HLS INTERFACE m_axi port=a offset=slave depth=kDepthPortA bundle=a_dmem
#pragma HLS INTERFACE m_axi port=b offset=slave depth=kDepthPortB bundle=b_dmem
#pragma HLS INTERFACE m_axi port=c offset=slave depth=kDepthPortC bundle=c_dmem
#endif
  svd::gemv<M, N, Tm, Tn>(a, b, c);
}

/**
 * @brief      Perform the matrix-vector multiplication of the recurrent gate
 *             matrixes with the LSTM input vector: (hidden_size, 4 *
 *             hidden_size).T @ (hidden_size,)
 *
 * @param[in]  a     The gate matrixes. Shape (4 * hidden_size, hidden_size)
 * @param[in]  b     The LSTM input vector. Shape: (hidden_size,)
 * @param      c     The output matrix. Shape: (4 * hidden_size, hidden_size)
 */
#ifdef SDS_DESIGN
#pragma SDS data zero_copy(a[0:HIDDEN_SIZE * HIDDEN_SIZE * 4])
#pragma SDS data zero_copy(b[0:HIDDEN_SIZE])
#pragma SDS data zero_copy(c[0:HIDDEN_SIZE])
#pragma SDS data mem_attribute(a:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(b:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(c:PHYSICAL_CONTIGUOUS)
#endif
void svd_fpga_rec_gemv(const svd::ActivationD *a, const svd::ActivationD *b, svd::ActivationD *c) {
  const int M = HIDDEN_SIZE * 4;
  const int N = HIDDEN_SIZE;

  const int Tm = 32; // HIDDEN_TILE_SIZE;
  const int Tn = 32; // HIDDEN_TILE_SIZE;

#ifndef SDS_DESIGN
  const int kDepthPortA = M * N;
  const int kDepthPortB = N;
  const int kDepthPortC = M;
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl
#pragma HLS INTERFACE m_axi port=a offset=slave depth=kDepthPortA bundle=a_dmem
#pragma HLS INTERFACE m_axi port=b offset=slave depth=kDepthPortB bundle=b_dmem
#pragma HLS INTERFACE m_axi port=c offset=slave depth=kDepthPortC bundle=c_dmem
#endif
  svd::gemv<M, N, Tm, Tn>(a, b, c);
}

#ifdef SDS_DESIGN
#pragma SDS data zero_copy(c_rec[0:HIDDEN_SIZE / (AXI_PORT_WIDTH / FIX_WIDTH)])
#pragma SDS data zero_copy(cur_gate_i[0:HIDDEN_SIZE / (AXI_PORT_WIDTH / FIX_WIDTH)])
#pragma SDS data zero_copy(cur_gate_f[0:HIDDEN_SIZE / (AXI_PORT_WIDTH / FIX_WIDTH)])
#pragma SDS data zero_copy(cur_gate_c[0:HIDDEN_SIZE / (AXI_PORT_WIDTH / FIX_WIDTH)])
#pragma SDS data zero_copy(cur_gate_o[0:HIDDEN_SIZE / (AXI_PORT_WIDTH / FIX_WIDTH)])
#pragma SDS data zero_copy(rec_gate_i[0:HIDDEN_SIZE / (AXI_PORT_WIDTH / FIX_WIDTH)])
#pragma SDS data zero_copy(rec_gate_f[0:HIDDEN_SIZE / (AXI_PORT_WIDTH / FIX_WIDTH)])
#pragma SDS data zero_copy(rec_gate_c[0:HIDDEN_SIZE / (AXI_PORT_WIDTH / FIX_WIDTH)])
#pragma SDS data zero_copy(rec_gate_o[0:HIDDEN_SIZE / (AXI_PORT_WIDTH / FIX_WIDTH)])
#pragma SDS data zero_copy(bias_i[0:HIDDEN_SIZE / (AXI_PORT_WIDTH / FIX_WIDTH)])
#pragma SDS data zero_copy(bias_f[0:HIDDEN_SIZE / (AXI_PORT_WIDTH / FIX_WIDTH)])
#pragma SDS data zero_copy(bias_c[0:HIDDEN_SIZE / (AXI_PORT_WIDTH / FIX_WIDTH)])
#pragma SDS data zero_copy(bias_o[0:HIDDEN_SIZE / (AXI_PORT_WIDTH / FIX_WIDTH)])
#pragma SDS data zero_copy(c_cur[0:HIDDEN_SIZE / (AXI_PORT_WIDTH / FIX_WIDTH)])
#pragma SDS data zero_copy(h_port[0:HIDDEN_SIZE / (AXI_PORT_WIDTH / FIX_WIDTH)])
#pragma SDS data access_pattern(c_rec:SEQUENTIAL)
#pragma SDS data access_pattern(cur_gate_i:SEQUENTIAL)
#pragma SDS data access_pattern(cur_gate_f:SEQUENTIAL)
#pragma SDS data access_pattern(cur_gate_c:SEQUENTIAL)
#pragma SDS data access_pattern(cur_gate_o:SEQUENTIAL)
#pragma SDS data access_pattern(rec_gate_i:SEQUENTIAL)
#pragma SDS data access_pattern(rec_gate_f:SEQUENTIAL)
#pragma SDS data access_pattern(rec_gate_c:SEQUENTIAL)
#pragma SDS data access_pattern(rec_gate_o:SEQUENTIAL)
#pragma SDS data access_pattern(bias_i:SEQUENTIAL)
#pragma SDS data access_pattern(bias_f:SEQUENTIAL)
#pragma SDS data access_pattern(bias_c:SEQUENTIAL)
#pragma SDS data access_pattern(bias_o:SEQUENTIAL)
#pragma SDS data access_pattern(c_cur:SEQUENTIAL)
#pragma SDS data access_pattern(h_port:SEQUENTIAL)
#pragma SDS data mem_attribute(c_rec:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(cur_gate_i:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(cur_gate_f:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(cur_gate_c:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(cur_gate_o:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(rec_gate_i:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(rec_gate_f:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(rec_gate_c:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(rec_gate_o:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(bias_i:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(bias_f:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(bias_c:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(bias_o:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(c_cur:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(h_port:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#endif
void svd_fpga_non_lin(const svd::AxiD *c_rec,
                      const svd::AxiD *cur_gate_i,
                      const svd::AxiD *cur_gate_f,
                      const svd::AxiD *cur_gate_c,
                      const svd::AxiD *cur_gate_o,
                      const svd::AxiD *rec_gate_i,
                      const svd::AxiD *rec_gate_f,
                      const svd::AxiD *rec_gate_c,
                      const svd::AxiD *rec_gate_o,
                      const svd::AxiD *bias_i,
                      const svd::AxiD *bias_f,
                      const svd::AxiD *bias_c,
                      const svd::AxiD *bias_o,
                      svd::AxiD *c_cur,
                      svd::AxiD *h_port) {
  const int kWidthDivider = svd::AxiD::width / FIX_WIDTH;
  const int kHiddenDepth = HIDDEN_SIZE / kWidthDivider;
#ifndef SDS_DESIGN
// #pragma HLS INTERFACE s_axilite port=return bundle=ctrl
// #pragma HLS INTERFACE m_axi port=c_rec offset=slave depth=kHiddenDepth bundle=c_rec_dmem
// #pragma HLS INTERFACE m_axi port=cur_gate_i offset=slave depth=kHiddenDepth bundle=cur_gate_i_dmem
// #pragma HLS INTERFACE m_axi port=cur_gate_f offset=slave depth=kHiddenDepth bundle=cur_gate_f_dmem
// #pragma HLS INTERFACE m_axi port=cur_gate_c offset=slave depth=kHiddenDepth bundle=cur_gate_c_dmem
// #pragma HLS INTERFACE m_axi port=cur_gate_o offset=slave depth=kHiddenDepth bundle=cur_gate_o_dmem
// #pragma HLS INTERFACE m_axi port=rec_gate_i offset=slave depth=kHiddenDepth bundle=rec_gate_i_dmem
// #pragma HLS INTERFACE m_axi port=rec_gate_f offset=slave depth=kHiddenDepth bundle=rec_gate_f_dmem
// #pragma HLS INTERFACE m_axi port=rec_gate_c offset=slave depth=kHiddenDepth bundle=rec_gate_c_dmem
// #pragma HLS INTERFACE m_axi port=rec_gate_o offset=slave depth=kHiddenDepth bundle=rec_gate_o_dmem
// #pragma HLS INTERFACE m_axi port=bias_i offset=slave depth=kHiddenDepth bundle=bias_i_dmem
// #pragma HLS INTERFACE m_axi port=bias_f offset=slave depth=kHiddenDepth bundle=bias_f_dmem
// #pragma HLS INTERFACE m_axi port=bias_c offset=slave depth=kHiddenDepth bundle=bias_c_dmem
// #pragma HLS INTERFACE m_axi port=bias_o offset=slave depth=kHiddenDepth bundle=bias_o_dmem
// #pragma HLS INTERFACE m_axi port=c_cur offset=slave depth=kHiddenDepth bundle=c_cur_dmem
// #pragma HLS INTERFACE m_axi port=h_port offset=slave depth=kHiddenDepth bundle=h_port_dmem
#endif
#pragma HLS DATAFLOW

#if (FIX_WIDTH == 16) || (FIX_WIDTH == 8)
  const int kLutSize = 512;
#else
  const int kLutSize = 256;
#endif
  const bool kHasBias = true;
  
  for (int i = 0; i < HIDDEN_SIZE / kWidthDivider; ++i) {
#pragma HLS PIPELINE II=1
    svd::AxiD cur_i_reg = cur_gate_i[i];
    svd::AxiD cur_f_reg = cur_gate_f[i];
    svd::AxiD cur_c_reg = cur_gate_c[i];
    svd::AxiD cur_o_reg = cur_gate_o[i];
    svd::AxiD rec_i_reg = rec_gate_i[i];
    svd::AxiD rec_f_reg = rec_gate_f[i];
    svd::AxiD rec_c_reg = rec_gate_c[i];
    svd::AxiD rec_o_reg = rec_gate_o[i];
    svd::AxiD bias_i_reg = bias_i[i];
    svd::AxiD bias_f_reg = bias_f[i];
    svd::AxiD bias_c_reg = bias_c[i];
    svd::AxiD bias_o_reg = bias_o[i];
    svd::AxiD c_prev_reg = c_rec[i];
    svd::AxiD c_curr_reg = 0;
    svd::AxiD h_curr_reg = 0;

    for (int j = 0; j < kWidthDivider; ++j) {
      const int kHi = FIX_WIDTH * (j + 1) - 1;
      const int kLo = FIX_WIDTH * j;
      svd::ActivationD cur_i_fix, cur_f_fix, cur_c_fix, cur_o_fix;
      svd::ActivationD rec_i_fix, rec_f_fix, rec_c_fix, rec_o_fix;
      svd::ActivationD bias_i_fix, bias_f_fix, bias_c_fix, bias_o_fix;
      svd::ActivationD c_prev_fix;
      svd::ActivationD c_curr_fix;
      svd::ActivationD h_curr_fix;
#if USE_FIX
      cur_i_fix.range() = cur_i_reg.range(kHi, kLo);
      cur_f_fix.range() = cur_f_reg.range(kHi, kLo);
      cur_c_fix.range() = cur_c_reg.range(kHi, kLo);
      cur_o_fix.range() = cur_o_reg.range(kHi, kLo);
      rec_i_fix.range() = rec_i_reg.range(kHi, kLo);
      rec_f_fix.range() = rec_f_reg.range(kHi, kLo);
      rec_c_fix.range() = rec_c_reg.range(kHi, kLo);
      rec_o_fix.range() = rec_o_reg.range(kHi, kLo);
      bias_i_fix.range() = bias_i_reg.range(kHi, kLo);
      bias_f_fix.range() = bias_f_reg.range(kHi, kLo);
      bias_c_fix.range() = bias_c_reg.range(kHi, kLo);
      bias_o_fix.range() = bias_o_reg.range(kHi, kLo);
      c_prev_fix.range() = c_prev_reg.range(kHi, kLo);
      c_curr_fix.range() = c_curr_reg.range(kHi, kLo);
      h_curr_fix.range() = h_curr_reg.range(kHi, kLo);
#else
      auto cur_i_tmp = cur_i_reg.range(kHi, kLo);
      auto cur_f_tmp = cur_f_reg.range(kHi, kLo);
      auto cur_c_tmp = cur_c_reg.range(kHi, kLo);
      auto cur_o_tmp = cur_o_reg.range(kHi, kLo);
      auto rec_i_tmp = rec_i_reg.range(kHi, kLo);
      auto rec_f_tmp = rec_f_reg.range(kHi, kLo);
      auto rec_c_tmp = rec_c_reg.range(kHi, kLo);
      auto rec_o_tmp = rec_o_reg.range(kHi, kLo);
      auto bias_i_tmp = bias_i_reg.range(kHi, kLo);
      auto bias_f_tmp = bias_f_reg.range(kHi, kLo);
      auto bias_c_tmp = bias_c_reg.range(kHi, kLo);
      auto bias_o_tmp = bias_o_reg.range(kHi, kLo);
      auto c_prev_tmp = c_prev_reg.range(kHi, kLo);
      auto c_curr_tmp = c_curr_reg.range(kHi, kLo);
      auto h_curr_tmp = h_curr_reg.range(kHi, kLo);

      cur_i_fix = *((svd::ActivationD*)&cur_i_tmp);
      cur_f_fix = *((svd::ActivationD*)&cur_f_tmp);
      cur_c_fix = *((svd::ActivationD*)&cur_c_tmp);
      cur_o_fix = *((svd::ActivationD*)&cur_o_tmp);
      rec_i_fix = *((svd::ActivationD*)&rec_i_tmp);
      rec_f_fix = *((svd::ActivationD*)&rec_f_tmp);
      rec_c_fix = *((svd::ActivationD*)&rec_c_tmp);
      rec_o_fix = *((svd::ActivationD*)&rec_o_tmp);
      bias_i_fix = *((svd::ActivationD*)&bias_i_tmp);
      bias_f_fix = *((svd::ActivationD*)&bias_f_tmp);
      bias_c_fix = *((svd::ActivationD*)&bias_c_tmp);
      bias_o_fix = *((svd::ActivationD*)&bias_o_tmp);
      c_prev_fix = *((svd::ActivationD*)&c_prev_tmp);
      c_curr_fix = *((svd::ActivationD*)&c_curr_tmp);
      h_curr_fix = *((svd::ActivationD*)&h_curr_tmp);
#endif
      svd::LstmNonLinearFunctions<svd::ActivationD, svd::WeightD, kLutSize>(kHasBias,
        cur_i_fix, cur_f_fix, cur_c_fix, cur_o_fix,
        rec_i_fix, rec_f_fix, rec_c_fix, rec_o_fix,
        bias_i_fix, bias_f_fix, bias_c_fix, bias_o_fix,
        c_prev_fix, c_curr_fix, h_curr_fix);
#if USE_FIX
      c_curr_reg.range(kHi, kLo) = c_curr_fix.range();
      h_curr_reg.range(kHi, kLo) = h_curr_fix.range();
#else
      c_curr_reg.range(kHi, kLo) = *((ap_uint<FIX_WIDTH>*) &c_curr_fix);
      h_curr_reg.range(kHi, kLo) = *((ap_uint<FIX_WIDTH>*) &h_curr_fix);
#endif
    }
    c_cur[i] = c_curr_reg;
    h_port[i] = h_curr_reg;
  }

// #if USE_FIX
//   const int kNumTiles = svd::AxiD::width / svd::ActivationD::width;
//   const int kTileSize = HIDDEN_SIZE / kNumTiles;
// #else
//   const int kNumTiles = 4;
//   const int kTileSize = HIDDEN_SIZE / kNumTiles;
// #endif
//   svd::ActivationStream cur_gate_i_stream[kNumTiles];
//   svd::ActivationStream cur_gate_f_stream[kNumTiles];
//   svd::ActivationStream cur_gate_c_stream[kNumTiles];
//   svd::ActivationStream cur_gate_o_stream[kNumTiles];
//   svd::ActivationStream rec_gate_i_stream[kNumTiles];
//   svd::ActivationStream rec_gate_f_stream[kNumTiles];
//   svd::ActivationStream rec_gate_c_stream[kNumTiles];
//   svd::ActivationStream rec_gate_o_stream[kNumTiles];
//   svd::ActivationStream i_bias_stream[kNumTiles];
//   svd::ActivationStream f_bias_stream[kNumTiles];
//   svd::ActivationStream c_bias_stream[kNumTiles];
//   svd::ActivationStream o_bias_stream[kNumTiles];
//   svd::ActivationStream h_stream[kNumTiles];
//   svd::ActivationStream c_rec_stream[kNumTiles];
//   svd::ActivationStream c_cur_stream[kNumTiles];
// #pragma HLS ARRAY_PARTITION variable=cur_gate_i_stream complete dim=1
// #pragma HLS ARRAY_PARTITION variable=cur_gate_f_stream complete dim=1
// #pragma HLS ARRAY_PARTITION variable=cur_gate_c_stream complete dim=1
// #pragma HLS ARRAY_PARTITION variable=cur_gate_o_stream complete dim=1
// #pragma HLS ARRAY_PARTITION variable=rec_gate_i_stream complete dim=1
// #pragma HLS ARRAY_PARTITION variable=rec_gate_f_stream complete dim=1
// #pragma HLS ARRAY_PARTITION variable=rec_gate_c_stream complete dim=1
// #pragma HLS ARRAY_PARTITION variable=rec_gate_o_stream complete dim=1
// #pragma HLS ARRAY_PARTITION variable=i_bias_stream complete dim=1
// #pragma HLS ARRAY_PARTITION variable=f_bias_stream complete dim=1
// #pragma HLS ARRAY_PARTITION variable=c_bias_stream complete dim=1
// #pragma HLS ARRAY_PARTITION variable=o_bias_stream complete dim=1
// #pragma HLS ARRAY_PARTITION variable=h_stream complete dim=1
// #pragma HLS ARRAY_PARTITION variable=c_rec_stream complete dim=1
// #pragma HLS ARRAY_PARTITION variable=c_cur_stream complete dim=1
// #if USE_FIX
//   svd::ParallelDMA(HIDDEN_SIZE, c_rec, c_rec_stream);
//   svd::ParallelDMA(HIDDEN_SIZE, cur_gate_i, cur_gate_i_stream);
//   svd::ParallelDMA(HIDDEN_SIZE, cur_gate_f, cur_gate_f_stream);
//   svd::ParallelDMA(HIDDEN_SIZE, cur_gate_c, cur_gate_c_stream);
//   svd::ParallelDMA(HIDDEN_SIZE, cur_gate_o, cur_gate_o_stream);
//   svd::ParallelDMA(HIDDEN_SIZE, rec_gate_i, rec_gate_i_stream);
//   svd::ParallelDMA(HIDDEN_SIZE, rec_gate_f, rec_gate_f_stream);
//   svd::ParallelDMA(HIDDEN_SIZE, rec_gate_c, rec_gate_c_stream);
//   svd::ParallelDMA(HIDDEN_SIZE, rec_gate_o, rec_gate_o_stream);
//   svd::ParallelDMA(HIDDEN_SIZE, bias_i, i_bias_stream);
//   svd::ParallelDMA(HIDDEN_SIZE, bias_f, f_bias_stream);
//   svd::ParallelDMA(HIDDEN_SIZE, bias_c, c_bias_stream);
//   svd::ParallelDMA(HIDDEN_SIZE, bias_o, o_bias_stream);
// #else
// #endif
//   for (int i = 0; i < kNumTiles; ++i) {
// #pragma HLS UNROLL
//     svd::NonLinearityUnitTile(kTileSize,
//                               c_rec_stream[i],
//                               cur_gate_i_stream[i],
//                               cur_gate_f_stream[i],
//                               cur_gate_c_stream[i],
//                               cur_gate_o_stream[i],
//                               rec_gate_i_stream[i],
//                               rec_gate_f_stream[i],
//                               rec_gate_c_stream[i],
//                               rec_gate_o_stream[i],
//                               h_stream[i],
//                               c_cur_stream[i],
//                               true,
//                               &i_bias_stream[i],
//                               &f_bias_stream[i],
//                               &c_bias_stream[i],
//                               &o_bias_stream[i],
//                               false,
//                               nullptr);
//   }
// #if USE_FIX
//   svd::ParallelDMA(HIDDEN_SIZE, h_stream, h_port);
//   svd::ParallelDMA(HIDDEN_SIZE, c_cur_stream, c_cur);
// #else
// #endif
}


#ifdef SDS_DESIGN
#pragma SDS data zero_copy(x[0:INPUT_SIZE])
#pragma SDS data zero_copy(h[0:HIDDEN_SIZE])
#pragma SDS data zero_copy(cur_gates[0:4 * INPUT_SIZE * HIDDEN_SIZE])
#pragma SDS data zero_copy(rec_gates[0:4 * HIDDEN_SIZE * HIDDEN_SIZE])
#pragma SDS data copy(i_bias[0:HIDDEN_SIZE])
#pragma SDS data copy(f_bias[0:HIDDEN_SIZE])
#pragma SDS data copy(c_bias[0:HIDDEN_SIZE])
#pragma SDS data copy(o_bias[0:HIDDEN_SIZE])
#pragma SDS data copy(c_rec[0:HIDDEN_SIZE])
#pragma SDS data copy(c_cur[0:HIDDEN_SIZE])
#pragma SDS data copy(out[0:HIDDEN_SIZE])
#pragma SDS data access_pattern(i_bias:SEQUENTIAL)
#pragma SDS data access_pattern(f_bias:SEQUENTIAL)
#pragma SDS data access_pattern(c_bias:SEQUENTIAL)
#pragma SDS data access_pattern(o_bias:SEQUENTIAL)
#pragma SDS data access_pattern(c_rec:SEQUENTIAL)
#pragma SDS data access_pattern(c_cur:SEQUENTIAL)
#pragma SDS data access_pattern(out:SEQUENTIAL)
#pragma SDS data mem_attribute(x:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(h:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(cur_gates:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(rec_gates:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(i_bias:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(f_bias:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(c_bias:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(o_bias:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(c_rec:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(c_cur:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(out:PHYSICAL_CONTIGUOUS)
#endif
void svd_fpga_lstm(const svd::ActivationD *x,
                   const svd::ActivationD *h,
                   const svd::WeightD *cur_gates,
                   const svd::WeightD *rec_gates,
                   const svd::WeightD *i_bias,
                   const svd::WeightD *f_bias,
                   const svd::WeightD *c_bias,
                   const svd::WeightD *o_bias,
                   const svd::ActivationD *c_rec,
                   svd::ActivationD *c_cur,
                   svd::ActivationD *out) {
  const int kInputDepth = INPUT_SIZE;
  const int kHiddenDepth = HIDDEN_SIZE;
  const int kCurGateDepth = 4 * INPUT_SIZE * HIDDEN_SIZE;
  const int kRecGateDepth = 4 * HIDDEN_SIZE * HIDDEN_SIZE;
#ifndef SDS_DESIGN
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl
#pragma HLS INTERFACE m_axi port=x depth=kInputDepth
#pragma HLS INTERFACE m_axi port=h depth=kHiddenDepth
#pragma HLS INTERFACE m_axi port=cur_gates depth=kCurGateDepth
#pragma HLS INTERFACE m_axi port=rec_gates depth=kRecGateDepth
#pragma HLS INTERFACE ap_fifo port=i_bias depth=kHiddenDepth
#pragma HLS INTERFACE ap_fifo port=f_bias depth=kHiddenDepth
#pragma HLS INTERFACE ap_fifo port=c_bias depth=kHiddenDepth
#pragma HLS INTERFACE ap_fifo port=o_bias depth=kHiddenDepth
#pragma HLS INTERFACE ap_fifo port=c_rec depth=kHiddenDepth
#pragma HLS INTERFACE ap_fifo port=c_cur depth=kHiddenDepth
#pragma HLS INTERFACE ap_fifo port=out depth=kHiddenDepth
#endif
#pragma HLS DATAFLOW

  const int kNumGates = 4;
  svd::ActivationD cur_y[HIDDEN_SIZE * kNumGates];
  svd::ActivationD rec_y[HIDDEN_SIZE * kNumGates];
#pragma HLS ARRAY_PARTITION variable=cur_y block factor=kNumGates
#pragma HLS ARRAY_PARTITION variable=rec_y block factor=kNumGates
#pragma HLS STREAM variable=cur_y depth=kHiddenDepth
#pragma HLS STREAM variable=rec_y depth=kHiddenDepth

  const bool kWritebackOnce = true;
  const int kCurM = HIDDEN_SIZE * kNumGates;
  const int kRecM = HIDDEN_SIZE * kNumGates;
  const int kCurN = INPUT_SIZE;
  const int kRecN = HIDDEN_SIZE;
  const int kCurTileSizeM = 64;
  const int kCurTileSizeN = 32;
  const int kRecTileSizeM = 32;
  const int kRecTileSizeN = 32;
  svd::gemv<kCurM, kCurN, kCurTileSizeM, kCurTileSizeN>(cur_gates, x, cur_y, kWritebackOnce);
  svd::gemv<kRecM, kRecN, kRecTileSizeM, kRecTileSizeN>(rec_gates, h, rec_y, kWritebackOnce);

  svd::NonLinearityUnitPE(HIDDEN_SIZE, c_rec,
    &cur_y[0 * HIDDEN_SIZE], &cur_y[1 * HIDDEN_SIZE],
    &cur_y[2 * HIDDEN_SIZE], &cur_y[3 * HIDDEN_SIZE],
    &rec_y[0 * HIDDEN_SIZE], &rec_y[1 * HIDDEN_SIZE],
    &rec_y[2 * HIDDEN_SIZE], &rec_y[3 * HIDDEN_SIZE],
    out, c_cur, true, i_bias, f_bias, c_bias, o_bias);
}

#ifdef SDS_DESIGN
#pragma SDS data zero_copy(x[0:INPUT_SIZE])
#pragma SDS data zero_copy(h[0:HIDDEN_SIZE])
#pragma SDS data zero_copy(cur_i[0:INPUT_SIZE * HIDDEN_SIZE])
#pragma SDS data zero_copy(cur_f[0:INPUT_SIZE * HIDDEN_SIZE])
#pragma SDS data zero_copy(cur_c[0:INPUT_SIZE * HIDDEN_SIZE])
#pragma SDS data zero_copy(cur_o[0:INPUT_SIZE * HIDDEN_SIZE])
#pragma SDS data zero_copy(rec_i[0:HIDDEN_SIZE * HIDDEN_SIZE])
#pragma SDS data zero_copy(rec_f[0:HIDDEN_SIZE * HIDDEN_SIZE])
#pragma SDS data zero_copy(rec_c[0:HIDDEN_SIZE * HIDDEN_SIZE])
#pragma SDS data zero_copy(rec_o[0:HIDDEN_SIZE * HIDDEN_SIZE])
#pragma SDS data copy(i_bias[0:HIDDEN_SIZE])
#pragma SDS data copy(f_bias[0:HIDDEN_SIZE])
#pragma SDS data copy(c_bias[0:HIDDEN_SIZE])
#pragma SDS data copy(o_bias[0:HIDDEN_SIZE])
#pragma SDS data copy(c_rec[0:HIDDEN_SIZE])
#pragma SDS data copy(c_cur[0:HIDDEN_SIZE])
#pragma SDS data copy(out[0:HIDDEN_SIZE])
#pragma SDS data access_pattern(i_bias:SEQUENTIAL)
#pragma SDS data access_pattern(f_bias:SEQUENTIAL)
#pragma SDS data access_pattern(c_bias:SEQUENTIAL)
#pragma SDS data access_pattern(o_bias:SEQUENTIAL)
#pragma SDS data access_pattern(c_rec:SEQUENTIAL)
#pragma SDS data access_pattern(c_cur:SEQUENTIAL)
#pragma SDS data access_pattern(out:SEQUENTIAL)
#pragma SDS data mem_attribute(x:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(h:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(cur_i:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(cur_f:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(cur_c:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(cur_o:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(rec_i:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(rec_f:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(rec_c:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(rec_o:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(i_bias:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(f_bias:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(c_bias:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(o_bias:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(c_rec:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(c_cur:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(out:PHYSICAL_CONTIGUOUS)
#endif
void svd_fpga_lstm_v2(const svd::ActivationD *x,
                      const svd::ActivationD *h,
                      const svd::WeightD *cur_i,
                      const svd::WeightD *cur_f,
                      const svd::WeightD *cur_c,
                      const svd::WeightD *cur_o,
                      const svd::WeightD *rec_i,
                      const svd::WeightD *rec_f,
                      const svd::WeightD *rec_c,
                      const svd::WeightD *rec_o,
                      const svd::WeightD *i_bias,
                      const svd::WeightD *f_bias,
                      const svd::WeightD *c_bias,
                      const svd::WeightD *o_bias,
                      const svd::ActivationD *c_rec,
                      svd::ActivationD *c_cur,
                      svd::ActivationD *out) {
#ifndef SDS_DESIGN
  const int kInputDepth = INPUT_SIZE;
  const int kHiddenDepth = HIDDEN_SIZE;
  const int kCurGateDepth = INPUT_SIZE * HIDDEN_SIZE;
  const int kRecGateDepth = HIDDEN_SIZE * HIDDEN_SIZE;
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl
#pragma HLS INTERFACE m_axi port=x depth=kInputDepth
#pragma HLS INTERFACE m_axi port=h depth=kHiddenDepth
#pragma HLS INTERFACE m_axi port=cur_i depth=kCurGateDepth
#pragma HLS INTERFACE m_axi port=cur_f depth=kCurGateDepth
#pragma HLS INTERFACE m_axi port=cur_c depth=kCurGateDepth
#pragma HLS INTERFACE m_axi port=cur_o depth=kCurGateDepth
#pragma HLS INTERFACE m_axi port=rec_i depth=kRecGateDepth
#pragma HLS INTERFACE m_axi port=rec_f depth=kRecGateDepth
#pragma HLS INTERFACE m_axi port=rec_c depth=kRecGateDepth
#pragma HLS INTERFACE m_axi port=rec_o depth=kRecGateDepth
#pragma HLS INTERFACE ap_fifo port=i_bias depth=kHiddenDepth
#pragma HLS INTERFACE ap_fifo port=f_bias depth=kHiddenDepth
#pragma HLS INTERFACE ap_fifo port=c_bias depth=kHiddenDepth
#pragma HLS INTERFACE ap_fifo port=o_bias depth=kHiddenDepth
#pragma HLS INTERFACE ap_fifo port=c_rec depth=kHiddenDepth
#pragma HLS INTERFACE ap_fifo port=c_cur depth=kHiddenDepth
#pragma HLS INTERFACE ap_fifo port=out depth=kHiddenDepth
#endif
#pragma HLS DATAFLOW

  const int kNumGates = 4;
  svd::ActivationD cur_x[kNumGates][INPUT_SIZE];
  svd::ActivationD rec_h[kNumGates][HIDDEN_SIZE];
  svd::ActivationD cur_y[kNumGates][HIDDEN_SIZE];
  svd::ActivationD rec_y[kNumGates][HIDDEN_SIZE];
#pragma HLS ARRAY_PARTITION variable=cur_x complete dim=1
#pragma HLS ARRAY_PARTITION variable=rec_h complete dim=1
#pragma HLS ARRAY_PARTITION variable=cur_y complete dim=1
#pragma HLS ARRAY_PARTITION variable=rec_y complete dim=1
#pragma HLS STREAM variable=cur_x depth=1
#pragma HLS STREAM variable=rec_h depth=1
#pragma HLS STREAM variable=cur_y depth=16
#pragma HLS STREAM variable=rec_y depth=16

  X_Dispatcher: for (int i = 0; i < INPUT_SIZE; ++i) {
#pragma HLS PIPELINE II=1
    for (int j = 0; j < kNumGates; ++j) {
      cur_x[j][i] = x[i];
    }
  }

  H_Dispatcher: for (int i = 0; i < HIDDEN_SIZE; ++i) {
#pragma HLS PIPELINE II=1
    for (int j = 0; j < kNumGates; ++j) {
      rec_h[j][i] = h[i];
    }
  }

  const bool kWritebackOnce = true;
  const int kCurM = HIDDEN_SIZE;
  const int kRecM = HIDDEN_SIZE;
  const int kCurN = INPUT_SIZE;
  const int kRecN = HIDDEN_SIZE;
  const int kCurTileSizeM = 64;
  const int kCurTileSizeN = 64;
  const int kRecTileSizeM = 64;
  const int kRecTileSizeN = 64;

  svd::gemv<kCurM, kCurN, kCurTileSizeM, kCurTileSizeN>(cur_i, cur_x[0], cur_y[0], kWritebackOnce);
  svd::gemv<kCurM, kCurN, kCurTileSizeM, kCurTileSizeN>(cur_f, cur_x[1], cur_y[1], kWritebackOnce);
  svd::gemv<kCurM, kCurN, kCurTileSizeM, kCurTileSizeN>(cur_c, cur_x[2], cur_y[2], kWritebackOnce);
  svd::gemv<kCurM, kCurN, kCurTileSizeM, kCurTileSizeN>(cur_o, cur_x[3], cur_y[3], kWritebackOnce);

  svd::gemv<kRecM, kRecN, kRecTileSizeM, kRecTileSizeN>(rec_i, rec_h[0], rec_y[0], kWritebackOnce);
  svd::gemv<kRecM, kRecN, kRecTileSizeM, kRecTileSizeN>(rec_f, rec_h[1], rec_y[1], kWritebackOnce);
  svd::gemv<kRecM, kRecN, kRecTileSizeM, kRecTileSizeN>(rec_c, rec_h[2], rec_y[2], kWritebackOnce);
  svd::gemv<kRecM, kRecN, kRecTileSizeM, kRecTileSizeN>(rec_o, rec_h[3], rec_y[3], kWritebackOnce);

  svd::NonLinearityUnitPE(HIDDEN_SIZE, c_rec,
                          cur_y[0], cur_y[1], cur_y[2], cur_y[3],
                          rec_y[0], rec_y[1], rec_y[2], rec_y[3],
                          out, c_cur, true, i_bias, f_bias, c_bias, o_bias);
}

#ifdef SDS_DESIGN
// Array size and zero copy (no DMA overhead)
#pragma SDS data zero_copy(x[0:INPUT_SIZE / (AXI_PORT_WIDTH / FIX_WIDTH)])
#pragma SDS data zero_copy(h[0:HIDDEN_SIZE / (AXI_PORT_WIDTH / FIX_WIDTH)])
#pragma SDS data zero_copy(cur_i_T[0:INPUT_SIZE * HIDDEN_SIZE / (AXI_PORT_WIDTH / FIX_WIDTH)])
#pragma SDS data zero_copy(cur_f_T[0:INPUT_SIZE * HIDDEN_SIZE / (AXI_PORT_WIDTH / FIX_WIDTH)])
#pragma SDS data zero_copy(cur_c_T[0:INPUT_SIZE * HIDDEN_SIZE / (AXI_PORT_WIDTH / FIX_WIDTH)])
#pragma SDS data zero_copy(cur_o_T[0:INPUT_SIZE * HIDDEN_SIZE / (AXI_PORT_WIDTH / FIX_WIDTH)])
#pragma SDS data zero_copy(rec_i_T[0:HIDDEN_SIZE * HIDDEN_SIZE / (AXI_PORT_WIDTH / FIX_WIDTH)])
#pragma SDS data zero_copy(rec_f_T[0:HIDDEN_SIZE * HIDDEN_SIZE / (AXI_PORT_WIDTH / FIX_WIDTH)])
#pragma SDS data zero_copy(rec_c_T[0:HIDDEN_SIZE * HIDDEN_SIZE / (AXI_PORT_WIDTH / FIX_WIDTH)])
#pragma SDS data zero_copy(rec_o_T[0:HIDDEN_SIZE * HIDDEN_SIZE / (AXI_PORT_WIDTH / FIX_WIDTH)])
#pragma SDS data zero_copy(i_bias[0:HIDDEN_SIZE / (AXI_PORT_WIDTH / FIX_WIDTH)])
#pragma SDS data zero_copy(f_bias[0:HIDDEN_SIZE / (AXI_PORT_WIDTH / FIX_WIDTH)])
#pragma SDS data zero_copy(c_bias[0:HIDDEN_SIZE / (AXI_PORT_WIDTH / FIX_WIDTH)])
#pragma SDS data zero_copy(o_bias[0:HIDDEN_SIZE / (AXI_PORT_WIDTH / FIX_WIDTH)])
#pragma SDS data zero_copy(c_rec[0:HIDDEN_SIZE / (AXI_PORT_WIDTH / FIX_WIDTH)])
#pragma SDS data zero_copy(c_cur[0:HIDDEN_SIZE / (AXI_PORT_WIDTH / FIX_WIDTH)])
#pragma SDS data zero_copy(out[0:HIDDEN_SIZE / (AXI_PORT_WIDTH / FIX_WIDTH)])
// Sequential access pattern
#pragma SDS data access_pattern(x:SEQUENTIAL)
#pragma SDS data access_pattern(h:SEQUENTIAL)
#pragma SDS data access_pattern(cur_i_T:SEQUENTIAL)
#pragma SDS data access_pattern(cur_f_T:SEQUENTIAL)
#pragma SDS data access_pattern(cur_c_T:SEQUENTIAL)
#pragma SDS data access_pattern(cur_o_T:SEQUENTIAL)
#pragma SDS data access_pattern(rec_i_T:SEQUENTIAL)
#pragma SDS data access_pattern(rec_f_T:SEQUENTIAL)
#pragma SDS data access_pattern(rec_c_T:SEQUENTIAL)
#pragma SDS data access_pattern(rec_o_T:SEQUENTIAL)
#pragma SDS data access_pattern(i_bias:SEQUENTIAL)
#pragma SDS data access_pattern(f_bias:SEQUENTIAL)
#pragma SDS data access_pattern(c_bias:SEQUENTIAL)
#pragma SDS data access_pattern(o_bias:SEQUENTIAL)
#pragma SDS data access_pattern(c_rec:SEQUENTIAL)
#pragma SDS data access_pattern(c_cur:SEQUENTIAL)
#pragma SDS data access_pattern(out:SEQUENTIAL)
// Allocated with sds_alloc_non_cacheable()
#pragma SDS data mem_attribute(x:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(h:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(cur_i_T:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(cur_f_T:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(cur_c_T:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(cur_o_T:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(rec_i_T:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(rec_f_T:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(rec_c_T:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(rec_o_T:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(i_bias:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(f_bias:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(c_bias:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(o_bias:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(c_rec:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(c_cur:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#pragma SDS data mem_attribute(out:PHYSICAL_CONTIGUOUS|NON_CACHEABLE)
#endif
void svd_fpga_lstm_v3(const svd::AxiD *x,
                      const svd::AxiD *h,
                      const svd::AxiD *cur_i_T,
                      const svd::AxiD *cur_f_T,
                      const svd::AxiD *cur_c_T,
                      const svd::AxiD *cur_o_T,
                      const svd::AxiD *rec_i_T,
                      const svd::AxiD *rec_f_T,
                      const svd::AxiD *rec_c_T,
                      const svd::AxiD *rec_o_T,
                      const svd::AxiD *i_bias,
                      const svd::AxiD *f_bias,
                      const svd::AxiD *c_bias,
                      const svd::AxiD *o_bias,
                      const svd::AxiD *c_rec,
                      svd::AxiD *c_cur,
                      svd::AxiD *out) {
#ifndef SDS_DESIGN
  const int kInputDepth = INPUT_SIZE / (AXI_PORT_WIDTH / FIX_WIDTH);
  const int kHiddenDepth = HIDDEN_SIZE / (AXI_PORT_WIDTH / FIX_WIDTH);
  const int kCurGateDepth = INPUT_SIZE * HIDDEN_SIZE / (AXI_PORT_WIDTH / FIX_WIDTH);
  const int kRecGateDepth = HIDDEN_SIZE * HIDDEN_SIZE / (AXI_PORT_WIDTH / FIX_WIDTH);
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl
#pragma HLS INTERFACE m_axi port=x depth=kInputDepth
#pragma HLS INTERFACE m_axi port=h depth=kHiddenDepth
#pragma HLS INTERFACE m_axi port=cur_i_T depth=kCurGateDepth
#pragma HLS INTERFACE m_axi port=cur_f_T depth=kCurGateDepth
#pragma HLS INTERFACE m_axi port=cur_c_T depth=kCurGateDepth
#pragma HLS INTERFACE m_axi port=cur_o_T depth=kCurGateDepth
#pragma HLS INTERFACE m_axi port=rec_i_T depth=kRecGateDepth
#pragma HLS INTERFACE m_axi port=rec_f_T depth=kRecGateDepth
#pragma HLS INTERFACE m_axi port=rec_c_T depth=kRecGateDepth
#pragma HLS INTERFACE m_axi port=rec_o_T depth=kRecGateDepth
#pragma HLS INTERFACE m_axi port=i_bias depth=kHiddenDepth
#pragma HLS INTERFACE m_axi port=f_bias depth=kHiddenDepth
#pragma HLS INTERFACE m_axi port=c_bias depth=kHiddenDepth
#pragma HLS INTERFACE m_axi port=o_bias depth=kHiddenDepth
#pragma HLS INTERFACE m_axi port=c_rec depth=kHiddenDepth
#pragma HLS INTERFACE m_axi port=c_cur depth=kHiddenDepth
#pragma HLS INTERFACE m_axi port=out depth=kHiddenDepth
#endif
#pragma HLS DATAFLOW

  const int kNumGates = 4;
  const int kDivider = svd::AxiD::width / FIX_WIDTH;
  svd::AxiD cur_x[kNumGates][INPUT_SIZE / kDivider];
  svd::AxiD rec_h[kNumGates][HIDDEN_SIZE / kDivider];
  svd::AxiD cur_y[kNumGates][HIDDEN_SIZE / kDivider];
  svd::AxiD rec_y[kNumGates][HIDDEN_SIZE / kDivider];
#pragma HLS ARRAY_PARTITION variable=cur_x complete dim=1
#pragma HLS ARRAY_PARTITION variable=rec_h complete dim=1
#pragma HLS ARRAY_PARTITION variable=cur_y complete dim=1
#pragma HLS ARRAY_PARTITION variable=rec_y complete dim=1
  const int kDepthFifoX = 1;
  const int kDepthFifoH = 1;
  const int kDepthFifoY = 2;
#pragma HLS STREAM variable=cur_x depth=kDepthFifoX
#pragma HLS STREAM variable=rec_h depth=kDepthFifoH
#pragma HLS STREAM variable=cur_y depth=kDepthFifoY
#pragma HLS STREAM variable=rec_y depth=kDepthFifoY

  X_Dispatcher: for (int i = 0; i < INPUT_SIZE / kDivider; ++i) {
#pragma HLS PIPELINE II=1
    for (int j = 0; j < kNumGates; ++j) {
      cur_x[j][i] = x[i];
    }
  }

  H_Dispatcher: for (int i = 0; i < HIDDEN_SIZE / kDivider; ++i) {
#pragma HLS PIPELINE II=1
    for (int j = 0; j < kNumGates; ++j) {
      rec_h[j][i] = h[i];
    }
  }

  // svd::gemv_systolic<svd::ActivationD, svd::ActivationD, HIDDEN_SIZE, INPUT_SIZE, kDivider, kDivider>(cur_i_T, cur_x[0], cur_y[0]);
  // svd::gemv_systolic<svd::ActivationD, svd::ActivationD, HIDDEN_SIZE, INPUT_SIZE, kDivider, kDivider>(cur_f_T, cur_x[1], cur_y[1]);
  // svd::gemv_systolic<svd::ActivationD, svd::ActivationD, HIDDEN_SIZE, INPUT_SIZE, kDivider, kDivider>(cur_c_T, cur_x[2], cur_y[2]);
  // svd::gemv_systolic<svd::ActivationD, svd::ActivationD, HIDDEN_SIZE, INPUT_SIZE, kDivider, kDivider>(cur_o_T, cur_x[3], cur_y[3]);

  // svd::gemv_systolic<svd::ActivationD, svd::ActivationD, HIDDEN_SIZE, HIDDEN_SIZE, kDivider, kDivider>(rec_i_T, rec_h[0], rec_y[0]);
  // svd::gemv_systolic<svd::ActivationD, svd::ActivationD, HIDDEN_SIZE, HIDDEN_SIZE, kDivider, kDivider>(rec_f_T, rec_h[1], rec_y[1]);
  // svd::gemv_systolic<svd::ActivationD, svd::ActivationD, HIDDEN_SIZE, HIDDEN_SIZE, kDivider, kDivider>(rec_c_T, rec_h[2], rec_y[2]);
  // svd::gemv_systolic<svd::ActivationD, svd::ActivationD, HIDDEN_SIZE, HIDDEN_SIZE, kDivider, kDivider>(rec_o_T, rec_h[3], rec_y[3]);

  svd_fpga_non_lin(c_rec, cur_y[0], cur_y[1], cur_y[2], cur_y[3],
    rec_y[0], rec_y[1], rec_y[2], rec_y[3], i_bias, f_bias, c_bias, o_bias,
    c_cur, out);
}

void dummy_kernel(const int size, const svd::ActivationD *x, svd::ActivationD *y) {
  for (int i = 0; i < size; ++i) {
#pragma HLS PIPELINE II=1
    y[i] += x[i] + 12;
  }
}

#ifdef SDS_DESIGN
#pragma SDS data zero_copy(a[0:16][0:16])
#pragma SDS data zero_copy(b[0:16][0:16])
#pragma SDS data zero_copy(c[0:16][0:16])
// #pragma SDS data mem_attribute(a:PHYSICAL_CONTIGUOUS)
// #pragma SDS data mem_attribute(b:PHYSICAL_CONTIGUOUS)
// #pragma SDS data mem_attribute(c:PHYSICAL_CONTIGUOUS)
#endif
void dummy_gemm_v0(const svd::ActivationD a[16][16], const svd::ActivationD b[16][16],
    svd::ActivationD c[16][16]) {
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl
#pragma HLS ARRAY_PARTITION variable=a complete dim=1
#pragma HLS ARRAY_PARTITION variable=b complete dim=1
#pragma HLS ARRAY_PARTITION variable=c complete dim=1

#pragma HLS INTERFACE axis port=a
#pragma HLS INTERFACE axis port=b
#pragma HLS INTERFACE axis port=c

// #pragma HLS INTERFACE m_axi port=a offset=slave depth=16
// #pragma HLS INTERFACE m_axi port=b offset=slave depth=16
// #pragma HLS INTERFACE m_axi port=c offset=slave depth=16

  svd::ActivationD d[16][16];
#pragma HLS ARRAY_PARTITION variable=d complete dim=1

  for (int i = 0; i < 16; ++i) {
#pragma HLS UNROLL
    dummy_kernel(16, a[i], d[i]);
  }
  for (int i = 0; i < 16; ++i) {
#pragma HLS UNROLL
    dummy_kernel(16, b[i], d[i]);
  }

  for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE II=1
    for (int j = 0; j < 16; ++j) {
      c[i][j] = d[i][j];
    }
  }

//   svd::ActivationD c_val = 0;
//   for (int i = 0; i < 16; ++i) {
//     for (int j = 0; j < 16; ++j) {
// #pragma HLS PIPELINE II=1
//       for (int k = 0; k < 16; ++k) {
//         if (k == 0) {
//           c_val = 0;
//         }
//         c_val += a[i][k] * b[k][j];
//         if (k == 15) {
//           c[i][j] = c_val;
//         }
//       }
//     }
//   }
}

void dummy_gemm(svd::DmaInterfaceD a[2], svd::DmaInterfaceD b[2], svd::DmaInterfaceD c[2]) {
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl
#pragma HLS ARRAY_PARTITION variable=a complete dim=1
#pragma HLS ARRAY_PARTITION variable=b complete dim=1
#pragma HLS ARRAY_PARTITION variable=c complete dim=1

#pragma HLS INTERFACE axis port=a
#pragma HLS INTERFACE axis port=b
#pragma HLS INTERFACE axis port=c

  for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE II=1
    for (int j = 0; j < 2; ++j) {
      svd::ActivationD a_val = 0;
      svd::ActivationD b_val = 0;
      svd::ActivationD c_val = 0;
#if USE_FIX
      a_val.range() = a[j].read().data.range();
      b_val.range() = b[j].read().data.range();
#else
      auto a_tmp = a[j].read().data.range();
      auto b_tmp = b[j].read().data.range();
      a_val = *((svd::ActivationD*)&a_tmp);
      b_val = *((svd::ActivationD*)&b_tmp);
#endif
      c_val = a_val + b_val;
      svd::AxisPacketD c_packet;
#if USE_FIX
      c_packet.data.range() = c_val.range();
#else
      ap_uint<FIX_WIDTH> c_tmp = *((ap_uint<FIX_WIDTH>*)&c_val);
      c_packet.data.range() = c_tmp.range();
#endif
      if (i == 15) {
        c_packet.last = 1;
      } else {
        c_packet.last = 0;
      }
      c[j].write(c_packet);
    }
  }
}

#if 1
void dummy_dispatcher(hls::stream<ap_uint<1 * 8> > &x, hls::stream<ap_uint<1 * 8> > y[4]) {
  const int kNumInputElem = 32;
  const int kNumPE = 4;
  const int kVerbose = 0;
  svd::PipelinedDispatcher<8, kNumPE>(kNumInputElem, x, y, kVerbose);
}

template <int BitWidth>
void set_random_uint(ap_uint<BitWidth> &x) {
  for (int i = 0; i < BitWidth; ++i) {
    const int random_bit = (rand() % 2 == 0) ? 1 : 0;
    x[i] = random_bit;
  }
}

void test_dispatcher() {
  const int kNumInputElem = 32;
  const int kNumPE = 4;
  const int kTileSize = 32;

  ap_uint<1 * 8> *x = new ap_uint<1 * 8>[kNumInputElem];
  hls::stream<ap_uint<1 * 8> > x_stream("x_stream");
  for (int i = 0; i < kNumInputElem; ++i) {
    set_random_uint(x[i]);
    x_stream.write(x[i]);
  }

#ifndef __VITIS_HLS__
  std::cout << "[Dispatcher] Running IP. x_stream.size() = " << x_stream.size() << "\n";
#endif
  hls::stream<ap_uint<1 * 8> > y_stream[kNumPE];
  dummy_dispatcher(x_stream, y_stream);

  ap_uint<1 * 8> *y = new ap_uint<1 * 8>[kNumInputElem];
  for (int i = 0; i < kNumInputElem / kNumPE; ++i) {
    for (int j = 0; j < kNumPE; ++j) {
      y[i * kNumPE + j] = y_stream[j].read();
    }
  }

  std::cout << "[Dispatcher] Checking correctness.\n";
  int num_errors = 0;
  for (int i = 0; i < kNumInputElem; ++i) {
    if (x[i] != y[i]) {
      num_errors++;
    }
  }
  std::cout << "[Dispatcher] There were " << num_errors << "/" << kNumInputElem << " errors (" << float(num_errors) / float(kNumInputElem) * 100.0 << "%)\n";
}
#endif