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
 * @file        lstm_hardware.h
 * 
 * @author      Stefano Ribes
 *
 * Library of templated HLS functions for BNN deployment. 
 * This file lists a set of functions to access memory mapped values into 
 * streams 
 *
 *****************************************************************************/
#ifndef LSTM_HLS_LSTM_HARDWARE_H_
#define LSTM_HLS_LSTM_HARDWARE_H_

#include "math_utils/activation_functions.h"
#include "svd_params.h"

#ifdef AP_INT_MAX_W
#undef AP_INT_MAX_W
#define AP_INT_MAX_W 4096
#endif
#include "ap_int.h"
#include "hls_linear_algebra.h"
#include "ap_axi_sdata.h"

#ifndef __SYNTHESIS__
#include <chrono>
#endif

#ifdef SDS_DESIGN
#include <stdlib.h>
#include "sds_lib.h"
#endif

#ifndef TIMESTEPS_SIZE
#define TIMESTEPS_SIZE NUM_TIMESTEPS
#endif

#ifndef TIMESTEPS_TILE_SIZE
#define TIMESTEPS_TILE_SIZE NUM_TIMESTEPS // M // (svd::AxiD::width / FIX_WIDTH)
#endif
#ifndef HIDDEN_TILE_SIZE
#define HIDDEN_TILE_SIZE 8 // N // (svd::AxiD::width / FIX_WIDTH)
#endif
#ifndef INPUT_TILE_SIZE
#define INPUT_TILE_SIZE 8 // K // (svd::AxiD::width / FIX_WIDTH)
#endif

namespace svd {

// struct MY_CONFIG: hls::matrix_multiply_traits<hls::NoTranspose, hls::NoTranspose,
//     A_ROWS, A_COLS, B_ROWS, B_COLS, MATRIX_T,  MATRIX_T> {
//   static const int ARCH = 4;
//   static const int INNER_II = 1;
//   static const int UNROLL_FACTOR = 2; // ARCH4 will completely unroll the inner loop anyway.
// };

struct MatrixConfigFixCurrent: hls::matrix_multiply_traits <
    hls::NoTranspose,
    hls::NoTranspose,
    TIMESTEPS_TILE_SIZE,
    INPUT_TILE_SIZE,
    INPUT_TILE_SIZE,
    HIDDEN_TILE_SIZE,
    ActivationD,
    ActivationD > {
    // static const int RowsATrans        = HIDDEN_TILE_SIZE; // ( TransposeFormA::TransposeType != 0 ? ColsA : RowsA);
    // static const int ColsATrans        = TIMESTEPS_TILE_SIZE; // ( TransposeFormA::TransposeType != 0 ? RowsA : ColsA);
    // static const int RowsBTrans        = ( TransposeFormB::TransposeType != 0 ? ColsB : RowsB);
    // static const int ColsBTrans        = ( TransposeFormB::TransposeType != 0 ? RowsB : ColsB);
    // static const int B_UNROLL_DIM      = ( TransposeFormB::TransposeType != 0 ? 1 : 2);
    // static const int A_FULL_UNROLL_DIM = ( TransposeFormA::TransposeType != 0 ? 1 : 2);
    // static const int B_FULL_UNROLL_DIM = ( TransposeFormB::TransposeType != 0 ? 2 : 1);
    // typedef ap_fixed<W1,                                I1,                                Q1,     O1     ,N1>  INPUT_T;
    // typedef ap_fixed<W1+W1,                             I1+I1,                             AP_TRN, AP_WRAP, 0>  MULT_T;
    // typedef ap_fixed<W1+W1+BitWidth<ColsATrans>::Value, I1+I1+BitWidth<ColsATrans>::Value, AP_TRN, AP_WRAP, 0>  ACCUM_T;
    typedef ActivationD INPUT_T;
    typedef MultD MULT_T;
    typedef AccumD ACCUM_T;
    static const int ARCH = 4;
    static const int INNER_II = 1;
    // static const int UNROLL_FACTOR = 1;
    static const int M = TIMESTEPS_TILE_SIZE;
    static const int N = HIDDEN_TILE_SIZE;
    static const int K = INPUT_TILE_SIZE;
};

struct MatrixConfigFixRecurrent: hls::matrix_multiply_traits <
    hls::NoTranspose,
    hls::NoTranspose,
    TIMESTEPS_TILE_SIZE,
    HIDDEN_TILE_SIZE,
    HIDDEN_TILE_SIZE,
    1,
    ActivationD,
    ActivationD> {
    // static const int RowsATrans        = ( TransposeFormA::TransposeType != 0 ? ColsA : RowsA);
    // static const int ColsATrans        = ( TransposeFormA::TransposeType != 0 ? RowsA : ColsA);
    // static const int RowsBTrans        = ( TransposeFormB::TransposeType != 0 ? ColsB : RowsB);
    // static const int ColsBTrans        = ( TransposeFormB::TransposeType != 0 ? RowsB : ColsB);
    // static const int B_UNROLL_DIM      = ( TransposeFormB::TransposeType != 0 ? 1 : 2);
    // static const int A_FULL_UNROLL_DIM = ( TransposeFormA::TransposeType != 0 ? 1 : 2);
    // static const int B_FULL_UNROLL_DIM = ( TransposeFormB::TransposeType != 0 ? 2 : 1);
    // typedef ap_fixed<W1,                                I1,                                Q1,     O1     ,N1>  INPUT_T;
    // typedef ap_fixed<W1+W1,                             I1+I1,                             AP_TRN, AP_WRAP, 0>  MULT_T;
    // typedef ap_fixed<W1+W1+BitWidth<ColsATrans>::Value, I1+I1+BitWidth<ColsATrans>::Value, AP_TRN, AP_WRAP, 0>  ACCUM_T;
    typedef ActivationD INPUT_T;
    typedef MultD MULT_T;
    typedef AccumD ACCUM_T;
    static const int ARCH = 4;
    static const int INNER_II = 1;
    // static const int UNROLL_FACTOR = 1;
    static const int M = TIMESTEPS_TILE_SIZE;
    static const int N = 1;
    static const int K = HIDDEN_TILE_SIZE;
};

void svd_fpga_cur_gemm_axi(const AxiD *a, const AxiD *b, AxiD *c);

template <int M, int N, int K, int Tm, int Tn, int Tk>
void cur_gemm(const ActivationD *a, const ActivationD *b, ActivationD *c);

typedef struct {
  ap_uint<FIX_WIDTH> data;
  ap_uint<1> last;
} AxisPacketD;
typedef hls::stream<AxisPacketD> DmaInterfaceD;

} // end namespace svd

void svd_fpga_cur_gemm_gate(const svd::ActivationD *a, const svd::ActivationD *b, svd::ActivationD *c);
void svd_fpga_cur_gemm_summa_gate(const svd::ActivationD *a, const svd::ActivationD *b, svd::ActivationD *c);
void svd_fpga_cur_gemm(const svd::ActivationD *a, const svd::ActivationD *b, svd::ActivationD *c);

void svd_fpga_rec_gemv_gate(const svd::ActivationD *a, const svd::ActivationD *b, svd::ActivationD *c);
void svd_fpga_rec_gemv(const svd::ActivationD *a, const svd::ActivationD *b, svd::ActivationD *c);

void svd_fpga_cur_gemv_gate_systolic(const ap_uint<FIX_WIDTH * INPUT_TILE_SIZE> *a, const ap_uint<FIX_WIDTH * INPUT_TILE_SIZE> *b, ap_uint<FIX_WIDTH * INPUT_TILE_SIZE> *c);
void svd_fpga_rec_gemv_gate_systolic(const ap_uint<FIX_WIDTH * HIDDEN_TILE_SIZE> *a, const ap_uint<FIX_WIDTH * HIDDEN_TILE_SIZE> *b, ap_uint<FIX_WIDTH * HIDDEN_TILE_SIZE> *c);

void svd_fpga_non_lin(const svd::AxiD *c_rec, const svd::AxiD *cur_gate_i,
    const svd::AxiD *cur_gate_f, const svd::AxiD *cur_gate_c,
    const svd::AxiD *cur_gate_o, const svd::AxiD *rec_gate_i,
    const svd::AxiD *rec_gate_f, const svd::AxiD *rec_gate_c,
    const svd::AxiD *rec_gate_o, const svd::AxiD *bias_i,
    const svd::AxiD *bias_f, const svd::AxiD *bias_c, const svd::AxiD *bias_o,
    svd::AxiD *c_cur, svd::AxiD *h_port);

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
                   svd::ActivationD *out);

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
                      svd::ActivationD *out);

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
                      svd::AxiD *out);

void dummy_gemm(svd::DmaInterfaceD a[2], svd::DmaInterfaceD b[2], svd::DmaInterfaceD c[2]);

void dummy_gemm_v0(const svd::ActivationD a[16][16], const svd::ActivationD b[16][16],
    svd::ActivationD c[16][16]);

void test_dispatcher();

#endif // end LSTM_HLS_LSTM_HARDWARE_H_