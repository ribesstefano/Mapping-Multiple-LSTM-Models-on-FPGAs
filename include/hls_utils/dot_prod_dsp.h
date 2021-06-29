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
 * @file dsp_functions.h
 * 
 * @author     Stefano Ribes
 *
 * Library of templated HLS functions for BNN deployment. 
 * This file lists a set of functions to access memory mapped values into 
 * streams 
 *
 *****************************************************************************/
#ifndef HLS_UTILS_DOT_PROD_DSP_H_
#define HLS_UTILS_DOT_PROD_DSP_H_

#include "ap_int.h"
#include "assert.h"

/**
 * @brief      Implements p0 += y_dsp * w_dsp + y_lut * w_lut; p1 += x_dsp *
 *             w_dsp + x_lut * w_lut;
 *
 * @param[in]  x_dsp  The x to be mapped to a DSP
 * @param[in]  y_dsp  The y to be mapped to a DSP
 * @param[in]  w_dsp  The w to be mapped to a DSP
 * @param[in]  x_lut  The x to be mapped to a LUT
 * @param[in]  y_lut  The y to be mapped to a LUT
 * @param[in]  w_lut  The w to be mapped to a LUT
 * @param      p0     The output p0
 * @param      p1     The output p1
 *
 * @tparam     DspD   Must be a 8bit ap_(u)int or ap_(u)fixed.
 */
template <typename DspD>
void dot_prod_dsp_lut(const DspD x_dsp, const DspD y_dsp, const DspD w_dsp,
                      const DspD x_lut, const DspD y_lut, const DspD w_lut,
                      DspD &p0, DspD &p1) {
#pragma HLS PIPELINE II=3
// NOTE: inlining prevents a clear parent's structure and a simple ctrl logic.
#pragma HLS INLINE off
  assert(DspD::width == 8); // Only allow 8bit ap_uint or ap_fixed
  // ===========================================================================
  // LUT Multiplication
  // ===========================================================================
  // NOTE: The method range() MUST be used, otherwise there's a cast operation
  // happening instead of a bit-by-bit copy.
  ap_uint<17> xw_lut = (x_lut * w_lut).range();
  ap_uint<17> yw_lut = (y_lut * w_lut).range();
#pragma HLS RESOURCE variable=xw_lut core=Mul_LUT latency=1
#pragma HLS RESOURCE variable=yw_lut core=Mul_LUT latency=1
  ap_uint<48> p_lut = 0;
  p_lut(16, 0) = yw_lut;
  p_lut(32, 17) = xw_lut;
  // ===========================================================================
  // DSP
  // ===========================================================================
  ap_int<25> x_dsp25 = 0;
  ap_int<25> y_dsp25 = 0;
  ap_int<18> w_dsp18 = 0;
  x_dsp25(24, 17) = x_dsp.range();
  y_dsp25(7, 0) = y_dsp.range();
  w_dsp18(7, 0) = w_dsp.range();
  // ===========================================================================
  // Sign extension
  // ===========================================================================
  const ap_uint<17> kNegativeSignY = 0b11111111111111111;
  const ap_uint<17> kPositiveSignY = 0b00000000000000000;
  const ap_uint<10> kNegativeSignW = 0b1111111111;
  const ap_uint<10> kPositiveSignW = 0b0000000000;
  y_dsp25(24, 8) = y_dsp[7] == 1 ? kNegativeSignY : kPositiveSignY;
  w_dsp18(17, 8) = w_dsp[7] == 1 ? kNegativeSignW : kPositiveSignW;
//  ap_int<48> p_dsp = (x_dsp25 + y_dsp25) * w_dsp18;
//#pragma HLS RESOURCE variable=p_dsp core=DSP48
  // ===========================================================================
  // Adjust LSB LUT
  // ===========================================================================
  p_lut[16] = p_lut[15] = (y_dsp[7] ^ w_dsp[7]);
  // ===========================================================================
  // Final Sum DSP + LUT
  // ===========================================================================
  // auto p = p_dsp + p_lut;
  auto p = (x_dsp25 + y_dsp25) * w_dsp18 + p_lut;
#pragma HLS RESOURCE variable=p core=DSP48
  p[16] = yw_lut[15] ^ (y_dsp25[7] ^ w_dsp18[7]);
  // ===========================================================================
  // Accumulation
  // ===========================================================================
  const int kIntWidth = DspD::width; // svd::SVDParameters.kFixFracWidth;
  ap_fixed<17, kIntWidth*2+1> p0_reg = 0;
  ap_fixed<17, kIntWidth*2+1> p1_reg = 0;
  p0_reg.range() = p(16, 0);
  p1_reg.range() = p(32, 17);
  p0 += p0_reg;
  p1 += p1_reg;
}

/**
 * @brief      Implements p0 += y_dsp * w_dsp + y_lut * w_lut; p1 += x_dsp *
 *             w_dsp + x_lut * w_lut;
 *
 * @param[in]  x_dsp  The x to be mapped to a DSP
 * @param[in]  y_dsp  The y to be mapped to a DSP
 * @param[in]  w_dsp  The w to be mapped to a DSP
 * @param[in]  x_lut  The x to be mapped to a LUT
 * @param[in]  y_lut  The y to be mapped to a LUT
 * @param[in]  w_lut  The w to be mapped to a LUT
 * @param      p0     The output p0
 * @param      p1     The output p1
 *
 * @tparam     T      Must be a 8bit ap_(u)int or ap_(u)fixed.
 */
template <typename T>
void dot_prod_dsp_lut_generic(const T x_dsp, const T y_dsp, const T w_dsp,
                              const T x_lut, const T y_lut, const T w_lut,
                              T &p0, T &p1) {
#pragma HLS PIPELINE II=3
// NOTE: inlining prevents a clear parent's structure and a simple ctrl logic.
#pragma HLS INLINE off
  auto p0_tmp = y_dsp * w_dsp + y_lut * w_lut;
  auto p1_tmp = x_dsp * w_dsp + x_lut * w_lut;
#pragma HLS RESOURCE variable=p1_tmp core=DSP48
  p0 += p0_tmp;
  p1 += p1_tmp;
}

#endif // end HLS_UTILS_DOT_PROD_DSP_H_