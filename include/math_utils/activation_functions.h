#ifndef MATH_ACTIVATION_FUNCTIONS_H_
#define MATH_ACTIVATION_FUNCTIONS_H_

#include "svd_params.h"

#include "hls_stream.h"
#include "assert.h"
#ifdef __VITIS_HLS__
#include "hls_vector.h"
#endif

#include <cmath>

template <typename Dtype>
void svd_sigmoid(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
void svd_hard_sigmoid(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
void svd_tanh(const int n, const Dtype* a, Dtype* y);

/**
 * @brief      Hard sigmoid function:
 *
 *             f(x) = 0             , if x < -2.5
 *             f(x) = 1             , if x > 2.5
 *             f(x) = x * 0.2 + 0.5 , elsewhere
 *
 * @param[in]  x     The input value
 *
 * @tparam     T     The inut data type
 *
 * @return     The result of hsigmoid(x)
 */
template <typename T>
T HardSigmoid(const T x) {
#pragma HLS INLINE
  const T kThreshold = 2.5;
  const T kZero = 0;
  const T kOne = 1;
  const T kSlope = 0.2;
  const T kShift = 0.5;
  if (x < -kThreshold) {
    return kZero;
  } else if (x > kThreshold) {
    return kOne;
  } else {
    // NOTE: having a DSP here is an overkill and the tool won't use it.
    return (kSlope * x + kShift);
  }
}

template <typename Dtype, int TableSize>
void InitTanhTable(Dtype tanh_table[TableSize]) {
  for (int i = 0; i < TableSize; i++) {
    // First, convert from table index to X-value (signed 8-bit, range -4 to +4)
    float in_val = 2 * 4.0 * (i - float(TableSize) / 2.0) / float(TableSize);
    // Next, compute lookup table function
    Dtype real_val = tanh(in_val);
    tanh_table[i] = real_val;
  }
}

template <typename Dtype, typename Dtable, int TableSize>
Dtype TanhLookup(const Dtype data_in, const Dtable table[TableSize]) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
  // int data_round = data_in * table_size / 8;
  // int index = data_round + 4 * table_size / 8;
  const int data_round = data_in * (TableSize >> 3);
  int index = data_round + ((TableSize << 2) >> 3);
  if (index < 0) {
    index = 0;
  }
  if (index > TableSize - 1) {
    index = TableSize - 1;
  }
  return (Dtype) table[index];
}

/**
 * @brief      Hyperbolic tangent function implemented as a table lookup.
 *
 * @param[in]  x           The function input
 * @param[in]  tanh_table  The hyperbolic tangent lookup table
 *
 * @tparam     DataD       The input data type
 * @tparam     TableSize   The table size
 *
 * @return     The result of tanh(x)
 */
template <typename DataD, int TableSize>
DataD TanH(const DataD x, const DataD tanh_table[TableSize]) {
#pragma HLS INLINE
#if USE_FIX
  return TanhLookup<DataD, DataD, TableSize>(x, tanh_table);
#else
  return tanh(float(x));
#endif
}

template <typename DataD, int TableSize>
DataD TanH(const DataD x) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
  DataD tanh_table[TableSize];
// #pragma HLS RESOURCE variable=tanh_table core=ROM
  InitTanhTable<DataD, TableSize>(tanh_table);
#if USE_FIX
  return TanhLookup<DataD, DataD, TableSize>(x, tanh_table);
#else
  return tanh(float(x));
#endif
}


/**
 * @brief      Non-linearity unit for software emulation.
 *
 * @param[in]  VectLength       The vect length
 * @param[in]  NumTiles         The number tiles
 * @param[in]  NumGates         The number gates
 * @param[in]  c_t_prev         The previous cell state c
 * @param      cur_gate_stream  The current gate stream
 * @param      rec_gate_stream  The recurrent gate stream
 * @param      h                The current output h
 * @param      c_t              The current cell state c
 *
 * @tparam     DataT            The weight data type
 * @tparam     DataA            The activation data type
 * @tparam     DataAcc          The accumulation data type
 * @tparam     TableSize        The tanh lookup table size
 */
template <typename DataA, typename DataW, typename DataAcc, int TableSize>
void NonLinearityUnitSoftware(const int VectLength,
                              const int NumTiles,
                              const int NumGates,
                              const DataA *c_t_prev,
                              hls::stream<DataA> **cur_gate_stream,
                              hls::stream<DataA> **rec_gate_stream,
                              DataA *h,
                              DataA *c_t,
                              const bool has_bias = false,
                              const DataW *bias = nullptr) {
  assert(VectLength % NumTiles == 0);
  assert(NumGates >= 4);
  DataW tanh_table[TableSize];
  InitTanhTable<DataW, TableSize>(tanh_table);
  const int kNumElemsTile = VectLength / NumTiles;
  for(int i = 0; i < NumTiles; ++i) {
    for (int j = 0; j < kNumElemsTile; ++j) {
      // =======================================================================
      // Python (Keras) Implementation:
      // i = self.hard_sigm(x_i + K.dot(h_tm1_i, self.recurrent_kernel_i))
      // f = self.hard_sigm(x_f + K.dot(h_tm1_f, self.recurrent_kernel_f))
      // c = f * c_tm1 + i * self.tanh(x_c + K.dot(h_tm1_c, self.recurrent_kernel_c))
      // o = self.hard_sigm(x_o + K.dot(h_tm1_o, self.recurrent_kernel_o))
      // h = o * self.tanh(c)
      // =======================================================================
      DataA i_gate = 0;
      DataA f_gate = 0;
      DataA c_gate = 0;
      DataA o_gate = 0;
      if (has_bias) {
        i_gate = cur_gate_stream[0][i].read() + rec_gate_stream[0][i].read() + bias[0 * VectLength + i * kNumElemsTile + j];
        f_gate = cur_gate_stream[1][i].read() + rec_gate_stream[1][i].read() + bias[1 * VectLength + i * kNumElemsTile + j];
        c_gate = cur_gate_stream[2][i].read() + rec_gate_stream[2][i].read() + bias[2 * VectLength + i * kNumElemsTile + j];
        o_gate = cur_gate_stream[3][i].read() + rec_gate_stream[3][i].read() + bias[3 * VectLength + i * kNumElemsTile + j];
      } else {
        i_gate = cur_gate_stream[0][i].read() + rec_gate_stream[0][i].read();
        f_gate = cur_gate_stream[1][i].read() + rec_gate_stream[1][i].read();
        o_gate = cur_gate_stream[2][i].read() + rec_gate_stream[2][i].read();
        c_gate = cur_gate_stream[3][i].read() + rec_gate_stream[3][i].read();
      }
      const auto sigma_i = HardSigmoid<DataA>(i_gate);
      const auto sigma_f = HardSigmoid<DataA>(f_gate);
      const auto sigma_o = HardSigmoid<DataA>(o_gate);
      const auto tanh_cell = TanH<DataW, TableSize>(c_gate, tanh_table);
      const auto c_lhs = sigma_f * c_t_prev[i * kNumElemsTile + j];
      const auto c_t_tile = c_lhs + sigma_i * tanh_cell;
      c_t[i * kNumElemsTile + j] = c_t_tile;
      const auto c_tanh = TanH<DataW, TableSize>(c_t_tile, tanh_table);
      const auto h_t_tile = sigma_o * c_tanh;
      h[i * kNumElemsTile + j] = h_t_tile;
    }
  }
}

namespace svd {

/**
 * @brief      LSTM non-linearity function to be applied to each output element.
 *             It implements the following Python (Keras) implementation:
 *
 *                i = self.recurrent_activation(x_i + K.dot(h_tm1_i,
 *                                                self.recurrent_kernel_i))
 *                f = self.recurrent_activation(x_f + K.dot(h_tm1_f,
 *                                                self.recurrent_kernel_f))
 *                c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1_c,
 *                                                self.recurrent_kernel_c))
 *                o = self.recurrent_activation(x_o + K.dot(h_tm1_o,
 *                                                self.recurrent_kernel_o))
 *                h = o * self.activation(c)
 *
 * @param[in]  has_bias        Indicates if bias is present.
 * @param[in]  cur_gate_i      The current gate i element
 * @param[in]  cur_gate_f      The current gate f element
 * @param[in]  cur_gate_c      The current gate c element
 * @param[in]  cur_gate_o      The current gate o element
 * @param[in]  rec_gate_i      The recurrent gate i element
 * @param[in]  rec_gate_f      The recurrent gate f element
 * @param[in]  rec_gate_c      The recurrent gate c element
 * @param[in]  rec_gate_o      The recurrent gate o element
 * @param[in]  bias_i          The bias i element
 * @param[in]  bias_f          The bias f element
 * @param[in]  bias_c          The bias c element
 * @param[in]  bias_o          The bias o element
 * @param[in]  c_prev          The previous c (cell) state
 * @param      c_curr          The current c (cell) state
 * @param      h_curr          The h current
 *
 * @tparam     ActivationType  The activation type
 * @tparam     WeightType      The weight type
 * @tparam     LutSize         The tanh LUT size: having it templated helps
 *                             inferring a ROM
 */
template <typename ActivationType, typename WeightType, int LutSize>
void LstmNonLinearFunctions(const bool has_bias,
                            const WeightType cur_gate_i,
                            const WeightType cur_gate_f,
                            const WeightType cur_gate_c,
                            const WeightType cur_gate_o,
                            const WeightType rec_gate_i,
                            const WeightType rec_gate_f,
                            const WeightType rec_gate_c,
                            const WeightType rec_gate_o,
                            const WeightType bias_i,
                            const WeightType bias_f,
                            const WeightType bias_c,
                            const WeightType bias_o,
                            const ActivationType c_prev,
                            ActivationType &c_curr,
                            ActivationType &h_curr) {
#pragma HLS FUNCTION_INSTANTIATE variable=has_bias
#pragma HLS PIPELINE II=1
  ActivationType i_gate = 0;
  ActivationType f_gate = 0;
  ActivationType c_gate = 0;
  ActivationType o_gate = 0;
  if (has_bias) {
    i_gate = cur_gate_i + rec_gate_i + bias_i;
    f_gate = cur_gate_f + rec_gate_f + bias_f;
    c_gate = cur_gate_c + rec_gate_c + bias_c;
    o_gate = cur_gate_o + rec_gate_o + bias_o;
#pragma HLS RESOURCE variable=i_gate core=AddSub_DSP // latency=3
#pragma HLS RESOURCE variable=f_gate core=AddSub_DSP // latency=3
#pragma HLS RESOURCE variable=c_gate core=AddSub_DSP // latency=3
#pragma HLS RESOURCE variable=o_gate core=AddSub_DSP // latency=3
  } else {
    i_gate = cur_gate_i + rec_gate_i;
    f_gate = cur_gate_f + rec_gate_f;
    o_gate = cur_gate_c + rec_gate_c;
    c_gate = cur_gate_o + rec_gate_o;
#pragma HLS RESOURCE variable=i_gate core=AddSub_DSP // latency=3
#pragma HLS RESOURCE variable=f_gate core=AddSub_DSP // latency=3
#pragma HLS RESOURCE variable=c_gate core=AddSub_DSP // latency=3
#pragma HLS RESOURCE variable=o_gate core=AddSub_DSP // latency=3
  }

  const auto sigma_i = HardSigmoid<ActivationType>(i_gate);
  const auto sigma_f = HardSigmoid<ActivationType>(f_gate);
  const auto sigma_o = HardSigmoid<ActivationType>(o_gate);
  const auto tanh_cell = TanH<ActivationType, LutSize>(c_gate);

  const auto c_lhs = sigma_f * c_prev;
  const auto c_reg = c_lhs + sigma_i * tanh_cell;
#pragma HLS RESOURCE variable=c_lhs core=DSP48 latency=3
#pragma HLS RESOURCE variable=c_reg core=DSP48 latency=3
  c_curr = c_reg;

  const auto c_tanh = TanH<ActivationType, LutSize>(c_reg);
  const auto h_reg = sigma_o * c_tanh;
#pragma HLS RESOURCE variable=h_reg core=DSP48 latency=3
  h_curr = h_reg;
}


#ifdef __VITIS_HLS__
/**
 * @brief      LSTM non-linearity function to be applied to each output element.
 *             It implements the following Python (Keras) implementation:
 *
 *                i = self.recurrent_activation(x_i + K.dot(h_tm1_i,
 *                                                self.recurrent_kernel_i))
 *                f = self.recurrent_activation(x_f + K.dot(h_tm1_f,
 *                                                self.recurrent_kernel_f))
 *                c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1_c,
 *                                                self.recurrent_kernel_c))
 *                o = self.recurrent_activation(x_o + K.dot(h_tm1_o,
 *                                                self.recurrent_kernel_o))
 *                h = o * self.activation(c)
 *
 * @param[in]  has_bias        Indicates if bias is present.
 * @param[in]  cur_gate_i      The current gate i element
 * @param[in]  cur_gate_f      The current gate f element
 * @param[in]  cur_gate_c      The current gate c element
 * @param[in]  cur_gate_o      The current gate o element
 * @param[in]  rec_gate_i      The recurrent gate i element
 * @param[in]  rec_gate_f      The recurrent gate f element
 * @param[in]  rec_gate_c      The recurrent gate c element
 * @param[in]  rec_gate_o      The recurrent gate o element
 * @param[in]  bias_i          The bias i element
 * @param[in]  bias_f          The bias f element
 * @param[in]  bias_c          The bias c element
 * @param[in]  bias_o          The bias o element
 * @param[in]  c_prev          The previous c (cell) state
 * @param      c_curr          The current c (cell) state
 * @param      h_curr          The h current
 *
 * @tparam     ActivationType  The activation type
 * @tparam     WeightType      The weight type
 * @tparam     LutSize         The tanh LUT size: having it templated helps
 *                             inferring a ROM
 */
template <typename T, int N, int LutSize>
void LstmVectNonLinearFunctions(const bool has_bias,
    const hls::vector<T, N> cur_gate_i,
    const hls::vector<T, N> cur_gate_f,
    const hls::vector<T, N> cur_gate_c,
    const hls::vector<T, N> cur_gate_o,
    const hls::vector<T, N> rec_gate_i,
    const hls::vector<T, N> rec_gate_f,
    const hls::vector<T, N> rec_gate_c,
    const hls::vector<T, N> rec_gate_o,
    const hls::vector<T, N> bias_i,
    const hls::vector<T, N> bias_f,
    const hls::vector<T, N> bias_c,
    const hls::vector<T, N> bias_o,
    const hls::vector<T, N> c_prev,
    const hls::vector<T, N> &c_curr,
    const hls::vector<T, N> &h_curr) {
#pragma HLS FUNCTION_INSTANTIATE variable=has_bias
#pragma HLS PIPELINE II=1
  const hls::vector<T, N> i_gate;
  const hls::vector<T, N> f_gate;
  const hls::vector<T, N> c_gate;
  const hls::vector<T, N> o_gate;
  if (has_bias) {
    i_gate = cur_gate_i + rec_gate_i + bias_i;
    f_gate = cur_gate_f + rec_gate_f + bias_f;
    c_gate = cur_gate_c + rec_gate_c + bias_c;
    o_gate = cur_gate_o + rec_gate_o + bias_o;
  } else {
    i_gate = cur_gate_i + rec_gate_i;
    f_gate = cur_gate_f + rec_gate_f;
    o_gate = cur_gate_c + rec_gate_c;
    c_gate = cur_gate_o + rec_gate_o;
  }
#pragma HLS BIND_OP variable=i_gate op=add impl=dsp
#pragma HLS BIND_OP variable=f_gate op=add impl=dsp
#pragma HLS BIND_OP variable=c_gate op=add impl=dsp
#pragma HLS BIND_OP variable=o_gate op=add impl=dsp
  hls::vector<T, N> sigma_i;
  hls::vector<T, N> sigma_f;
  hls::vector<T, N> sigma_o;
  hls::vector<T, N> tanh_cell;
  hls::vector<T, N> c_tanh;
  for (int i = 0; i < N; ++i) {
    sigma_i[i] = HardSigmoid<T>(i_gate[i]);
    sigma_f[i] = HardSigmoid<T>(f_gate[i]);
    sigma_o[i] = HardSigmoid<T>(o_gate[i]);
    tanh_cell[i] = TanH<T, LutSize>(c_gate[i]);
  }
  const auto c_lhs = sigma_f * c_prev;
  const auto c_reg = c_lhs + sigma_i * tanh_cell;
#pragma HLS BIND_OP variable=c_lhs op=add impl=dsp
#pragma HLS BIND_OP variable=c_reg op=add impl=dsp
  c_curr = c_reg;
  for (int i = 0; i < N; ++i) {
    c_tanh[i] = TanH<T, LutSize>(c_reg[i]);
  }
  const auto h_reg = sigma_o * c_tanh;
#pragma HLS BIND_OP variable=h_reg op=mul impl=dsp // latency=3
  h_curr = h_reg;
}
#endif // end __VITIS_HLS__



/**
 * @brief      Processing element used in SvdLstm. Deprecated.
 *
 * @deprecated Old inplementation, not flexible enough.
 *
 * @param[in]  size        The size
 * @param[in]  c_t_prev    The c t previous
 * @param[in]  cur_gate_i  The current gate i
 * @param[in]  cur_gate_f  The current gate f
 * @param[in]  cur_gate_c  The current gate c
 * @param[in]  cur_gate_o  The current gate o
 * @param[in]  rec_gate_i  The record gate i
 * @param[in]  rec_gate_f  The record gate f
 * @param[in]  rec_gate_c  The record gate c
 * @param[in]  rec_gate_o  The record gate o
 * @param      h           { parameter_description }
 * @param      c_t         { parameter_description }
 * @param[in]  has_bias    Indicates if bias
 * @param[in]  i_bias      I bias
 * @param[in]  f_bias      The f bias
 * @param[in]  c_bias      The c bias
 * @param[in]  o_bias      The o bias
 *
 * @tparam     A           { description }
 * @tparam     W           { description }
 */
template <typename A, typename W>
void NonLinearityUnitPE(const int size,
                        const A *c_t_prev,
                        const A *cur_gate_i,
                        const A *cur_gate_f,
                        const A *cur_gate_c,
                        const A *cur_gate_o,
                        const A *rec_gate_i,
                        const A *rec_gate_f,
                        const A *rec_gate_c,
                        const A *rec_gate_o,
                        A *h,
                        A *c_t,
                        const bool has_bias = false,
                        const W *i_bias = nullptr,
                        const W *f_bias = nullptr,
                        const W *c_bias = nullptr,
                        const W *o_bias = nullptr) {
#pragma HLS INLINE off
  // ===========================================================================
  // Initialize the lookup table
  // ===========================================================================
#if (FIX_WIDTH == 16) || (FIX_WIDTH == 8)
  const int kTableSize = 512;
#else
  const int kTableSize = 256;
#endif
#ifndef __SYNTHESIS__
  W tanh_table[kTableSize];
  InitTanhTable<W, kTableSize>(tanh_table);
#endif
  // ===========================================================================
  // Apply non-linearities to each vector element
  // ===========================================================================
  NonLinearityUnit_Elem_Loop:
  for(int i = 0; i < size; ++i) {
#pragma HLS PIPELINE II=1
    LstmNonLinearFunctions<A, W, kTableSize>(has_bias,
      cur_gate_i[i], cur_gate_f[i], cur_gate_c[i], cur_gate_o[i],
      rec_gate_i[i], rec_gate_f[i], rec_gate_c[i], rec_gate_o[i],
      i_bias[i], f_bias[i], c_bias[i], o_bias[i],
      c_t_prev[i], c_t[i], h[i]);
  }
}

/**
 * @brief      Sub module to apply non linearities in parallel.
 *
 * @param[in]  c_t_prev                 The previous LSTM cell state (internal internal)
 * @param      current_gate_i_stream    The current gate i stream
 * @param      current_gate_f_stream    The current gate f stream
 * @param      current_gate_c_stream    The current gate c stream
 * @param      current_gate_o_stream    The current gate o stream
 * @param      recurrent_gate_i_stream  The recurrent gate i stream
 * @param      recurrent_gate_f_stream  The recurrent gate f stream
 * @param      recurrent_gate_c_stream  The recurrent gate c stream
 * @param      recurrent_gate_o_stream  The recurrent gate o stream
 * @param      h                        The LSTM output
 * @param      c_t                      The current LSTM cell state t
 *
 * @tparam     VectLength               The output dimension
 * @tparam     NumTiles                 The number of tiles the output is divided into.
 */
template <int NumElemsTile>
void NonLinearityUnitTile(const svd::ActivationD *c_t_prev,
    svd::ActivationStream &current_gate_i_stream,
    svd::ActivationStream &current_gate_f_stream,
    svd::ActivationStream &current_gate_c_stream,
    svd::ActivationStream &current_gate_o_stream,
    svd::ActivationStream &recurrent_gate_i_stream,
    svd::ActivationStream &recurrent_gate_f_stream,
    svd::ActivationStream &recurrent_gate_c_stream,
    svd::ActivationStream &recurrent_gate_o_stream,
    svd::ActivationD *h,
    svd::ActivationD *c_t,
    const bool has_bias = false,
    svd::WeightStream *i_bias_stream = nullptr,
    svd::WeightStream *f_bias_stream = nullptr,
    svd::WeightStream *c_bias_stream = nullptr,
    svd::WeightStream *o_bias_stream = nullptr) {
#pragma HLS INLINE off
  // ===========================================================================
  // Initialize the lookup table
  // ===========================================================================
#if (FIX_WIDTH == 16) || (FIX_WIDTH == 8)
  const int kTableSize = 512;
#else
  const int kTableSize = 256;
#endif
  // ===========================================================================
  // Apply non-linearities to each vector element
  // ===========================================================================
  NonLinearityUnit_Elem_Loop:
  for(int i = 0; i < NumElemsTile; ++i) {
#pragma HLS PIPELINE II=1
    svd::ActivationD cur_i = current_gate_i_stream.read();
    svd::ActivationD cur_f = current_gate_f_stream.read();
    svd::ActivationD cur_c = current_gate_c_stream.read();
    svd::ActivationD cur_o = current_gate_o_stream.read();
    svd::ActivationD rec_i = recurrent_gate_i_stream.read();
    svd::ActivationD rec_f = recurrent_gate_f_stream.read();
    svd::ActivationD rec_c = recurrent_gate_c_stream.read();
    svd::ActivationD rec_o = recurrent_gate_o_stream.read();
    WeightD i_bias_reg = 0;
    WeightD f_bias_reg = 0;
    WeightD c_bias_reg = 0;
    WeightD o_bias_reg = 0;
    if (has_bias) {
      i_bias_reg = i_bias_stream->read();
      f_bias_reg = f_bias_stream->read();
      c_bias_reg = c_bias_stream->read();
      o_bias_reg = o_bias_stream->read();
    }
    LstmNonLinearFunctions<svd::ActivationD, WeightD, kTableSize>(has_bias,
      cur_i, cur_f, cur_c, cur_o,
      rec_i, rec_f, rec_c, rec_o,
      i_bias_reg, f_bias_reg, c_bias_reg, o_bias_reg,
      c_t_prev[i], c_t[i], h[i]);
  }
}


template <int VectLength, int NumTiles, int NumGates>
void NonLinearityUnit(const svd::ActivationD *c_t_prev,
    svd::ActivationStream (&current_gate_stream)[NumGates][VectLength / NumTiles],
    svd::ActivationStream (&recurrent_gate_stream)[NumGates][VectLength / NumTiles],
    svd::ActivationD *h,
    svd::ActivationD *c_t,
    const bool has_bias = false,
    const WeightD *bias_port = nullptr) {
// #pragma HLS INLINE
// #pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS DATAFLOW
  assert(VectLength % NumTiles == 0);
  assert(NumGates >= 4);
  const int kNumElemsTile = VectLength / NumTiles;
  // NOTE: There are kNumElemsTile different streams, which are read in round
  // robin fashion. Their depth is then set as their number plus 50%.
  const int kOutputStreamDepth = kNumElemsTile + kNumElemsTile / 2;

  svd::ActivationD h_t_curr_internal[kNumElemsTile][NumTiles];
  svd::ActivationD c_t_curr_internal[kNumElemsTile][NumTiles];
  svd::ActivationD c_t_prev_internal[kNumElemsTile][NumTiles];
#pragma HLS ARRAY_PARTITION variable=h_t_curr_internal complete dim=1
#pragma HLS ARRAY_PARTITION variable=c_t_curr_internal complete dim=1
#pragma HLS ARRAY_PARTITION variable=c_t_prev_internal complete dim=1
#pragma HLS STREAM variable=h_t_curr_internal depth=NumTiles
#pragma HLS STREAM variable=c_t_curr_internal depth=kOutputStreamDepth
#pragma HLS STREAM variable=c_t_prev_internal depth=kOutputStreamDepth

  NonLinearityUnit_Read2_c_prev:
  for (int i = 0; i < NumTiles; ++i) {
    NonLinearityUnit_Read_c_prev:
    for (int j = 0; j < kNumElemsTile; ++j) {
#pragma HLS PIPELINE II=1
      c_t_prev_internal[j][i] = c_t_prev[i * kNumElemsTile + j];
    }
  }

  svd::WeightStream i_bias_streams[kNumElemsTile];
  svd::WeightStream f_bias_streams[kNumElemsTile];
  svd::WeightStream c_bias_streams[kNumElemsTile];
  svd::WeightStream o_bias_streams[kNumElemsTile];
  if (has_bias) {
#pragma HLS ARRAY_PARTITION variable=i_bias_streams complete dim=1
#pragma HLS ARRAY_PARTITION variable=f_bias_streams complete dim=1
#pragma HLS ARRAY_PARTITION variable=c_bias_streams complete dim=1
#pragma HLS ARRAY_PARTITION variable=o_bias_streams complete dim=1
#pragma HLS STREAM variable=i_bias_streams depth=NumTiles
#pragma HLS STREAM variable=f_bias_streams depth=NumTiles
#pragma HLS STREAM variable=c_bias_streams depth=NumTiles
#pragma HLS STREAM variable=o_bias_streams depth=NumTiles
    for (int i = 0; i < NumTiles; ++i) {
      for (int j = 0; j < kNumElemsTile; ++j) {
#pragma HLS PIPELINE II=1
        i_bias_streams[j].write(bias_port[i * kNumElemsTile + j]);
      }
    }
    for (int i = 0; i < NumTiles; ++i) {
      for (int j = 0; j < kNumElemsTile; ++j) {
#pragma HLS PIPELINE II=1
        f_bias_streams[j].write(bias_port[VectLength + i * kNumElemsTile + j]);
      }
    }
    for (int i = 0; i < NumTiles; ++i) {
      for (int j = 0; j < kNumElemsTile; ++j) {
#pragma HLS PIPELINE II=1
        c_bias_streams[j].write(bias_port[2 * VectLength + i * kNumElemsTile + j]);
      }
    }
    for (int i = 0; i < NumTiles; ++i) {
      for (int j = 0; j < kNumElemsTile; ++j) {
#pragma HLS PIPELINE II=1
        o_bias_streams[j].write(bias_port[3 * VectLength + i * kNumElemsTile + j]);
      }
    }
  }

  NonLinearityUnit_Tile_Loop:
  for(int i = 0; i < kNumElemsTile; ++i) {
#pragma HLS UNROLL
    NonLinearityUnitTile<NumTiles>(c_t_prev_internal[i],
                                   current_gate_stream[0][i],
                                   current_gate_stream[1][i],
                                   current_gate_stream[2][i],
                                   current_gate_stream[3][i],
                                   recurrent_gate_stream[0][i],
                                   recurrent_gate_stream[1][i],
                                   recurrent_gate_stream[2][i],
                                   recurrent_gate_stream[3][i],
                                   h_t_curr_internal[i],
                                   c_t_curr_internal[i],
                                   has_bias,
                                   &i_bias_streams[i],
                                   &f_bias_streams[i],
                                   &c_bias_streams[i],
                                   &o_bias_streams[i]);
  }

  NonLinearityUnit_Writeback2_h:
  for (int i = 0; i < NumTiles; ++i) {
    NonLinearityUnit_Writeback_h:
    for (int j = 0; j < kNumElemsTile; ++j) {
#pragma HLS PIPELINE II=1
      h[i * kNumElemsTile + j] = h_t_curr_internal[j][i];
    }
  }

  NonLinearityUnit_Writeback2_c:
  for (int i = 0; i < NumTiles; ++i) {
    NonLinearityUnit_Writeback_c:
    for (int j = 0; j < kNumElemsTile; ++j) {
#pragma HLS PIPELINE II=1
      c_t[i * kNumElemsTile + j] = c_t_curr_internal[j][i];
    }
  }
}

} // svd


#endif // end MATH_ACTIVATION_FUNCTIONS_H_