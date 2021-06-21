#ifndef MATH_ACTIVATION_FUNCTIONS_H_
#define MATH_ACTIVATION_FUNCTIONS_H_

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
 * @param      rec_gate_stream  The record gate stream
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
      ActivationD i_gate = 0;
      ActivationD f_gate = 0;
      ActivationD c_gate = 0;
      ActivationD o_gate = 0;
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


#endif // end MATH_ACTIVATION_FUNCTIONS_H_