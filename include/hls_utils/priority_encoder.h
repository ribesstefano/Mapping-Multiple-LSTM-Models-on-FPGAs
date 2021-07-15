#ifndef HLS_UTILS_PRIORITY_ENCODER_H_
#define HLS_UTILS_PRIORITY_ENCODER_H_

#include "hls_utils/hls_metaprogramming.h"

#include "assert.h"

namespace hlsutils {

/**
 * @brief      Priority Encoder: returns the MSB set bit.
 *
 * @param[in]  a         The input value.
 *
 * @tparam     Bitwidth  The bit width of the input value.
 *
 * @return     The index of the set MSB. Zero if no bit is set.
 */
template <int Bitwidth>
int PriorityEncoderMSB(const ap_uint<Bitwidth> a) {
#pragma HLS PIPELINE II=1
  int index = 0;
  for (int i = 0; i < Bitwidth; ++i) {
    if (a[i] == 1) {
      index = i;
    }
  }
  return index;
}

template <int Bitwidth>
int PriorityEncoderLSB(const ap_uint<Bitwidth> a) {
#pragma HLS PIPELINE II=1
  int index = 0;
  for (int i = Bitwidth - 1; i >= 0; --i) {
    if (a[i] == 1) {
      index = i;
    }
  }
  return index;
}

template <typename T>
int PriorityEncoderMSB(const T a) {
#pragma HLS PIPELINE II=1
  int index = 0;
  for (int i = 0; i < T::width; ++i) {
    if (a[i] == 1) {
      index = i;
    }
  }
  return index;
}

template <typename T>
int PriorityEncoderLSB(const T a) {
#pragma HLS INLINE
#pragma HLS PIPELINE II=1
  int index = 0;
  for (int i = T::width - 1; i >= 0; --i) {
    if (a[i] == 1) {
      index = i;
    }
  }
  return index;
}

template <int NumTiles>
void PriorityEncoder(const int num_zero_tiles, const ap_uint<NumTiles> a, hls::stream<ap_uint<hlsutils::log2<NumTiles>::value> > &idx_stream) {
  ap_uint<NumTiles> tmp = a;
  for (int i = 0; i < NumTiles - num_zero_tiles; ++i) {
#pragma HLS PIPELINE II=1
    int bit_idx = PriorityEncoderLSB<NumTiles>(tmp);
    assert(bit_idx < NumTiles);
    tmp[bit_idx] = 0;
    idx_stream.write(bit_idx);
  }
}

template <typename T>
void PriorityEncoder(const int num_zero_tiles, const T a, hls::stream<ap_uint<hlsutils::log2<T::width>::value> > &idx_stream) {
  T tmp = a;
  for (int i = 0; i < T::width - num_zero_tiles; ++i) {
#pragma HLS PIPELINE II=1
    int bit_idx = PriorityEncoderLSB<T::width>(tmp);
    assert(bit_idx < T::width);
    tmp[bit_idx] = 0;
    idx_stream.write(bit_idx);
  }
}

} // end namespace hlsutils

#endif // end HLS_UTILS_PRIORITY_ENCODER_H_