#ifndef KERNEL_GEMV_KERNEL_H_
#define KERNEL_GEMV_KERNEL_H_

#include "assert.h"
#include "hls_stream.h"
#ifdef __VITIS_HLS__
#include "hls_vector.h"
#endif

namespace testgemv {

typedef int DataType;
const int N = 2;
const int I = 1024;
const int T = 4;
const int R = 64;

} // testgemv

namespace svd {

/**
 * @brief      Given x with shape (N, I) and w with shape (N, I, R), returns y
 *             with shape (N, R).
 *
 *             The x streams however, contain the `N * I / T` values repeated R
 *             times (broadcasted in the R dimension).
 *
 *             The w streams instead should be broadcasted to the N dimension.
 *
 * @param[in]  num_rows   The number rows, dimension I in the example above
 * @param[in]  num_cols   The number cols, dimension R in the example above
 * @param      x_streams  The x streams
 * @param      w_streams  The w streams
 * @param      y_streams  The y streams
 *
 * @tparam     Type       The data type of the operands
 * @tparam     T          The tile size of the streams
 * @tparam     N          The number of parallel inputs.
 */
#ifdef __VITIS_HLS__
template <typename Type, int T, int N>
void GemvKernel(const int num_rows, const int num_cols,
  hls::stream<hls::vector<Type, T> > x_streams[N],
  hls::stream<hls::vector<Type, T> > w_streams[N],
  hls::stream<Type> y_streams[N]) {
  assert(num_rows % T == 0);
  const int kNumTiles = num_rows / T;
  for (int i = 0; i < num_cols; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=testgemv::R max=testgemv::R
    hls::vector<Type, T> tmp[N] = {hls::vector<Type, T>(0)};
#pragma HLS ARRAY_PARTITION variable=tmp complete
    for (int j = 0; j < kNumTiles; ++j) {
#pragma HLS LOOP_TRIPCOUNT min=testgemv::I/T max=testgemv::I/T
#pragma HLS PIPELINE II=1
      for (int k = 0; k < N; ++k) {
        tmp[k] += x_streams[k].read() * w_streams[k].read();
        if (j == kNumTiles - 1) {
          y_streams[k] << tmp[k].reduce_add();
        }
      }
    }
  }
}
#endif

} // svd

void HlsGemvKernel(const int num_rows, const int num_cols,
  hls::stream<hls::vector<testgemv::DataType, testgemv::T> >& x1_port,
  hls::stream<hls::vector<testgemv::DataType, testgemv::T> >& x2_port,
  hls::stream<hls::vector<testgemv::DataType, testgemv::T> >& w1_port,
  hls::stream<hls::vector<testgemv::DataType, testgemv::T> >& w2_port,
  hls::stream<testgemv::DataType>& y1_port,
  hls::stream<testgemv::DataType>& y2_port);

#endif // end KERNEL_GEMV_KERNEL_H_