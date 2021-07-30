#ifndef KERNEL_GEMV_KERNEL_H_
#define KERNEL_GEMV_KERNEL_H_

#include "assert.h"
#include "hls_stream.h"
#ifdef __VITIS_HLS__
#include "hls_vector.h"
#endif

namespace svd {

#ifdef __VITIS_HLS__
template <typename Type, int T, int N>
void GemvKernel(const int num_rows, const int num_cols,
  hls::stream<hls::vector<Type, T> > x_streams[N],
  hls::stream<hls::vector<Type, T> > w_streams[N],
  hls::stream<Type> y_streams[N]) {
  assert(num_rows % T == 0);
  const int kNumTiles = num_rows / T;

  for (int i = 0; i < num_cols; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=8 max=8
    hls::vector<Type, T> tmp[N] = {hls::vector<Type, T>(0)};
#pragma HLS ARRAY_PARTITION variable=tmp complete


    for (int j = 0; j < kNumTiles; ++j) {
#pragma HLS LOOP_TRIPCOUNT min=1024/4 max=1024/4
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
  hls::stream<hls::vector<short, 4> >& x1_port,
  hls::stream<hls::vector<short, 4> >& x2_port,
  hls::stream<hls::vector<short, 4> >& w1_port,
  hls::stream<hls::vector<short, 4> >& w2_port,
  hls::stream<short>& y1_port,
  hls::stream<short>& y2_port);

#endif // end KERNEL_GEMV_KERNEL_H_