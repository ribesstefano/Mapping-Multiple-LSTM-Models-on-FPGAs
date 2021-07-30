#include "kernel/gemv_kernel.h"

#ifdef __VITIS_HLS__

void HlsGemvKernel(const int num_rows, const int num_cols,
  hls::stream<hls::vector<short, 4> >& x1_port,
  hls::stream<hls::vector<short, 4> >& x2_port,
  hls::stream<hls::vector<short, 4> >& w1_port,
  hls::stream<hls::vector<short, 4> >& w2_port,
  hls::stream<short>& y1_port,
  hls::stream<short>& y2_port) {
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl
#pragma HLS INTERFACE s_axilite port=num_cols bundle=ctrl
#pragma HLS INTERFACE s_axilite port=num_rows bundle=ctrl
#pragma HLS DATAFLOW

  hls::stream<hls::vector<short, 4> > x_streams[2];
  hls::stream<hls::vector<short, 4> > w_streams[2];
  hls::stream<short> y_streams[2];
#pragma HLS ARRAY_PARTITION variable=x_streams complete
#pragma HLS ARRAY_PARTITION variable=w_streams complete
#pragma HLS ARRAY_PARTITION variable=y_streams complete


  const int kNumTiles = num_rows / 4;

  DMA_in:
  for (int i = 0; i < kNumTiles; ++i) {
    for (int j = 0; j < num_cols; ++j) {
#pragma HLS PIPELINE II=1
      x_streams[0] << x1_port.read();
      x_streams[1] << x2_port.read();
      w_streams[0] << w1_port.read();
      w_streams[1] << w2_port.read();
    }
  }

  svd::GemvKernel<short, 4, 2>(num_rows, num_cols, x_streams, w_streams, y_streams);

  DMA_out:
  for (int i = 0; i < num_cols; ++i) {
#pragma HLS PIPELINE II=1
    y1_port.write(y_streams[0].read());
    y2_port.write(y_streams[1].read());
  }
}

#endif
