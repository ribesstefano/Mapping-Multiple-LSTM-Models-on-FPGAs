#include "testbenches/test_dense_svd.h"
#include "dma/axis_lib.h"

#ifdef __VITIS_HLS__
#include "hls_vector.h"
#endif
#include "ap_int.h"
#include "hls_stream.h"
#include <iostream>
#include <cstdlib>

int main(int argc, char const *argv[]) {
#ifndef __VITIS_HLS__
  return 0;
#else
  return 0;
#endif // end __VITIS_HLS__
}