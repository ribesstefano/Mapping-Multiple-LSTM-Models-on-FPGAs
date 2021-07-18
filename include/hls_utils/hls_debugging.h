#ifndef HLS_UTILS_HLS_DEBUGGING
#define HLS_UTILS_HLS_DEBUGGING

#include "hls_utils/hw_timer.h"
#ifdef __VITIS_HLS__
#include "hls_vector.h"
#endif

#include <iostream>
#include <cstring>

#ifndef HLS_DEBUG_LEVEL
#define HLS_DEBUG_LEVEL 0
#endif

namespace hlsutils {

static int hls_debug_level = HLS_DEBUG_LEVEL;

template <typename T>
void Log(const int verbose_level, const T* str) {
#ifndef __SYNTHESIS__
  if (verbose_level < hls_debug_level) {
    std::cout << str << std::endl;
  }
#endif
}

#ifdef __VITIS_HLS__
template <typename T, long long unsigned int N>
void PrintVector(hls::vector<T, N> &x) {
  for (int i = 0; i < N; ++i) {
    std::cout << x[i] << " ";
  }
  std::cout << std::endl;
}
#endif

} // hlsutils

#endif // HLS_UTILS_HLS_DEBUGGING