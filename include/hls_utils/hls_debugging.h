#ifndef HLS_UTILS_HLS_DEBUGGING
#define HLS_UTILS_HLS_DEBUGGING

#include "hls_utils/hw_timer.h"

#include <iostream>
#include <cstring>

#ifndef HLS_DEBUG_LEVEL
#define HLS_DEBUG_LEVEL 0
#endif

namespace hls_utils {

static int hls_debug_level = HLS_DEBUG_LEVEL;

void Log(const int verbose_level, const std::string &str) {
  std::cout << str << std::endl;
#ifndef __SYNTHESIS__
  if (verbose_level < hls_debug_level) {
    std::cout << str << std::endl;
  }
#endif
}


} // hls_utils

#endif // HLS_UTILS_HLS_DEBUGGING