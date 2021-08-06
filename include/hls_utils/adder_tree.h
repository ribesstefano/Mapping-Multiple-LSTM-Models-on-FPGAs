#ifndef HLS_UTILS_ADDER_TREE_H_
#define HLS_UTILS_ADDER_TREE_H_

#include "hls_stream.h"
#include "hls_utils/hls_metaprogramming.h"

#ifdef __VITIS_HLS__
#include "hls_vector.h"
#endif

namespace hlsutils {

/**
 * @brief      Given a static array, sum-reduce all its elements.
 *
 *             NOTE: The array will be fully partitioned.
 *
 * @param      x         The input static array
 *
 * @tparam     DataType  The input and output data type
 * @tparam     NumPE     The number of array elements
 *
 * @return     The sum of all the array elements.
 */
template <typename DataType, int NumPE>
DataType adder_tree(DataType x[NumPE]) {
#pragma HLS ARRAY_PARTITION variable=x complete // to force II=1
#pragma HLS PIPELINE II=1
  // Determine the number of ranks for the adder tree and declare array:
  // - The adder_tree is larger than required as each rank only needs to be
  //   half the size of the previous rank.
  const unsigned kNumPEsLog2 = hlsutils::log2<NumPE>::value;
  const unsigned kNumPEsSub1Log2 = hlsutils::log2<NumPE - 1>::value;
  const unsigned kNumRanks = kNumPEsLog2 != kNumPEsSub1Log2 ? kNumPEsLog2 : kNumPEsLog2 + 1;
  DataType adder_tree[kNumRanks][NumPE];
#pragma HLS ARRAY_PARTITION variable=adder_tree complete dim=0

  unsigned rank_size = NumPE;
  DataType ret_val = 0;

  add_level_loop:
  for(int adder_tree_rank = kNumRanks - 1; adder_tree_rank >= 0; --adder_tree_rank) {
    const bool kLoopInit = adder_tree_rank == kNumRanks - 1 ? true : false;
    const bool kLoopEpilog = adder_tree_rank == 0 ? true : false;

    if (kLoopInit) {
      rank_size = NumPE;
    }

    const bool prev_rank_is_odd = rank_size % 2 == 0 ? false : true;
    rank_size = (rank_size + 1) / 2;

    add_col_loop:
    for(int jj = 0; jj < (NumPE + 1) / 2; ++jj) {
      if (jj < rank_size) {
        if (prev_rank_is_odd && jj == rank_size - 1) {
          // Bypass, no adder required.
          if (kLoopInit) {
            adder_tree[adder_tree_rank][jj] = x[jj * 2];
            // adder_tree[adder_tree_rank][jj] = x[jj * 2];
          } else {
            adder_tree[adder_tree_rank][jj] = adder_tree[adder_tree_rank + 1][jj * 2];
          }
        } else {
          if (kLoopInit) {
            auto y_acc = x[jj * 2] + x[jj * 2 + 1];
            // auto y_acc = x[jj * 2] + x[jj * 2 + 1];
#pragma HLS RESOURCE variable=y_acc core=AddSub_DSP
            adder_tree[adder_tree_rank][jj] = y_acc;
          } else{
            auto y_acc = adder_tree[adder_tree_rank + 1][jj * 2] + adder_tree[adder_tree_rank + 1][jj * 2 + 1];
#pragma HLS RESOURCE variable=y_acc core=AddSub_DSP
            adder_tree[adder_tree_rank][jj] = y_acc;
          }
        }
      }
    }
    if (kLoopEpilog) {
      ret_val = adder_tree[0][0];
    }
  }
  return ret_val;
}

/**
 * @brief      Given a set of parallel streams, read the first elements element
 *             from each stream and then sum-reduce them.
 *
 *             NOTE: The streams will be fully partitioned.
 *
 * @param      x         The input parallel streams
 *
 * @tparam     DataType  The input and output data type
 * @tparam     NumPE     The number of parallel streams
 *
 * @return     The sum of all the array elements.
 */
template <typename DataType, int NumPE>
DataType adder_tree(hls::stream<DataType> x[NumPE]) {
#pragma HLS ARRAY_PARTITION variable=x complete // to force II=1
#pragma HLS PIPELINE II=1
  // Determine the number of ranks for the adder tree and declare array:
  // - The adder_tree is larger than required as each rank only needs to be
  //   half the size of the previous rank.
  const unsigned kNumPEsLog2 = hlsutils::log2<NumPE>::value;
  const unsigned kNumPEsSub1Log2 = hlsutils::log2<NumPE - 1>::value;
  const unsigned kNumRanks = kNumPEsLog2 != kNumPEsSub1Log2 ? kNumPEsLog2 : kNumPEsLog2 + 1;
  DataType adder_tree[kNumRanks][NumPE];
#pragma HLS ARRAY_PARTITION variable=adder_tree complete dim=0

  unsigned rank_size = NumPE;
  DataType ret_val = 0;

  add_level_loop:
  for(int adder_tree_rank = kNumRanks - 1; adder_tree_rank >= 0; --adder_tree_rank) {
    const bool kLoopInit = adder_tree_rank == kNumRanks - 1 ? true : false;
    const bool kLoopEpilog = adder_tree_rank == 0 ? true : false;

    if (kLoopInit) {
      rank_size = NumPE;
    }

    const bool prev_rank_is_odd = rank_size % 2 == 0 ? false : true;
    rank_size = (rank_size + 1) / 2;

    add_col_loop:
    for(int jj = 0; jj < (NumPE + 1) / 2; ++jj) {
      if (jj < rank_size) {
        if (prev_rank_is_odd && jj == rank_size - 1) {
          // Bypass, no adder required.
          if (kLoopInit) {
            adder_tree[adder_tree_rank][jj] = x[jj * 2].read();
            // adder_tree[adder_tree_rank][jj] = x[jj * 2];
          } else {
            adder_tree[adder_tree_rank][jj] = adder_tree[adder_tree_rank + 1][jj * 2];
          }
        } else {
          if (kLoopInit) {
            auto y_acc = x[jj * 2].read() + x[jj * 2 + 1].read();
            // auto y_acc = x[jj * 2] + x[jj * 2 + 1];
#pragma HLS RESOURCE variable=y_acc core=AddSub_DSP
            adder_tree[adder_tree_rank][jj] = y_acc;
          } else{
            auto y_acc = adder_tree[adder_tree_rank + 1][jj * 2] + adder_tree[adder_tree_rank + 1][jj * 2 + 1];
#pragma HLS RESOURCE variable=y_acc core=AddSub_DSP
            adder_tree[adder_tree_rank][jj] = y_acc;
          }
        }
      }
    }
    if (kLoopEpilog) {
      ret_val = adder_tree[0][0];
    }
  }
  return ret_val;
}

#ifdef __VITIS_HLS__
template <typename DataType, int NumPE>
DataType adder_tree(hls::vector<DataType, NumPE> x) {
#pragma HLS ARRAY_PARTITION variable=x complete // to force II=1
#pragma HLS PIPELINE II=1
  // Determine the number of ranks for the adder tree and declare array:
  // - The adder_tree is larger than required as each rank only needs to be
  //   half the size of the previous rank.
  const unsigned kNumPEsLog2 = hlsutils::log2<NumPE>::value;
  const unsigned kNumPEsSub1Log2 = hlsutils::log2<NumPE - 1>::value;
  const unsigned kNumRanks = kNumPEsLog2 != kNumPEsSub1Log2 ? kNumPEsLog2 : kNumPEsLog2 + 1;
  DataType adder_tree[kNumRanks][NumPE];
#pragma HLS ARRAY_PARTITION variable=adder_tree complete dim=0

  unsigned rank_size = NumPE;
  DataType ret_val = 0;

  add_level_loop:
  for(int adder_tree_rank = kNumRanks - 1; adder_tree_rank >= 0; --adder_tree_rank) {
    const bool kLoopInit = adder_tree_rank == kNumRanks - 1 ? true : false;
    const bool kLoopEpilog = adder_tree_rank == 0 ? true : false;

    if (kLoopInit) {
      rank_size = NumPE;
    }

    const bool prev_rank_is_odd = rank_size % 2 == 0 ? false : true;
    rank_size = (rank_size + 1) / 2;

    add_col_loop:
    for(int jj = 0; jj < (NumPE + 1) / 2; ++jj) {
      if (jj < rank_size) {
        if (prev_rank_is_odd && jj == rank_size - 1) {
          // Bypass, no adder required.
          if (kLoopInit) {
            adder_tree[adder_tree_rank][jj] = x[jj * 2];
            // adder_tree[adder_tree_rank][jj] = x[jj * 2];
          } else {
            adder_tree[adder_tree_rank][jj] = adder_tree[adder_tree_rank + 1][jj * 2];
          }
        } else {
          if (kLoopInit) {
            auto y_acc = x[jj * 2] + x[jj * 2 + 1];
            // auto y_acc = x[jj * 2] + x[jj * 2 + 1];
#pragma HLS RESOURCE variable=y_acc core=AddSub_DSP
            adder_tree[adder_tree_rank][jj] = y_acc;
          } else{
            auto y_acc = adder_tree[adder_tree_rank + 1][jj * 2] + adder_tree[adder_tree_rank + 1][jj * 2 + 1];
#pragma HLS RESOURCE variable=y_acc core=AddSub_DSP
            adder_tree[adder_tree_rank][jj] = y_acc;
          }
        }
      }
    }
    if (kLoopEpilog) {
      ret_val = adder_tree[0][0];
    }
  }
  return ret_val;
}
#endif

} // hlsutils

#endif // end HLS_UTILS_ADDER_TREE_H_