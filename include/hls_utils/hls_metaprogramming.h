#ifndef HLS_UTILS_HLS_METAPROGRAMMING_H_
#define HLS_UTILS_HLS_METAPROGRAMMING_H_

namespace hlsutils {

template <int x>
struct log2 {
  enum {value = 1 + hlsutils::log2<x / 2>::value};
};

template <>
struct log2<1> {
  enum {value = 0};
};

/**
 * @brief      Class for Greatest Common Divisor (GCD) compile-time function.
 *
 * @tparam     N     First input
 * @tparam     M     Second input
 * @tparam     K     Temporary variable used for recursion
 */
template<int N, int M, int K>
class GCDbase;

template<int N, int M>
class GCD {
public:
    static const int value = hlsutils::GCDbase<N, M, N % M>::value;
};

template<int N, int M, int K>
class GCDbase {
public:
    static const int value = hlsutils::GCDbase<M, K, M % K>::value;
};

template<int N, int M>
class GCDbase<N, M, 0>{
public:
    static const int value = M;
};

template<typename T>
struct Bitwidth {
  static const int value = T::width;
};

template<>
struct Bitwidth<float> {
  static const int value = 32;
};

template<>
struct Bitwidth<double> {
  static const int value = 64;
};

template<int X, int N, int T, int ZT>
struct PrunedSize {
  static const int value = N * X / T * (T - ZT);
};


#ifndef IS_POW2
#define IS_POW2(x) (x & (x - 1)) == 0
#endif

template <int N>
struct is_pow2 {
  static const bool value = (N & (N - 1)) == 0;
};

} // end namespace hls

#endif // end HLS_UTILS_HLS_METAPROGRAMMING_H_