#ifndef DMA_AXIS_LIB_H_
#define DMA_AXIS_LIB_H_

#include "hls_utils/hls_metaprogramming.h"

#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"

#include <cstdio>
#include <iostream>
#include <cstring>
#include <cassert>

#ifdef __VITIS_HLS__
#include "hls_vector.h"
#endif

namespace svd {

template<int Bitwidth>
struct AxiuPacketTlastOnlyType {
  ap_uint<Bitwidth> data;
  ap_uint<1> last = 0;
  // ap_uint<4> keep = 0xF;
};

template<int Bitwidth>
class AxiStreamFifo {
public:
  typedef ap_int<Bitwidth> PacketType;

  AxiStreamFifo(hls::stream<PacketType>& port) : _port(port) {
#pragma HLS INLINE
  }

  ~AxiStreamFifo() {};

  template<typename T>
  inline void Push(const T &x, bool is_last = false) {
#pragma HLS INLINE
    PacketType packet = *((PacketType*)&x);
    this->_port.write(packet);
  }

  /**
   * @brief      Pushes the last value.
   *
   * @param[in]  x     The value to push on the FIFO
   *
   * @tparam     T     The type of the value
   */
  template<typename T>
  inline void PushLast(const T &x) {
#pragma HLS INLINE
    PacketType packet = *((PacketType*)&x);
    this->_port.write(packet);
  }

  /**
   * @brief      Pushes a series of values from a buffer to the FIFO.
   *
   * @param[in]  size  The buffer size
   * @param[in]  x     The buffer to read from
   *
   * @tparam     T     The type of the buffer
   */
  template<typename T>
  inline void PushFromBuffer(const int size, const T *x) {
#pragma HLS INLINE
    PacketType packet;
    for (int i = 0; i < size; ++i) {
#pragma HLS PIPELINE II=1
      packet = *((PacketType*)&x[i]);
      this->_port.write(packet);
    }
  }

  /**
   * @brief      Pushes a series of values from a stream to the FIFO.
   *
   * @param[in]  size  The size
   * @param[in]  x     The stream to read from
   *
   * @tparam     T     The type of the stream
   */
  template<typename T>
  inline void PushFromStream(const int size, const hls::stream<T> &x) {
#pragma HLS INLINE
    PacketType packet;
    for (int i = 0; i < size; ++i) {
#pragma HLS PIPELINE II=1
      T x_val = x.read();
      packet = *((PacketType*)&x_val);
      this->_port.write(packet);
    }
  }

  /**
   * @brief      Pops a value from the FIFO and converts it.
   *
   * @tparam     T     The type of the returned value
   *
   * @return     The value from the FIFO
   */
  template<typename T>
  T Pop() {
    PacketType packet = this->_port.read();
    return *((T*)&packet);
  }

  /**
   * @brief      Read value and returns false (used for compatibility).
   *
   * @param      y     The value read from the FIFO
   *
   * @tparam     T     The type of the read value
   *
   * @return     True if the specified y is the last value to pop, False
   *             otherwise.
   */
  template<typename T>
  bool isLastPop(T &y) {
    PacketType packet = this->_port.read();
    y = *((T*)&packet);
    return false;
  }

  /**
   * @brief      Pops a series of values from the FIFO and writes them into a
   *             buffer. It also converts from ap_uint<> to T.
   *
   * @param[in]  size  The size
   * @param      y     The output buffer
   *
   * @tparam     T     The type of the output buffer
   */
  template<typename T>
  inline void PopToBuffer(const int size, T *y) {
#pragma HLS INLINE
    PacketType packet;
    for (int i = 0; i < size; ++i) {
#pragma HLS PIPELINE II=1
      packet = this->_port.read();
      y[i] = *((T*)&packet);
    }
  }

  /**
   * @brief      Pops a series of values from the FIFO and writes them into a
   *             stream. It also converts from ap_uint<> to T.
   *
   * @param[in]  size  The stream size
   * @param      y     The output stream
   *
   * @tparam     T     The type of the output stream
   */
  template<typename T>
  inline void PopToStream(const int size, hls::stream<T> &y) {
#pragma HLS INLINE
    PacketType packet;
    for (int i = 0; i < size; ++i) {
#pragma HLS PIPELINE II=1
      packet = this->_port.read();
      y.write(*((T*)&packet));
    }
  }

#ifdef __VITIS_HLS__
    /**
     * @brief      Push a vector into the FIFO with default TLAST set to low.
     *             From the AXIS specification: The following options are
     *             available:
     *             * Set TLAST LOW. This indicates that all transfers are within
     *               the same packet. This option provides maximum opportunity
     *               for merging and upsizing but means that transfers could be
     *               delayed in a stream with intermittent bursts. A permanently
     *               LOW TLAST signal might also affect the interleaving of
     *               streams across a shared channel because the interconnect
     *               can use the TLAST signal to influence the arbitration
     *               process.
     *             * Set TLAST HIGH. This indicates that all transfers are
     *               individual packets. This option ensures that transfers do
     *               not get delayed in the infrastructure. It also ensures that
     *               the stream does not unnecessarily block the use of a shared
     *               channel by influencing the arbitration scheme. This option
     *               prevents merging on streams from masters that have this
     *               default setting and prevents efficient upsizing.
     *
     * @param[in]  x        The vector to push
     * @param[in]  is_last  Indicates if last packet to push. Default false.
     *
     * @tparam     T        The type of the vector, its type must be of the same
     *                      size of the FIFO.
     * @tparam     N        Number of elements in the vector
     */
  template<typename T, int N>
  inline void PushVector(const hls::vector<T, N>& x, bool is_last = false) {
#pragma HLS INLINE
    static_assert(hlsutils::Bitwidth<T>::value * N == Bitwidth, "AxiStreamPort must have same bitwidth as hls::vector");
    assert(hlsutils::Bitwidth<T>::value * N == Bitwidth);
    const int kElemBitwidth = hlsutils::Bitwidth<T>::value;
    PacketType packet;
    for (int i = 0; i < N; ++i) {
      const int kHi = (i + 1) * kElemBitwidth - 1;
      const int kLo = i * kElemBitwidth;
      auto tmp = x[i];
      packet.range(kHi, kLo) = *((ap_uint<kElemBitwidth>*)&tmp);
    }
    this->_port.write(packet);
  }

  /**
   * @brief      Pushes the last vector, i.e. a packet with TLAST set to high.
   *
   * @param[in]  x     The vector to push on the FIFO
   *
   * @tparam     T     The type of the vector
   * @tparam     N     Number of elements in the vector
   */
  template<typename T, int N>
  inline void PushLastVector(const hls::vector<T, N>& x) {
#pragma HLS INLINE
    static_assert(hlsutils::Bitwidth<T>::value * N == Bitwidth, "AxiStreamPort must have same bitwidth as hls::vector");
    assert(hlsutils::Bitwidth<T>::value * N == Bitwidth);
    const int kElemBitwidth = hlsutils::Bitwidth<T>::value;
    PacketType packet;
    for (int i = 0; i < N; ++i) {
      const int kHi = (i + 1) * kElemBitwidth - 1;
      const int kLo = i * kElemBitwidth;
      auto tmp = x[i];
      packet.range(kHi, kLo) = *((ap_uint<kElemBitwidth>*)&tmp);
    }
    this->_port.write(packet);
  }

  /**
   * @brief      Pops a vector from the FIFO and converts it.
   *
   * @tparam     T     The type of the returned vector
   * @tparam     N     The number of elements in the vector
   *
   * @return     The vector from the FIFO
   */
  template<typename T, int N>
  hls::vector<T, N> PopVector() {
#pragma HLS INLINE
    static_assert(hlsutils::Bitwidth<T>::value * N == Bitwidth, "AxiStreamPort must have same bitwidth as hls::vector");
    assert(hlsutils::Bitwidth<T>::value * N == Bitwidth);
    const int kElemBitwidth = hlsutils::Bitwidth<T>::value;
    PacketType packet = this->_port.read();
    hls::vector<T, N> y;
    for (int i = 0; i < N; ++i) {
      const int kHi = (i + 1) * kElemBitwidth - 1;
      const int kLo = i * kElemBitwidth;
      ap_uint<kElemBitwidth> tmp = packet.range(kHi, kLo);
      y[i] = *((T*)&tmp);
    }
    return y;
  }

  /**
   * @brief      Read vector and returns false (used for compatibility reasons).
   *
   * @param      y     The vector read from the FIFO
   *
   * @tparam     T     The type of the read vector
   * @tparam     N     The number of elements in the vector.
   *
   * @return     False
   */
  template<typename T, int N>
  bool isLastPopVector(hls::vector<T, N>& y) {
#pragma HLS INLINE
    static_assert(hlsutils::Bitwidth<T>::value * N == Bitwidth, "AxiStreamPort must have same bitwidth as hls::vector");
    assert(hlsutils::Bitwidth<T>::value * N == Bitwidth);
    const int kElemBitwidth = hlsutils::Bitwidth<T>::value;
    PacketType packet = this->_port.read();
    for (int i = 0; i < N; ++i) {
      const int kHi = (i + 1) * kElemBitwidth - 1;
      const int kLo = i * kElemBitwidth;
      ap_uint<kElemBitwidth> tmp = packet.range(kHi, kLo);
      y[i] = *((T*)&tmp);
    }
    return false;
  }
#endif // __VITIS_HLS__

  hls::stream<PacketType>& get_port() {
    return this->_port;
  }

private:
  hls::stream<PacketType>& _port;
};

/**
 * @brief      Wrapper class for an AXI stream interface port.
 *
 *             The function instantiating this class must apply the respective
 *             HLS directive in order to synthesize a proper AXI stream
 *             interface.
 *
 *             For documentation on TKEEP and TSTRB, please visit:
 *             https://developer.arm.com/documentation/ihi0051/a/Interface-Signals/Byte-qualifiers/TKEEP-and-TSTRB-combinations
 *
 * @tparam     Bitwidth  The bitwidth of the interface.
 */
template <int Bitwidth>
class AxiStreamPort {
public:
  // typedef AxiuPacketTlastOnlyType<Bitwidth> PacketType;
  typedef ap_axiu<Bitwidth, 0, 0, 0> PacketType;
  typedef ap_uint<hls::bytewidth<ap_uint<Bitwidth> > > SideChannelsType;

  AxiStreamPort(hls::stream<PacketType>& port) : _port(port),
    _all_ones(~(SideChannelsType(0))), _has_side_channels(true) {
#pragma HLS INLINE
  };

  ~AxiStreamPort() {};

  inline void set_name(const std::string name) {
#ifndef __SYNTHESIS__
    _name = name;
#endif
  }

  std::string name() {
#ifndef __SYNTHESIS__
    return this->_name;
#else
    return "";
#endif
  }

  /**
   * @brief      Push a value into the FIFO with default TLAST set to low. From
   *             the AXIS specification: The following options are available:
   *             * Set TLAST LOW. This indicates that all transfers are within
   *               the same packet. This option provides maximum opportunity for
   *               merging and upsizing but means that transfers could be
   *               delayed in a stream with intermittent bursts. A permanently
   *               LOW TLAST signal might also affect the interleaving of
   *               streams across a shared channel because the interconnect can
   *               use the TLAST signal to influence the arbitration process.
   *             * Set TLAST HIGH. This indicates that all transfers are
   *               individual packets. This option ensures that transfers do not
   *               get delayed in the infrastructure. It also ensures that the
   *               stream does not unnecessarily block the use of a shared
   *               channel by influencing the arbitration scheme. This option
   *               prevents merging on streams from masters that have this
   *               default setting and prevents efficient upsizing.
   *
   * @param[in]  x        The value to push
   * @param[in]  is_last  Indicates if last packet to push. Default false.
   *
   * @tparam     T        The type of the value, its type must be of the same
   *                      size of the FIFO.
   */
  template<typename T>
  inline void Push(const T &x, bool is_last = false) {
#pragma HLS INLINE
    PacketType packet;
    packet.data = *((ap_uint<Bitwidth>*)&x);
    packet.last = is_last? 1 : 0;
    // NOTE: If TKEEP and TSTRB both high, the packet is a data type.
    packet.keep = this->_all_ones; // Set TKEEP to all ones.
    packet.strb = this->_all_ones; // Set TSTRB to all ones.
    this->_port.write(packet);
  }

  /**
   * @brief      Pushes the last value, i.e. a packet with TLAST set to high.
   *
   * @param[in]  x     The value to push on the FIFO
   *
   * @tparam     T     The type of the value
   */
  template<typename T>
  inline void PushLast(const T &x) {
#pragma HLS INLINE
    PacketType packet;
    packet.data = *((ap_uint<Bitwidth>*)&x);
    packet.last = 1;
    // NOTE: If TKEEP and TSTRB both high, the packet is a data type.
    packet.keep = this->_all_ones; // Set TKEEP to all ones.
    packet.strb = this->_all_ones; // Set TSTRB to all ones.
    this->_port.write(packet);
  }

  /**
   * @brief      Pushes a series of values from a buffer to the FIFO.
   *
   * @param[in]  size  The buffer size
   * @param[in]  x     The buffer to read from
   *
   * @tparam     T     The type of the buffer
   */
  template<typename T>
  inline void PushFromBuffer(const int size, const T *x) {
#pragma HLS INLINE
    PacketType packet;
    for (int i = 0; i < size; ++i) {
#pragma HLS PIPELINE II=1
      packet.data = *((ap_uint<Bitwidth>*)&x[i]);
      if (i == size - 1) { // The last packet needs special care.
        packet.last = 1;
      }
      // NOTE: If TKEEP and TSTRB both high, the packet is a data type.
      packet.keep = this->_all_ones; // Set TKEEP to all ones.
      packet.strb = this->_all_ones; // Set TSTRB to all ones.
      this->_port.write(packet);
    }
  }

  /**
   * @brief      Pushes a series of values from a stream to the FIFO.
   *
   * @param[in]  size  The size
   * @param[in]  x     The stream to read from
   *
   * @tparam     T     The type of the stream
   */
  template<typename T>
  inline void PushFromStream(const int size, const hls::stream<T> &x) {
#pragma HLS INLINE
    PacketType packet;
    for (int i = 0; i < size; ++i) {
#pragma HLS PIPELINE II=1
      T x_val = x.read();
      packet.data = *((ap_uint<Bitwidth>*)&x_val);
      if (i == size - 1) { // The last packet needs special care.
        packet.last = 1;
      }
      // NOTE: If TKEEP and TSTRB both high, the packet is a data type.
      packet.keep = this->_all_ones; // Set TKEEP to all ones.
      packet.strb = this->_all_ones; // Set TSTRB to all ones.
      this->_port.write(packet);
    }
  }

  /**
   * @brief      Pops a value from the FIFO and converts it.
   *
   * @tparam     T     The type of the returned value
   *
   * @return     The value from the FIFO
   */
  template<typename T>
  T Pop() {
    PacketType packet;
    packet = this->_port.read();
    return *((T*)&packet.data);
  }

  /**
   * @brief      Read value and return true if the specified y is the last value
   *             to pop, i.e. with TLAST set high. It also converts the read
   *             value to the specified type.
   *
   * @param      y     The value read from the FIFO
   *
   * @tparam     T     The type of the read value
   *
   * @return     True if the specified y is the last value to pop, False
   *             otherwise.
   */
  template<typename T>
  bool isLastPop(T &y) {
    PacketType packet;
    packet = this->_port.read();
    y = *((T*)&packet.data);
    return packet.last == 1 ? true : false;
  }

  /**
   * @brief      Pops a series of values from the FIFO and writes them into a
   *             buffer. It also converts from ap_uint<> to T.
   *
   * @param[in]  size  The size
   * @param      y     The output buffer
   *
   * @tparam     T     The type of the output buffer
   */
  template<typename T>
  inline void PopToBuffer(const int size, T *y) {
#pragma HLS INLINE
    PacketType packet;
    for (int i = 0; i < size; ++i) {
#pragma HLS PIPELINE II=1
      packet = this->_port.read();
      y[i] = *((T*)&packet.data);
    }
  }

  /**
   * @brief      Pops a series of values from the FIFO and writes them into a
   *             stream. It also converts from ap_uint<> to T.
   *
   * @param[in]  size  The stream size
   * @param      y     The output stream
   *
   * @tparam     T     The type of the output stream
   */
  template<typename T>
  inline void PopToStream(const int size, hls::stream<T> &y) {
#pragma HLS INLINE
    PacketType packet;
    for (int i = 0; i < size; ++i) {
#pragma HLS PIPELINE II=1
      packet = this->_port.read();
      y.write(*((T*)&packet.data));
    }
  }

#ifdef __VITIS_HLS__
    /**
     * @brief      Push a vector into the FIFO with default TLAST set to low.
     *             From the AXIS specification: The following options are
     *             available:
     *             * Set TLAST LOW. This indicates that all transfers are within
     *               the same packet. This option provides maximum opportunity
     *               for merging and upsizing but means that transfers could be
     *               delayed in a stream with intermittent bursts. A permanently
     *               LOW TLAST signal might also affect the interleaving of
     *               streams across a shared channel because the interconnect
     *               can use the TLAST signal to influence the arbitration
     *               process.
     *             * Set TLAST HIGH. This indicates that all transfers are
     *               individual packets. This option ensures that transfers do
     *               not get delayed in the infrastructure. It also ensures that
     *               the stream does not unnecessarily block the use of a shared
     *               channel by influencing the arbitration scheme. This option
     *               prevents merging on streams from masters that have this
     *               default setting and prevents efficient upsizing.
     *
     * @param[in]  x        The vector to push
     * @param[in]  is_last  Indicates if last packet to push. Default false.
     *
     * @tparam     T        The type of the vector, its type must be of the same
     *                      size of the FIFO.
     * @tparam     N        Number of elements in the vector
     */
  template<typename T, int N>
  inline void PushVector(const hls::vector<T, N>& x, bool is_last = false) {
#pragma HLS INLINE
    static_assert(hlsutils::Bitwidth<T>::value * N == Bitwidth, "AxiStreamPort must have same bitwidth as hls::vector");
    assert(hlsutils::Bitwidth<T>::value * N == Bitwidth);
    const int kElemBitwidth = hlsutils::Bitwidth<T>::value;
    PacketType packet;
    for (int i = 0; i < N; ++i) {
      const int kHi = (i + 1) * kElemBitwidth - 1;
      const int kLo = i * kElemBitwidth;
      auto tmp = x[i];
      packet.data.range(kHi, kLo) = *((ap_uint<kElemBitwidth>*)&tmp);
    }
    packet.last = is_last? 1 : 0;
    // NOTE: If TKEEP and TSTRB both high, the packet is a data type.
    packet.keep = this->_all_ones; // Set TKEEP to all ones.
    packet.strb = this->_all_ones; // Set TSTRB to all ones.
    this->_port.write(packet);
  }

  /**
   * @brief      Pushes the last vector, i.e. a packet with TLAST set to high.
   *
   * @param[in]  x     The vector to push on the FIFO
   *
   * @tparam     T     The type of the vector
   * @tparam     N     Number of elements in the vector
   */
  template<typename T, int N>
  inline void PushLastVector(const hls::vector<T, N>& x) {
#pragma HLS INLINE
    static_assert(hlsutils::Bitwidth<T>::value * N == Bitwidth, "AxiStreamPort must have same bitwidth as hls::vector");
    assert(hlsutils::Bitwidth<T>::value * N == Bitwidth);
    const int kElemBitwidth = hlsutils::Bitwidth<T>::value;
    PacketType packet;
    for (int i = 0; i < N; ++i) {
      const int kHi = (i + 1) * kElemBitwidth - 1;
      const int kLo = i * kElemBitwidth;
      auto tmp = x[i];
      packet.data.range(kHi, kLo) = *((ap_uint<kElemBitwidth>*)&tmp);
    }
    packet.last = 1;
    // NOTE: If TKEEP and TSTRB both high, the packet is a data type.
    packet.keep = this->_all_ones; // Set TKEEP to all ones.
    packet.strb = this->_all_ones; // Set TSTRB to all ones.
    this->_port.write(packet);
  }

  /**
   * @brief      Pops a vector from the FIFO and converts it.
   *
   * @tparam     T     The type of the returned vector
   * @tparam     N     The number of elements in the vector
   *
   * @return     The vector from the FIFO
   */
  template<typename T, int N>
  hls::vector<T, N> PopVector() {
#pragma HLS INLINE
    static_assert(hlsutils::Bitwidth<T>::value * N == Bitwidth, "AxiStreamPort must have same bitwidth as hls::vector");
    assert(hlsutils::Bitwidth<T>::value * N == Bitwidth);
    const int kElemBitwidth = hlsutils::Bitwidth<T>::value;
    PacketType packet;
    packet = this->_port.read();
    hls::vector<T, N> y;
    for (int i = 0; i < N; ++i) {
      const int kHi = (i + 1) * kElemBitwidth - 1;
      const int kLo = i * kElemBitwidth;
      ap_uint<kElemBitwidth> tmp = packet.data.range(kHi, kLo);
      y[i] = *((T*)&tmp);
    }
    return y;
  }

  /**
   * @brief      Read vector and return true if the specified y is the last
   *             vector to pop, i.e. with TLAST set high. It also converts the
   *             read vector to the specified type.
   *
   * @param      y     The vector read from the FIFO
   *
   * @tparam     T     The type of the read vector
   * @tparam     N     The number of elements in the vector.
   *
   * @return     True if the specified y is the last vector to pop, False
   *             otherwise.
   */
  template<typename T, int N>
  bool isLastPopVector(hls::vector<T, N>& y) {
#pragma HLS INLINE
    static_assert(hlsutils::Bitwidth<T>::value * N == Bitwidth, "AxiStreamPort must have same bitwidth as hls::vector");
    assert(hlsutils::Bitwidth<T>::value * N == Bitwidth);
    const int kElemBitwidth = hlsutils::Bitwidth<T>::value;
    PacketType packet;
    packet = this->_port.read();
    for (int i = 0; i < N; ++i) {
      const int kHi = (i + 1) * kElemBitwidth - 1;
      const int kLo = i * kElemBitwidth;
      ap_uint<kElemBitwidth> tmp = packet.data.range(kHi, kLo);
      y[i] = *((T*)&tmp);
    }
    return packet.last == 1 ? true : false;
  }
#endif // __VITIS_HLS__

  hls::stream<PacketType>& get_port() {
    return this->_port;
  }

private:
  hls::stream<PacketType>& _port;
  SideChannelsType _all_ones;
  bool _has_side_channels;
#ifndef __SYNTHESIS__
  std::string _name;
#endif
};

/**
 * @brief      This class describes an AXI stream interface (Policy-based
 *             design).
 *
 *             It has to be used as a "generic" interface whithin a kernel. The
 *             port of the kernel attached to this class can then be either a
 *             FIFO or a AXIS port.
 *
 * @tparam     AxiClass  The policy class.
 */
template <typename AxiClass>
class AxiStreamInterface : private AxiClass {
public:
  AxiStreamInterface(hls::stream<typename AxiClass::PacketType>& port): AxiClass(port) {
#pragma HLS INLINE
  }

  ~AxiStreamInterface() {};

  template<typename T>
  inline void Push(const T &x, bool is_last = false) {
#pragma HLS INLINE
    AxiClass::template Push<T>(x, is_last);
  }

  /**
   * @brief      Pushes the last value, i.e. a packet with TLAST set to high.
   *
   * @param[in]  x     The value to push on the FIFO
   *
   * @tparam     T     The type of the value
   */
  template<typename T>
  inline void PushLast(const T &x) {
#pragma HLS INLINE
    AxiClass::template PushLast<T>(x);
  }

  /**
   * @brief      Pushes a series of values from a buffer to the FIFO.
   *
   * @param[in]  size  The buffer size
   * @param[in]  x     The buffer to read from
   *
   * @tparam     T     The type of the buffer
   */
  template<typename T>
  inline void PushFromBuffer(const int size, const T *x) {
#pragma HLS INLINE
    AxiClass::template PushFromBuffer<T>(size, x);
  }

  /**
   * @brief      Pushes a series of values from a stream to the FIFO.
   *
   * @param[in]  size  The size
   * @param[in]  x     The stream to read from
   *
   * @tparam     T     The type of the stream
   */
  template<typename T>
  inline void PushFromStream(const int size, const hls::stream<T> &x) {
#pragma HLS INLINE
    AxiClass::template PushFromStream<T>(size, x);
  }

  /**
   * @brief      Pops a value from the FIFO and converts it.
   *
   * @tparam     T     The type of the returned value
   *
   * @return     The value from the FIFO
   */
  template<typename T>
  T Pop() {
#pragma HLS INLINE
    return AxiClass::template Pop<T>();
  }

  /**
   * @brief      Read value and return true if the specified y is the last value
   *             to pop, i.e. with TLAST set high. It also converts the read
   *             value to the specified type.
   *
   * @param      y     The value read from the FIFO
   *
   * @tparam     T     The type of the read value
   *
   * @return     True if the specified y is the last value to pop, False
   *             otherwise.
   */
  template<typename T>
  bool isLastPop(T &y) {
#pragma HLS INLINE
    return AxiClass::template isLastPop<T>(y);
  }

  /**
   * @brief      Pops a series of values from the FIFO and writes them into a
   *             buffer. It also converts from ap_uint<> to T.
   *
   * @param[in]  size  The size
   * @param      y     The output buffer
   *
   * @tparam     T     The type of the output buffer
   */
  template<typename T>
  inline void PopToBuffer(const int size, T *y) {
#pragma HLS INLINE
    AxiClass::template PopToBuffer<T>(size, y);
  }

  /**
   * @brief      Pops a series of values from the FIFO and writes them into a
   *             stream. It also converts from ap_uint<> to T.
   *
   * @param[in]  size  The stream size
   * @param      y     The output stream
   *
   * @tparam     T     The type of the output stream
   */
  template<typename T>
  inline void PopToStream(const int size, hls::stream<T> &y) {
#pragma HLS INLINE
    AxiClass::template PopToStream<T>(size, y);
  }

#ifdef __VITIS_HLS__
    /**
     * @brief      Push a vector into the FIFO with default TLAST set to low.
     *             From the AXIS specification: The following options are
     *             available:
     *             * Set TLAST LOW. This indicates that all transfers are within
     *               the same packet. This option provides maximum opportunity
     *               for merging and upsizing but means that transfers could be
     *               delayed in a stream with intermittent bursts. A permanently
     *               LOW TLAST signal might also affect the interleaving of
     *               streams across a shared channel because the interconnect
     *               can use the TLAST signal to influence the arbitration
     *               process.
     *             * Set TLAST HIGH. This indicates that all transfers are
     *               individual packets. This option ensures that transfers do
     *               not get delayed in the infrastructure. It also ensures that
     *               the stream does not unnecessarily block the use of a shared
     *               channel by influencing the arbitration scheme. This option
     *               prevents merging on streams from masters that have this
     *               default setting and prevents efficient upsizing.
     *
     * @param[in]  x        The vector to push
     * @param[in]  is_last  Indicates if last packet to push. Default false.
     *
     * @tparam     T        The type of the vector, its type must be of the same
     *                      size of the FIFO.
     * @tparam     N        Number of elements in the vector
     */
  template<typename T, int N>
  inline void PushVector(const hls::vector<T, N>& x, bool is_last = false) {
#pragma HLS INLINE
    AxiClass::template PushVector<T, N>(x, is_last);
  }

  /**
   * @brief      Pushes the last vector, i.e. a packet with TLAST set to high.
   *
   * @param[in]  x     The vector to push on the FIFO
   *
   * @tparam     T     The type of the vector
   * @tparam     N     Number of elements in the vector
   */
  template<typename T, int N>
  inline void PushLastVector(const hls::vector<T, N>& x) {
#pragma HLS INLINE
    AxiClass::template PushLastVector<T, N>(x);
  }

  /**
   * @brief      Pops a vector from the FIFO and converts it.
   *
   * @tparam     T     The type of the returned vector
   * @tparam     N     The number of elements in the vector
   *
   * @return     The vector from the FIFO
   */
  template<typename T, int N>
  hls::vector<T, N> PopVector() {
#pragma HLS INLINE
    return AxiClass::template PopVector<T, N>();
  }

  /**
   * @brief      Read vector and return true if the specified y is the last
   *             vector to pop, i.e. with TLAST set high. It also converts the
   *             read vector to the specified type.
   *
   * @param      y     The vector read from the FIFO
   *
   * @tparam     T     The type of the read vector
   * @tparam     N     The number of elements in the vector.
   *
   * @return     True if the specified y is the last vector to pop, False
   *             otherwise.
   */
  template<typename T, int N>
  bool isLastPopVector(hls::vector<T, N>& y) {
#pragma HLS INLINE
    return AxiClass::template isLastPopVector<T, N>(y);
  }
#endif // __VITIS_HLS__
};

} // svd

#endif // end DMA_AXIS_LIB_H_