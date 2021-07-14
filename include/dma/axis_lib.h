#ifndef DMA_AXIS_LIB_H_
#define DMA_AXIS_LIB_H_

#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"

#include <cstdio>
#include <iostream>
#include <cstring>

template<int D>
struct axiu_packet {
  ap_uint<D> data;
  ap_uint<1> last = 0;
  // ap_uint<4> keep = 0xF;
};

template <int D>
using ap_ufifo = hls::stream<axiu_packet<D> >;

/**
 * @brief      This class describes an AXI stream interface. The function using
 *             this class must apply the respective HLS directive in order
 *             to synthesize a proper AXI stream interface.
 *
 * @tparam     D     The bitwidth of the interface.
 */
template <int D>
class AxiStreamInterface {
public:
  AxiStreamInterface(hls::stream<axiu_packet<D> > &port) : _port(port) {};

  ~AxiStreamInterface() {};

  void set_name(const std::string name) {
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
  void Push(const T &x, bool is_last = false) {
    axiu_packet<D> packet;
    packet.data = *((ap_uint<D>*)&x);
    packet.last = is_last? 1 : 0;
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
  void PushLast(const T &x) {
    axiu_packet<D> packet;
    packet.data = *((ap_uint<D>*)&x);
    packet.last = 1;
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
  void PushFromBuffer(const int size, const T *x) {
    axiu_packet<D> packet;
    for (int i = 0; i < size; ++i) {
#pragma HLS PIPELINE II=1
      packet.data = *((ap_uint<D>*)&x[i]);
      if (i == size - 1) { // The last packet needs special care.
        packet.last = 1;
      }
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
  void PushFromStream(const int size, const hls::stream<T> &x) {
    axiu_packet<D> packet;
    for (int i = 0; i < size; ++i) {
#pragma HLS PIPELINE II=1
      T x_val = x.read();
      packet.data = *((ap_uint<D>*)&x_val);
      if (i == size - 1) { // The last packet needs special care.
        packet.last = 1;
      }
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
    axiu_packet<D> packet;
    packet = this->_port.read();
    return *((T*)&packet.data);
  }

  /**
   * @brief      Determines whether the specified y is the last value to pop,
   *             i.e. with TLAST set high. It also converts the read value to
   *             the specified type.
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
    axiu_packet<D> packet;
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
  void PopToBuffer(const int size, T *y) {
    axiu_packet<D> packet;
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
  void PopToStream(const int size, hls::stream<T> &y) {
    axiu_packet<D> packet;
    for (int i = 0; i < size; ++i) {
#pragma HLS PIPELINE II=1
      packet = this->_port.read();
      y.write(*((T*)&packet.data));
    }
  }

  hls::stream<axiu_packet<D> >& get_port() {
    return this->_port;
  }

private:
  hls::stream<axiu_packet<D> > &_port;
#ifndef __SYNTHESIS__
  std::string _name;
#endif
};

#endif // end DMA_AXIS_LIB_H_