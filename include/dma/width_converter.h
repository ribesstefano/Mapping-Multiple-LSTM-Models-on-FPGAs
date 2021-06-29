#ifndef DMA_WIDTH_CONVERTER_H_
#define DMA_WIDTH_CONVERTER_H_

#include "ap_int.h"

#include <type_traits>
#include <cstdint>

/**
 * @brief      Adjust stream width. Used for DMA-ing input ports.
 *
 * @param[in]  in          The input stream
 * @param[out] out         The output stream
 *
 * @tparam     OutD        { description }
 * @tparam     InWidth     Width of input stream
 * @tparam     OutWidth    Width of output stream
 * @tparam     NumInWords  Number of input words (OutWidth) to process
 */
template <typename OutD, int InWidth, int OutWidth, int NumInWords>
void Mem2MemDataWidthConverter(const ap_uint<InWidth> *in, OutD *out) {
  assert(InWidth % 8 == 0);
  assert(OutWidth % 8 == 0);
  if (InWidth > OutWidth) {
    // Store multiple output words per input word read
    assert(InWidth % OutWidth == 0);
    const unsigned kOutPerIn = InWidth / OutWidth;
    unsigned out_idx = 0;
    unsigned out_addr = 0;
    ap_uint<InWidth> elem_in = 0;
    for (int i = 0; i < NumInWords; ++i) {
#pragma HLS PIPELINE II=1
      if (out_idx == 0) {
        elem_in = in[i];
      }
      // TODO(15/03/2019 - performance opt)
      // if constexpr (std::is_same<OutD, float>::value || std::is_same<OutD, double>::value) {
      //   out[out_addr] = elem_in(OutWidth - 1, 0);
      // } else {
      //   out[out_addr].range() = elem_in(OutWidth - 1, 0);
      // }
#if USE_FIX
      out[out_addr].range() = elem_in(OutWidth - 1, 0);
#else
      out[out_addr] = elem_in(OutWidth - 1, 0);
#endif
      elem_in = elem_in >> OutWidth;
      out_addr++;
      out_idx++;
      // Wraparound indices to recreate the nested loop structure
      if (out_idx == kOutPerIn) {
        out_idx = 0;
      }
    }
  } else if (InWidth == OutWidth) {
    // Read multiple input words per output word stored
    assert(OutWidth % InWidth == 0);
  } else { // InWidth < OutWidth
    // Read multiple input words per output word stored
    assert(OutWidth % InWidth == 0);
  }
}

/**
 * @brief      Adjust stream width. Used for DMA-ing input ports.
 *
 * @param[in]  num_in_words  Number of input words (OutWidth) to process
 * @param[in]  in            The input stream
 * @param[out] out           The output stream
 *
 * @tparam     OutD          { description }
 * @tparam     InWidth       Width of input stream
 * @tparam     OutWidth      Width of output stream
 * @tparam     NumInWords  Number of input words (OutWidth) to process
 */
template <typename OutD, int InWidth, int OutWidth>
void Mem2MemDataWidthConverter(const int num_in_words,
                           const ap_uint<InWidth> *in, OutD *out) {
  assert(InWidth % 8 == 0);
  assert(OutWidth % 8 == 0);
  if (InWidth > OutWidth) {
    // Store multiple output words per input word read
    assert(InWidth % OutWidth == 0);
    const unsigned kOutPerIn = InWidth / OutWidth;
    unsigned out_idx = 0;
    unsigned out_addr = 0;
    ap_uint<InWidth> elem_in = 0;
    for (int i = 0; i < num_in_words; ++i) {
#pragma HLS PIPELINE II=1
      if (out_idx == 0) {
        elem_in = in[i];
      }
      // TODO(15/03/2019 - performance opt)
      // if constexpr (std::is_same<OutD, float>::value || std::is_same<OutD, double>::value) {
      //   out[out_addr] = elem_in(OutWidth - 1, 0);
      // } else {
      //   out[out_addr].range() = elem_in(OutWidth - 1, 0);
      // }
#if USE_FIX
      out[out_addr].range() = elem_in(OutWidth - 1, 0);
#else
      out[out_addr] = elem_in(OutWidth - 1, 0);
#endif
      elem_in = elem_in >> OutWidth;
      // Wraparound indices to recreate the nested loop structure
      if (out_idx == kOutPerIn - 1) {
        out_idx = 0;
      }
      out_addr++;
      out_idx++;
    }
  } else if (InWidth == OutWidth) {
    assert(OutWidth % InWidth == 0);
    ap_uint<InWidth> elem_in = 0;
    for (int i = 0; i < num_in_words; ++i) {
#pragma HLS PIPELINE II=1
      elem_in = in[i];
#if USE_FIX
      out[i].range() = elem_in;
#else
      out[i] = elem_in;
#endif
    }
  }
}

template <typename InD, int InWidth, int OutWidth>
void Mem2MemDataWidthConverter(const int num_in_words,
                           const InD *in, ap_uint<OutWidth> *out) {
  assert(InWidth % 8 == 0);
  assert(OutWidth % 8 == 0);
  if (InWidth < OutWidth) {
    // Read multiple input words per output word stored
    assert(OutWidth % InWidth == 0);


    const unsigned kOutPerIn = InWidth / OutWidth;
    const unsigned kInPerOut = OutWidth / InWidth;
    unsigned out_idx = 0;
    unsigned out_addr = 0;
    ap_uint<InWidth> elem_in = 0;
    ap_uint<OutWidth> elem_out = 0;

    for (int i = 0; i < num_in_words; ++i) {
#pragma HLS PIPELINE II=1
      const int kHi = ((i + 1) * InWidth) % OutWidth - 1;
      const int kLo = (i * InWidth) % OutWidth;
      std::cout << "(" << kHi << ", " << kLo << ")\n";

      // if constexpr (std::is_same<InD, float>::value || std::is_same<InD, double>::value) {
      //   elem_out(kHi, kLo) = in[i];
      // } else {
      //   elem_out(kHi, kLo) = in[i].range();
      // }
#if USE_FIX
      elem_out(kHi, kLo) = in[i].range();
#else
      elem_out(kHi, kLo) = in[i];
#endif
      // Wraparound indices to recreate the nested loop structure
      if (out_idx == kInPerOut - 1) {
        out[out_addr] = elem_out;
        out_addr++;
        out_idx = 0;
      }
      out_idx++;
    }
  } else if (InWidth == OutWidth) {
    assert(OutWidth % InWidth == 0);
    ap_uint<InWidth> elem_in = 0;
    for (int i = 0; i < num_in_words; ++i) {
#pragma HLS PIPELINE II=1
      elem_in = in[i];
#if USE_FIX
      out[i].range() = elem_in;
#else
      out[i] = elem_in;
#endif
    }
  }
}

template <unsigned InWidth, unsigned OutWidth, unsigned NumInWords>
void Mem2MemDataWidthConverter(const ap_uint<InWidth> *in_dmem,
    ap_uint<OutWidth> *out_dmem) {
  if (InWidth > OutWidth) {
    // Emit multiple output words per input word read
    assert(InWidth % OutWidth == 0);
    const unsigned int kOutPerIn = InWidth / OutWidth;
    unsigned int in_idx = 0;
    unsigned int out_idx = 0;
    ap_uint<InWidth> elem_in = 0;
    for (int i = 0; i < NumInWords; ++i) {
#pragma HLS PIPELINE II=1
      if (out_idx == 0) {
        elem_in = in_dmem[in_idx];
        ++in_idx;
      }
      ap_uint<OutWidth> elem_out = elem_in.range(OutWidth - 1, 0);
      out_dmem[i] = elem_out;
      elem_in = elem_in >> OutWidth;
      out_idx++;
      // Wraparound indices to recreate the nested loop structure
      if (out_idx == kOutPerIn) {
        out_idx = 0;
      }
    }
  } else if (InWidth == OutWidth) {
    for (int i = 0; i < NumInWords; ++i) {
#pragma HLS PIPELINE II=1
      out_dmem[i] = in_dmem[i];
    }
  } else { // InWidth < OutWidth
    // Read multiple input words per output word emitted
    assert(OutWidth % InWidth == 0);
    const unsigned int kInPerOut = OutWidth / InWidth;
    const unsigned int kTotalIters = NumInWords;
    unsigned int in_idx = 0;
    unsigned int out_idx = 0;
    ap_uint<OutWidth> elem_out = 0;
    for (int i = 0; i < kTotalIters; i++) {
#pragma HLS PIPELINE II=1
      auto elem_in = in_dmem[i];
      elem_out = elem_out >> InWidth;
      elem_out.range(OutWidth - 1, OutWidth - InWidth) = elem_in;
      in_idx++;
      // Wraparound logic to recreate nested loop functionality
      if (in_idx == kInPerOut) {
        in_idx = 0;
        out_dmem[out_idx] = elem_out;
        ++out_idx;
      }
    }
  }
}

template <int InWidth, int OutWidth>
void Mem2MemDataWidthConverter(const int num_in_words,
    const ap_uint<InWidth> *in_dmem, ap_uint<OutWidth> *out_dmem) {
  if (InWidth > OutWidth) {
    // Emit multiple output words per input word read
    assert(InWidth % OutWidth == 0);
    const unsigned int kOutPerIn = InWidth / OutWidth;
    ap_uint<InWidth> elem_in = 0;
    // =========================================================================
    // A NESTED FOR LOOP IS REQUIRED OTHERWISE THERE WOULD BE NO ITERATIONS WHEN
    // NumInWords IS EQUAL TO 1!
    // =========================================================================
    for (int i = 0; i < num_in_words; ++i) {
      for (int j = 0; j < kOutPerIn; ++j) {
#pragma HLS PIPELINE II=1
        if (j == 0) {
          elem_in = in_dmem[i];
        }
        ap_uint<OutWidth> elem_out = elem_in.range(OutWidth - 1, 0);
        out_dmem[i] = elem_out;
        elem_in = elem_in >> OutWidth;
      }
    }
  } else if (InWidth == OutWidth) {
    for (int i = 0; i < num_in_words; ++i) {
#pragma HLS PIPELINE II=1
      out_dmem[i] = in_dmem[i];
    }
  } else { // InWidth < OutWidth
    // Read multiple input words per output word emitted
    assert(OutWidth % InWidth == 0);
    const unsigned int kInPerOut = OutWidth / InWidth;
    const unsigned int kTotalIters = num_in_words;
    unsigned int in_idx = 0;
    unsigned int out_idx = 0;
    ap_uint<OutWidth> elem_out = 0;
    for (int i = 0; i < kTotalIters; i++) {
#pragma HLS PIPELINE II=1
      auto elem_in = in_dmem[i];
      elem_out = elem_out >> InWidth;
      elem_out.range(OutWidth - 1, OutWidth - InWidth) = elem_in;
      in_idx++;
      // Wraparound logic to recreate nested loop functionality
      if (in_idx == kInPerOut) {
        in_idx = 0;
        out_dmem[out_idx] = elem_out;
        ++out_idx;
      }
    }
  }
}

#endif // end DMA_WIDTH_CONVERTER_H_