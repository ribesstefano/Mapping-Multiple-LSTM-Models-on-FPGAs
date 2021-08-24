#ifndef SVD_PARAMS_H_
#define SVD_PARAMS_H_

#include "hls_utils/hls_metaprogramming.h"
#include "dma/axis_lib.h"

#include "ap_int.h"
#include "ap_axi_sdata.h"
#include "hls_stream.h"
#ifdef __VITIS_HLS__
#include "hls_vector.h"
#endif

#include <cstdint>

namespace svd {

template<typename Type, int Ni, int Ii, int Tui, int ZTui = 0, int Gi = 1>
struct ParamsU {
  static const int N = Ni;
  static const int I = Ii;
  static const int Tu = Tui;
  static const int ZTu = ZTui;
  static const int G = Gi;
  static const int TuElems = I / Tu;
  static const int TuBits = hlsutils::log2<Tu>::value > 0 ? hlsutils::log2<Tu>::value : 1;
  typedef ap_uint<Tu> UnzD;
  typedef ap_uint<TuBits> UnzIdxD;
  typedef Type ActivationD;
  typedef Type WeightD;
  typedef Type AccumulationD;
  typedef hls::stream<UnzD> UnzS;
  typedef hls::stream<ap_uint<TuBits> > UnzIdxS;
  typedef hls::stream<ActivationD> ActivationS;
  typedef hls::stream<WeightD> WeightS;
  typedef hls::stream<AccumulationD> AccumulationS;
  typedef ap_uint<hlsutils::Bitwidth<WeightD>::value * G> UPortD;
  static const int PrunedSizeU = I / Tu * (Tu - ZTu);
  static const int ActivationWidth = hlsutils::Bitwidth<ActivationD>::value;
  static const int WeightWidth = hlsutils::Bitwidth<WeightD>::value;
  static const int AccumulationWidth = hlsutils::Bitwidth<AccumulationD>::value;
  static const int VectTuAxiWidth = ActivationWidth * Tu;
  static const int VectN_AxiWidth = ActivationWidth * N;
  static const int VectG_AxiWidth = ActivationWidth * G;
  static const int VectGN_AxiWidth = ActivationWidth * G * N;
  typedef typename svd::AxiStreamPort<VectTuAxiWidth>::AxiuPacketType VectTuAxiPacketType;
  typedef typename svd::AxiStreamPort<VectN_AxiWidth>::AxiuPacketType VectN_AxiPacketType;
  typedef typename svd::AxiStreamPort<VectG_AxiWidth>::AxiuPacketType VectG_AxiPacketType;
  typedef typename svd::AxiStreamPort<VectGN_AxiWidth>::AxiuPacketType VectGN_AxiPacketType;
  typedef typename svd::AxiStreamWrapper<VectTuAxiWidth>::AxiuType VectTuAxiuType;
  typedef typename svd::AxiStreamWrapper<VectN_AxiWidth>::AxiuType VectN_AxiuType;
  typedef typename svd::AxiStreamWrapper<VectG_AxiWidth>::AxiuType VectG_AxiuType;
  typedef typename svd::AxiStreamWrapper<VectGN_AxiWidth>::AxiuType VectGN_AxiuType;
#ifdef __VITIS_HLS__
  typedef hls::vector<ActivationD, Tu> VectTuType;
  typedef hls::vector<ActivationD, N> VectN_Type;
  typedef hls::vector<ActivationD, G> VectG_Type;
  typedef hls::vector<ActivationD, G * N> VectGN_Type;
#endif
};

template <int Ni, int Ii, int Hi, int Ri, int Tui, int Tvi, int ZTui = 0,
  int ZTvi = 0, int Gi = 1,
  typename ActivationD_tp = ap_fixed<16, 3>,
  typename WeightD_tp = ap_fixed<8, 3>,
  typename AccumulationD_tp = ap_fixed<16, 3> >
struct SvdParameters {
  static const int N = Ni;
  static const int I = Ii;
  static const int H = Hi;
  static const int R = Ri;
  static const int Tu = Tui;
  static const int ZTu = ZTui;
  static const int Tv = Tvi;
  static const int ZTv = ZTvi;
  static const int G = Gi;
  static const int PeU = Tu - ZTu;
  static const int PeV = H / Tv;
  static const int TuElems = I / Tu;
  static const int TvElems = H / Tv;
  static const int TuBits = hlsutils::log2<Tu>::value > 0 ? hlsutils::log2<Tu>::value : 1;
  static const int TvBits = hlsutils::log2<Tv>::value > 0 ? hlsutils::log2<Tv>::value : 1;
  typedef ap_uint<Tu> UnzD;
  typedef ap_uint<Tv> VnzD;
  typedef ap_uint<TuBits> UnzIdxD;
  typedef ap_uint<TvBits> VnzIdxD;
  typedef ActivationD_tp ActivationD;
  typedef WeightD_tp WeightD;
  typedef AccumulationD_tp AccumulationD;
  typedef hls::stream<UnzD> UnzS;
  typedef hls::stream<VnzD> VnzS;
  typedef hls::stream<ap_uint<TuBits> > UnzIdxS;
  typedef hls::stream<ap_uint<TvBits> > VnzIdxS;
  typedef hls::stream<ActivationD> ActivationS;
  typedef hls::stream<WeightD> WeightS;
  typedef hls::stream<AccumulationD> AccumulationS;
  typedef ap_uint<hlsutils::Bitwidth<WeightD>::value * G> SPortD;
  typedef ap_uint<hlsutils::Bitwidth<WeightD>::value * G> UPortD;
  typedef ap_uint<hlsutils::Bitwidth<WeightD>::value * G> VPortD;
  static const int PrunedSizeU = I / Tu * (Tu - ZTu);
  static const int PrunedSizeV = H / Tv * (Tv - ZTv);
  static const int SizeS = R * G;
  static const int ActivationWidth = hlsutils::Bitwidth<ActivationD>::value;
  static const int WeightWidth = hlsutils::Bitwidth<WeightD>::value;
  static const int AccumulationWidth = hlsutils::Bitwidth<AccumulationD>::value;
  static const int VectTuAxiWidth = ActivationWidth * Tu;
  static const int VectTvAxiWidth = ActivationWidth * Tv;
  static const int VectN_AxiWidth = ActivationWidth * N;
  static const int VectG_AxiWidth = ActivationWidth * G;
  static const int VectGN_AxiWidth = ActivationWidth * G * N;
  static const int VectGTvAxiWidth = ActivationWidth * G * Tv;
  typedef typename svd::AxiStreamPort<VectTuAxiWidth>::AxiuPacketType VectTuAxiPacketType;
  typedef typename svd::AxiStreamPort<VectTvAxiWidth>::AxiuPacketType VectTvAxiPacketType;
  typedef typename svd::AxiStreamPort<VectN_AxiWidth>::AxiuPacketType VectN_AxiPacketType;
  typedef typename svd::AxiStreamPort<VectG_AxiWidth>::AxiuPacketType VectG_AxiPacketType;
  typedef typename svd::AxiStreamPort<VectGN_AxiWidth>::AxiuPacketType VectGN_AxiPacketType;
  typedef typename svd::AxiStreamPort<VectGTvAxiWidth>::AxiuPacketType VectGTvAxiPacketType;
#ifdef __VITIS_HLS__
  typedef hls::vector<ActivationD, Tu> VectTuType;
  typedef hls::vector<ActivationD, Tv> VectTvType;
  typedef hls::vector<ActivationD, N> VectN_Type;
  typedef hls::vector<ActivationD, G> VectG_Type;
  typedef hls::vector<ActivationD, G * N> VectGN_Type;
  typedef hls::vector<ActivationD, G * Tv> VectGTvType;
#endif
};

template<typename params>
class SvdStreams {
public:
  typename params::ActivationS x[params::N][params::G][params::PeU];
  typename params::UnzS nz_u[params::G];
  typename params::VnzS nz_v[params::G];
  typename params::WeightS u[params::G][params::PeU];
  typename params::WeightS v[params::G][params::PeV];
  typename params::WeightS u_dma[params::G];
  typename params::WeightS v_dma[params::G];
  typename params::WeightS s[params::N][params::G];
  // typename params::AccumulationD xu[params::N][params::G][params::R][params::PeU]; // defined as array, but a stream
  typename params::AccumulationS xu[params::N][params::G][params::PeU];
  typename params::AccumulationS xus[params::N][params::G][params::PeV];
  typename params::ActivationS xusv[params::N][params::G][params::PeV];
  typename params::UnzIdxS nz_u_idx[params::G][params::PeU];
  typename params::VnzIdxS nz_v_idx[params::G][params::PeV];
  typename params::UnzIdxS tile_idx_stream[params::N][params::G][params::PeU];

  SvdStreams() {
#pragma HLS STREAM depth=2 variable=this->x
#pragma HLS STREAM depth=2 variable=this->nz_u
#pragma HLS STREAM depth=2 variable=this->nz_v
#pragma HLS STREAM depth=2 variable=this->u
#pragma HLS STREAM depth=2 variable=this->s
#pragma HLS STREAM depth=2 variable=this->v
#pragma HLS STREAM depth=2 variable=this->xu
#pragma HLS STREAM depth=2 variable=this->xus
#pragma HLS STREAM depth=2 variable=this->xusv
#pragma HLS STREAM depth=2 variable=this->nz_u_idx
#pragma HLS STREAM depth=2 variable=this->nz_v_idx
#pragma HLS STREAM depth=2 variable=this->u_dma
#pragma HLS STREAM depth=2 variable=this->v_dma
#pragma HLS STREAM depth=2 variable=this->tile_idx_stream
#pragma HLS ARRAY_PARTITION complete dim=0 variable=this->x
#pragma HLS ARRAY_PARTITION complete dim=0 variable=this->nz_u
#pragma HLS ARRAY_PARTITION complete dim=0 variable=this->nz_v
#pragma HLS ARRAY_PARTITION complete dim=0 variable=this->u
#pragma HLS ARRAY_PARTITION complete dim=0 variable=this->s
#pragma HLS ARRAY_PARTITION complete dim=0 variable=this->v
#pragma HLS ARRAY_PARTITION complete dim=0 variable=this->u_dma
#pragma HLS ARRAY_PARTITION complete dim=0 variable=this->v_dma
#pragma HLS ARRAY_PARTITION complete dim=0 variable=this->xu
#pragma HLS ARRAY_PARTITION complete dim=0 variable=this->xus
#pragma HLS ARRAY_PARTITION complete dim=0 variable=this->xusv
#pragma HLS ARRAY_PARTITION complete dim=0 variable=this->nz_u_idx
#pragma HLS ARRAY_PARTITION complete dim=0 variable=this->nz_v_idx
#pragma HLS ARRAY_PARTITION variable=this->tile_idx_stream complete dim=0
  };
  ~SvdStreams() {};

private: // TODO: move parameters' access in members?
};

template<typename params>
class SvdBuffers {
public:
  typename params::ActivationD x[params::N][params::Tu][params::I / params::Tu];

  SvdBuffers() {
#pragma HLS ARRAY_PARTITION variable=x complete dim=1
#pragma HLS ARRAY_PARTITION variable=x complete dim=2
#pragma HLS RESOURCE variable=x core=RAM_2P_BRAM
  };
  ~SvdBuffers() {};

private: // TODO: move parameters' access in members?
};

#ifndef NUM_INPUTS
#define NUM_INPUTS 2
#endif

/**
 * Fashion-MNIST dataset LSTM-network:
 *  cur_gate.shape: (256, 128)
 *  rec_gate.shape: (128, 128)
 *  
 * MSCOCO dataset LSTM-network
 *  cur_gate.shape: (1024, 512)
 *  rec_gate.shape: (512, 512)
 */
#ifndef INPUT_SIZE
#define INPUT_SIZE 1024
#endif

#ifndef HIDDEN_SIZE
#define HIDDEN_SIZE 512
#endif

#ifndef NUM_GATES
#define NUM_GATES 4
#endif

#ifndef NUM_SAMPLES
#define NUM_SAMPLES 2
#endif

#ifndef NUM_TILES_U
#define NUM_TILES_U 32
#endif
#ifndef NUM_ZERO_TILES_U
#define NUM_ZERO_TILES_U 16
#endif
#ifndef NUM_TILES_V
#define NUM_TILES_V 64
#endif
#ifndef NUM_ZERO_TILES_V
#define NUM_ZERO_TILES_V 8
#endif
#ifndef TILE_SIZE_CUR_U
#define TILE_SIZE_CUR_U (INPUT_SIZE / NUM_TILES_U)
#endif
#ifndef TILE_SIZE_REC_U
#define TILE_SIZE_REC_U (HIDDEN_SIZE / NUM_TILES_U)
#endif
#ifndef TILE_SIZE_CUR_V
#define TILE_SIZE_CUR_V (HIDDEN_SIZE / NUM_TILES_V)
#endif
#ifndef TILE_SIZE_REC_V
#define TILE_SIZE_REC_V (HIDDEN_SIZE / NUM_TILES_V)
#endif

#ifndef NUM_ITERATIONS
#define NUM_ITERATIONS 16
#endif
#ifndef NUM_TIMESTEPS
#define NUM_TIMESTEPS 28
#endif

#if defined(USE_FIX)
  #define USE_FIX 1
  #define USE_FLOAT 0
  #define USE_DOUBLE 0
#elif defined(USE_FLOAT)
  #define USE_FIX 0
  #define USE_FLOAT 1
  #define USE_DOUBLE 0
#elif defined(USE_DOUBLE)
  #define USE_FIX 0
  #define USE_FLOAT 0
  #define USE_DOUBLE 1
#else
  #define USE_FIX 1 // default option
  #define USE_FLOAT 0
  #define USE_DOUBLE 0
#endif

#ifndef FIX_WIDTH
#define FIX_WIDTH 16
#endif
#ifndef FIX_FRACT_WIDTH
#define FIX_FRACT_WIDTH 7
#endif

#ifndef AXI_PORT_WIDTH
#define AXI_PORT_WIDTH 128
#endif

#if USE_FLOAT
typedef float WeightD;
typedef float ActivationD;
typedef float AccumD;
typedef float MultD;
#elif USE_DOUBLE
typedef double WeightD;
typedef double ActivationD;
typedef double AccumD;
typedef double MultD;
#else // USE_FIX
typedef ap_fixed<FIX_WIDTH, FIX_FRACT_WIDTH, AP_TRN, AP_SAT_SYM> WeightD;
typedef ap_fixed<FIX_WIDTH, FIX_FRACT_WIDTH, AP_TRN, AP_SAT_SYM> ActivationD;

#if FIX_WIDTH == 8
typedef ap_fixed<FIX_WIDTH * 2, FIX_FRACT_WIDTH * 2, AP_TRN, AP_SAT_SYM> AccumD;
typedef ap_fixed<FIX_WIDTH * 2, FIX_FRACT_WIDTH * 2, AP_TRN, AP_SAT_SYM> MultD;
typedef uint8_t AccelD;
#elif FIX_WIDTH == 16
typedef ap_fixed<FIX_WIDTH + 1, FIX_FRACT_WIDTH + 1, AP_TRN, AP_SAT_SYM> AccumD;
typedef ap_fixed<FIX_WIDTH + 1, FIX_FRACT_WIDTH + 1, AP_TRN, AP_SAT_SYM> MultD;
typedef uint16_t AccelD;
#elif FIX_WIDTH == 32
typedef ap_fixed<FIX_WIDTH, FIX_FRACT_WIDTH + 1, AP_TRN, AP_SAT_SYM> AccumD;
typedef ap_fixed<FIX_WIDTH, FIX_FRACT_WIDTH + 1, AP_TRN, AP_SAT_SYM> MultD;
typedef uint32_t AccelD;
#else
typedef ap_fixed<FIX_WIDTH, FIX_FRACT_WIDTH + 1, AP_TRN, AP_SAT_SYM> AccumD;
typedef ap_fixed<FIX_WIDTH, FIX_FRACT_WIDTH + 1, AP_TRN, AP_SAT_SYM> MultD;
typedef uint64_t AccelD;
#endif

#endif

typedef ap_uint<AXI_PORT_WIDTH> AxiD; // 64bit for ZedBoard HP and ACP ports (128bit for ZCU104)

typedef hls::stream<ActivationD> ActivationStream;
typedef hls::stream<WeightD> WeightStream;
typedef hls::stream<AxiD> AxiStream;
typedef hls::stream<AccumD> AccumStream;

// TODO: Remove CounterD and ProbeStream types from here.
// typedef long long CounterD;
// typedef hls::stream<bool> ProbeStream;

typedef svd::SvdParameters<NUM_INPUTS, INPUT_SIZE, HIDDEN_SIZE, NUM_ITERATIONS,
    NUM_TILES_U, NUM_TILES_V, NUM_ZERO_TILES_U, NUM_ZERO_TILES_V, NUM_GATES,
    ActivationD, WeightD, AccumD> svd_params;

} // namespace svd


namespace testsvd {

} // end testsvd

#endif // end SVD_PARAMS_H_