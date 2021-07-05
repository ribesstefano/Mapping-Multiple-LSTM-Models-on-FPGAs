#ifndef LSTM_HLS_LSTM_SVD_EMULATOR_H_
#define LSTM_HLS_LSTM_SVD_EMULATOR_H_

#include "math_utils/activation_functions.h"

#include "hls_stream.h"

#include <cstdlib>

namespace svd {

/**
 * @brief      Emulator used to test the accuracy of the HLS accelerator. It
 *             allows for testing different design points without recompiling.
 *
 * @param[in]  InputSize     The input size
 * @param[in]  HiddenSize    The hidden size
 * @param[in]  NumIter       The number of refinement steps
 * @param[in]  Tu            The number of tiles of u
 * @param[in]  ZTu           The number of pruned tiles of u
 * @param[in]  Tv            The number of tiles of v
 * @param[in]  ZTv           The number of pruned tiles of v
 * @param[in]  NumTimesteps  The number timesteps (deprecated)
 * @param[in]  x             The input data
 * @param[in]  cur_i_u       The current i u
 * @param[in]  cur_i_s       The current i s
 * @param[in]  cur_i_v       The current i v
 * @param[in]  cur_i_unz     The current i unz
 * @param[in]  cur_i_vnz     The current i vnz
 * @param[in]  cur_f_u       The current f u
 * @param[in]  cur_f_s       The current f s
 * @param[in]  cur_f_v       The current f v
 * @param[in]  cur_f_unz     The current f unz
 * @param[in]  cur_f_vnz     The current f vnz
 * @param[in]  cur_c_u       The current c u
 * @param[in]  cur_c_s       The current c s
 * @param[in]  cur_c_v       The current c v
 * @param[in]  cur_c_unz     The current c unz
 * @param[in]  cur_c_vnz     The current c vnz
 * @param[in]  cur_o_u       The current o u
 * @param[in]  cur_o_s       The current o s
 * @param[in]  cur_o_v       The current o v
 * @param[in]  cur_o_unz     The current o unz
 * @param[in]  cur_o_vnz     The current o vnz
 * @param[in]  rec_i_u       The recurrent i u
 * @param[in]  rec_i_s       The recurrent i s
 * @param[in]  rec_i_v       The recurrent i v
 * @param[in]  rec_i_unz     The recurrent i unz
 * @param[in]  rec_i_vnz     The recurrent i vnz
 * @param[in]  rec_f_u       The recurrent f u
 * @param[in]  rec_f_s       The recurrent f s
 * @param[in]  rec_f_v       The recurrent f v
 * @param[in]  rec_f_unz     The recurrent f unz
 * @param[in]  rec_f_vnz     The recurrent f vnz
 * @param[in]  rec_c_u       The recurrent c u
 * @param[in]  rec_c_s       The recurrent c s
 * @param[in]  rec_c_v       The recurrent c v
 * @param[in]  rec_c_unz     The recurrent c unz
 * @param[in]  rec_c_vnz     The recurrent c vnz
 * @param[in]  rec_o_u       The recurrent o u
 * @param[in]  rec_o_s       The recurrent o s
 * @param[in]  rec_o_v       The recurrent o v
 * @param[in]  rec_o_unz     The recurrent o unz
 * @param[in]  rec_o_vnz     The recurrent o vnz
 * @param[in]  bias          The bias
 * @param[in]  c_prev        The c previous
 * @param[in]  h_prev        The h previous
 * @param      c_curr        The c current
 * @param      h_curr        The h current
 *
 * @tparam     DataA         Activation type
 * @tparam     DataW         Weight type
 * @tparam     DataAcc       Accumulation type
 * @tparam     DataMul       Multiplication type
 * @tparam     TanhLutSize   Size of the hard sigmoid LUT
 */
template <typename DataA,
          typename DataW,
          typename DataAcc,
          typename DataMul,
          int TanhLutSize>
void LstmSvdSoftEmulator(const int InputSize,
                       const int HiddenSize,
                       const int NumIter,
                       const int Tu,
                       const int ZTu,
                       const int Tv,
                       const int ZTv,
                       const int NumTimesteps,
                       const DataA *x,
                       const DataW *cur_i_u,
                       const DataW *cur_i_s,
                       const DataW *cur_i_v,
                       const int *cur_i_unz,
                       const int *cur_i_vnz,
                       const DataW *cur_f_u,
                       const DataW *cur_f_s,
                       const DataW *cur_f_v,
                       const int *cur_f_unz,
                       const int *cur_f_vnz,
                       const DataW *cur_c_u,
                       const DataW *cur_c_s,
                       const DataW *cur_c_v,
                       const int *cur_c_unz,
                       const int *cur_c_vnz,
                       const DataW *cur_o_u,
                       const DataW *cur_o_s,
                       const DataW *cur_o_v,
                       const int *cur_o_unz,
                       const int *cur_o_vnz,
                       const DataW *rec_i_u,
                       const DataW *rec_i_s,
                       const DataW *rec_i_v,
                       const int *rec_i_unz,
                       const int *rec_i_vnz,
                       const DataW *rec_f_u,
                       const DataW *rec_f_s,
                       const DataW *rec_f_v,
                       const int *rec_f_unz,
                       const int *rec_f_vnz,
                       const DataW *rec_c_u,
                       const DataW *rec_c_s,
                       const DataW *rec_c_v,
                       const int *rec_c_unz,
                       const int *rec_c_vnz,
                       const DataW *rec_o_u,
                       const DataW *rec_o_s,
                       const DataW *rec_o_v,
                       const int *rec_o_unz,
                       const int *rec_o_vnz,
                       const DataW *bias,
                       DataA *c_prev,
                       DataA *h_prev,
                       DataA *c_curr,
                       DataA *h_curr) {
  assert(Tu % 2 == 0);
  assert(Tv % 2 == 0);
  assert(Tu >= 8);
  assert(Tv >= 8);
  assert(Tu > ZTu);
  assert(Tv > ZTv);
  assert(NumIter % 2 == 0);
  const DataW *u[8];
  const DataW *s[8];
  const DataW *v[8];
  const int *unz[8];
  const int *vnz[8];
  u[0] = cur_i_u; u[1] = cur_f_u; u[2] = cur_c_u; u[3] = cur_o_u;
  u[4] = rec_i_u; u[5] = rec_f_u; u[6] = rec_c_u; u[7] = rec_o_u; 
  s[0] = cur_i_s; s[1] = cur_f_s; s[2] = cur_c_s; s[3] = cur_o_s;
  s[4] = rec_i_s; s[5] = rec_f_s; s[6] = rec_c_s; s[7] = rec_o_s;
  v[0] = cur_i_v; v[1] = cur_f_v; v[2] = cur_c_v; v[3] = cur_o_v;
  v[4] = rec_i_v; v[5] = rec_f_v; v[6] = rec_c_v; v[7] = rec_o_v;
  unz[0] = cur_i_unz; unz[1] = cur_f_unz; unz[2] = cur_c_unz; unz[3] = cur_o_unz;
  unz[4] = rec_i_unz; unz[5] = rec_f_unz; unz[6] = rec_c_unz; unz[7] = rec_o_unz;
  vnz[0] = cur_i_vnz; vnz[1] = cur_f_vnz; vnz[2] = cur_c_vnz; vnz[3] = cur_o_vnz;
  vnz[4] = rec_i_vnz; vnz[5] = rec_f_vnz; vnz[6] = rec_c_vnz; vnz[7] = rec_o_vnz;
  hls::stream<DataA> **cur_out_fifo = new hls::stream<DataA>*[4];
  hls::stream<DataA> **rec_out_fifo = new hls::stream<DataA>*[4];
  for (int i = 0; i < 4; ++i) {
    cur_out_fifo[i] = new hls::stream<DataA>[Tv];
    rec_out_fifo[i] = new hls::stream<DataA>[Tv];
  }
  DataAcc *u_acc[8];
  DataAcc **acc_buffer[8];
  DataMul xs_val[8] = {0};
  for (int i = 0; i < 8; ++i) {
    u_acc[i] = new DataAcc[NumIter];
  }
  DataA *h[2];
  DataA *c[2];
  if (NumTimesteps > 1) {
    for (int i = 0; i < 2; ++i) {
      h[i] = new DataA[HiddenSize];
      c[i] = new DataA[HiddenSize];
      std::memset(h[i], 0, HiddenSize * sizeof(DataA));
      std::memset(c[i], 0, HiddenSize * sizeof(DataA));
    }
  } else {
    c[0] = c_prev;
    c[1] = c_curr;
    h[0] = h_prev;
    h[1] = h_curr;
  }
  for (int i = 0; i < 8; ++i) { 
    acc_buffer[i] = new DataAcc*[Tv];
    for (int j = 0; j < Tv; ++j) {
      acc_buffer[i][j] = new DataAcc[HiddenSize / Tv];
    }
  }
  for (int t = 0; t < NumTimesteps; ++t) {
    const int in_ptr = (t % 2) == 0 ? 0 : 1;
    const int out_ptr = (t % 2) == 0 ? 1 : 0;
    for (int i = 0; i < 8; ++i) { 
      std::memset(u_acc[i], 0, NumIter * sizeof(DataAcc));
      for (int j = 0; j < Tv; ++j) {
        std::memset(acc_buffer[i][j], 0, HiddenSize / Tv * sizeof(DataAcc));
      }
    }
    for (int i = 0; i < NumIter; ++i) {
      for (int q = 0; q < 4; ++q) {
        for (int j = 0; j < Tu - ZTu; ++j) {
          const int nz_idx = i * (Tu - ZTu) + j;
          for (int k = 0; k < InputSize / Tu; ++k) {
            int u_idx = i * InputSize / Tu * (Tu - ZTu) + j * InputSize / Tu + k;
            u_acc[q][i] += x[t * InputSize + unz[q][nz_idx] * InputSize / Tu + k] * u[q][u_idx];
          }
          for (int k = 0; k < HiddenSize / Tu; ++k) {
            int u_idx = i * HiddenSize / Tu * (Tu - ZTu) + j * HiddenSize / Tu + k;
            u_acc[q + 4][i] += h[in_ptr][unz[q + 4][nz_idx] * HiddenSize / Tu + k] * u[q + 4][u_idx];
          }
        }
      }
      for (int q = 0; q < 8; ++q) {
        xs_val[q] = s[q][i] * DataA(u_acc[q][i]);
        for (int j = 0; j < Tv - ZTv; ++j) {
          for (int k = 0; k < HiddenSize / Tv; ++k) {
            const int v_idx = i * HiddenSize / Tv * (Tv - ZTv) + j * HiddenSize / Tv + k;
            const int nz_idx = i * (Tv - ZTv) + j;
            acc_buffer[q][vnz[q][nz_idx]][k] += xs_val[q] * v[q][v_idx];
          }
        }
      }
    }
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < Tv; ++j) {
        for (int k = 0; k < HiddenSize / Tv; ++k) {
          cur_out_fifo[i][j].write(acc_buffer[i][j][k]);
          rec_out_fifo[i][j].write(acc_buffer[i + 4][j][k]);
        }
      }
    }
    NonLinearityUnitSoftware<DataA, DataW, DataAcc, TanhLutSize>(HiddenSize,
      Tv, 4, c[in_ptr], cur_out_fifo, rec_out_fifo, h[out_ptr], c[out_ptr],
      true, bias);
  }
  if (NumTimesteps > 1) {
    std::memcpy(h_curr, h[(NumTimesteps - 1) % 2 == 0 ? 1 : 0], HiddenSize * sizeof(DataA));
  }
  for (int i = 0; i < 4; ++i) {
    delete[] cur_out_fifo[i];
    delete[] rec_out_fifo[i];
  }
  delete[] cur_out_fifo;
  delete[] rec_out_fifo;
  for (int i = 0; i < 8; ++i) {
    delete[] u_acc[i];
    for (int j = 0; j < Tv; ++j) {
      delete[] acc_buffer[i][j];
    }
    delete[] acc_buffer[i];
  }
  if (NumTimesteps > 1) {
    for (int i = 0; i < 2; ++i) {
      delete[] h[i];
      delete[] c[i];
    }
  }
}

} // svd

#endif // LSTM_HLS_LSTM_SVD_EMULATOR_H_