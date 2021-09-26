#ifndef LAYERS_LSTM_DATA_HANDLER_H_
#define LAYERS_LSTM_DATA_HANDLER_H_

#include "math_utils/data_handler.h"
#include "hls_utils/hls_metaprogramming.h"

#include "ap_int.h"

#include <unordered_map>
#include <cstring>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>

namespace svd {

template <typename Tin, typename Tout>
void ArrangeWeights(const int arrange_type,
                    const int n_steps,
                    const int input_size, 
                    const Tin* i_gate,
                    const Tin* f_gate,
                    const Tin* c_gate,
                    const Tin* o_gate,
                    Tout* y) {
  int idx = 0;
  switch (arrange_type) {
    case 0:
      // NOTE: the following arrangement is: (N, G, E)
      for (int i = 0; i < n_steps; ++i) {
        for (int j = 0; j < input_size; ++j) {
          y[idx] = i_gate[i * input_size + j];
          idx++;
        }
        for (int j = 0; j < input_size; ++j) {
          y[idx] = f_gate[i * input_size + j];
          idx++;
        }
        for (int j = 0; j < input_size; ++j) {
          y[idx] = c_gate[i * input_size + j];
          idx++;
        }
        for (int j = 0; j < input_size; ++j) {
          y[idx] = o_gate[i * input_size + j];
          idx++;
        }
      }
    break;
    case 1:
      // NOTE: the following arrangement is: (G, N, E)
      for (int i = 0; i < n_steps; ++i) {
        for (int j = 0; j < input_size; ++j) {
          y[idx] = i_gate[i * input_size + j];
          idx++;
        }
      }
      for (int i = 0; i < n_steps; ++i) {
        for (int j = 0; j < input_size; ++j) {
          y[idx] = f_gate[i * input_size + j];
          idx++;
        }
      }
      for (int i = 0; i < n_steps; ++i) {
        for (int j = 0; j < input_size; ++j) {
          y[idx] = c_gate[i * input_size + j];
          idx++;
        }
      }
      for (int i = 0; i < n_steps; ++i) {
        for (int j = 0; j < input_size; ++j) {
          y[idx] = o_gate[i * input_size + j];
          idx++;
        }
      }
    break;
    case 2:
      // NOTE: the following arrangement is: (N, E, G)
      for (int i = 0; i < n_steps; ++i) {
        for (int j = 0; j < input_size; ++j) {
          y[idx] = i_gate[i * input_size + j];
          idx++;
          y[idx] = f_gate[i * input_size + j];
          idx++;
          y[idx] = c_gate[i * input_size + j];
          idx++;
          y[idx] = o_gate[i * input_size + j];
          idx++;
        }
      }
    break;
    default:
      // NOTE: the following arrangement is: (N, G, E)
      for (int i = 0; i < n_steps; ++i) {
        for (int j = 0; j < input_size; ++j) {
          y[idx] = i_gate[i * input_size + j];
          idx++;
        }
        for (int j = 0; j < input_size; ++j) {
          y[idx] = f_gate[i * input_size + j];
          idx++;
        }
        for (int j = 0; j < input_size; ++j) {
          y[idx] = c_gate[i * input_size + j];
          idx++;
        }
        for (int j = 0; j < input_size; ++j) {
          y[idx] = o_gate[i * input_size + j];
          idx++;
        }
      }
    break;
  }
}

template <typename Tin, typename Tout>
void ArrangeWeights(const int arrange_type, const int r_steps,
                    const int input_size, const int output_size, 
                    const Tin* cur_i, const Tin* cur_f, const Tin* cur_c,
                    const Tin* cur_o, const Tin* rec_i, const Tin* rec_f,
                    const Tin* rec_c, const Tin* rec_o, Tout* y) {
  std::cout << "ArrangeWeights for NZ." << std::endl;
  int idx = 0;
  switch (arrange_type) {
    case 0:
      std::cout << "RGE" << std::endl;
      // NOTE: the following arrangement is: (R, G, E)
      for (int i = 0; i < r_steps; ++i) {
        for (int j = 0; j < input_size; ++j) {
          y[idx] = cur_i[i * input_size + j];
          std::cout << idx << " ";
          idx++;
        }
        for (int j = 0; j < input_size; ++j) {
          y[idx] = cur_f[i * input_size + j];
          idx++;
        }
        for (int j = 0; j < input_size; ++j) {
          y[idx] = cur_c[i * input_size + j];
          idx++;
        }
        for (int j = 0; j < input_size; ++j) {
          y[idx] = cur_o[i * input_size + j];
          idx++;
        }
        for (int j = 0; j < output_size; ++j) {
          y[idx] = rec_i[i * output_size + j];
          idx++;
        }
        for (int j = 0; j < output_size; ++j) {
          y[idx] = rec_f[i * output_size + j];
          idx++;
        }
        for (int j = 0; j < output_size; ++j) {
          y[idx] = rec_c[i * output_size + j];
          idx++;
        }
        for (int j = 0; j < output_size; ++j) {
          y[idx] = rec_o[i * output_size + j];
          idx++;
        }
      }
      std::cout << std::endl;
    break;
    case 1:
      // NOTE: the following arrangement is: (G, R, E)
      for (int i = 0; i < r_steps; ++i) {
        for (int j = 0; j < input_size; ++j) {
          y[idx] = cur_i[i * input_size + j];
          idx++;
        }
      }
      for (int i = 0; i < r_steps; ++i) {
        for (int j = 0; j < input_size; ++j) {
          y[idx] = cur_f[i * input_size + j];
          idx++;
        }
      }
      for (int i = 0; i < r_steps; ++i) {
        for (int j = 0; j < input_size; ++j) {
          y[idx] = cur_c[i * input_size + j];
          idx++;
        }
      }
      for (int i = 0; i < r_steps; ++i) {
        for (int j = 0; j < input_size; ++j) {
          y[idx] = cur_o[i * input_size + j];
          idx++;
        }
      }
      for (int i = 0; i < r_steps; ++i) {
        for (int j = 0; j < output_size; ++j) {
          y[idx] = rec_i[i * output_size + j];
          idx++;
        }
      }
      for (int i = 0; i < r_steps; ++i) {
        for (int j = 0; j < output_size; ++j) {
          y[idx] = rec_f[i * output_size + j];
          idx++;
        }
      }
      for (int i = 0; i < r_steps; ++i) {
        for (int j = 0; j < output_size; ++j) {
          y[idx] = rec_c[i * output_size + j];
          idx++;
        }
      }
      for (int i = 0; i < r_steps; ++i) {
        for (int j = 0; j < output_size; ++j) {
          y[idx] = rec_o[i * output_size + j];
          idx++;
        }
      }
    break;
    case 2:
      assert(input_size == output_size);
      // NOTE: the following arrangement is: (R, E, G)
      for (int i = 0; i < r_steps; ++i) {
        for (int j = 0; j < input_size; ++j) {
          y[idx] = cur_i[i * input_size + j];
          idx++;
          y[idx] = cur_f[i * input_size + j];
          idx++;
          y[idx] = cur_c[i * input_size + j];
          idx++;
          y[idx] = cur_o[i * input_size + j];
          idx++;
          y[idx] = rec_i[i * output_size + j];
          idx++;
          y[idx] = rec_f[i * output_size + j];
          idx++;
          y[idx] = rec_c[i * output_size + j];
          idx++;
          y[idx] = rec_o[i * output_size + j];
          idx++;
        }
      }
    break;
    default:
      // NOTE: the following arrangement is: (R, G, E)
      for (int i = 0; i < r_steps; ++i) {
        for (int j = 0; j < input_size; ++j) {
          y[idx] = cur_i[i * input_size + j];
          idx++;
        }
        for (int j = 0; j < input_size; ++j) {
          y[idx] = cur_f[i * input_size + j];
          idx++;
        }
        for (int j = 0; j < input_size; ++j) {
          y[idx] = cur_c[i * input_size + j];
          idx++;
        }
        for (int j = 0; j < input_size; ++j) {
          y[idx] = cur_o[i * input_size + j];
          idx++;
        }
        for (int j = 0; j < output_size; ++j) {
          y[idx] = rec_i[i * output_size + j];
          idx++;
        }
        for (int j = 0; j < output_size; ++j) {
          y[idx] = rec_f[i * output_size + j];
          idx++;
        }
        for (int j = 0; j < output_size; ++j) {
          y[idx] = rec_c[i * output_size + j];
          idx++;
        }
        for (int j = 0; j < output_size; ++j) {
          y[idx] = rec_o[i * output_size + j];
          idx++;
        }
      }
    break;
  }
}

template <typename FloatType, typename FixType, int NumTilesU = 1, int NumTilesV = 1>
class AcceleratorBlob {
private:
  typedef svd::SvdComponents<FloatType, FixType, NumTilesU, NumTilesV> SvdVecType;
  int lstm_num_inputs_;
  int lstm_input_size_;
  int lstm_output_size_;
  int u_cur_size_;
  int u_rec_size_;
  int v_size_;
  int s_size_;
  std::unordered_map<std::string, SvdVecType*> cur_gates_;
  std::unordered_map<std::string, SvdVecType*> rec_gates_;
  FixType* fix_u_cur_;
  FixType* fix_u_rec_;
  FixType* fix_v_;
  std::vector<std::vector<FloatType> > x_;
  std::vector<std::vector<FloatType> > h_;
  std::vector<std::vector<FloatType> > c_;
  std::vector<std::vector<FloatType> > h_prev_;
  std::vector<std::vector<FloatType> > c_prev_;
  std::vector<std::vector<FloatType> > h_curr_;
  std::vector<std::vector<FloatType> > c_curr_;
  std::vector<std::vector<FloatType> > bias_;
  std::vector<FixType*> fix_x_;
  std::vector<FixType*> fix_h_;
  std::vector<FixType*> fix_c_;
  std::vector<FixType*> fix_h_prev_;
  std::vector<FixType*> fix_c_prev_;
  std::vector<FixType*> fix_h_curr_;
  std::vector<FixType*> fix_c_curr_;
  std::vector<FixType*> fix_bias_;
  std::vector<FixType*> fix_s_;
  ap_uint<NumTilesU>* fix_nz_u_;
  ap_uint<NumTilesV>* fix_nz_v_;

  void InitVector(const bool init_random, const int num_inputs, const int size,
      std::vector<FixType*>& fix_y, std::vector<std::vector<FloatType> >& y) {
    for (int i = 0; i < num_inputs; ++i) {
      fix_y[i] = svd::AllocateContiguously<FixType>(size);
      for (int j = 0; j < size; ++j) {
        FloatType tmp = init_random ? 0.00001 * rand() : 0;
        y[i][j] = tmp;
        fix_y[i][j] = FixType(tmp);
      }
    }
  }

  void ArrangeWeights(const int arrange_type, const int n_steps,
                        std::unordered_map<std::string, SvdVecType*>& gates,
                        FixType* y) {
    int idx = 0;
    switch (arrange_type) {
      case 0:
        // NOTE: the following arrangement is: (N, G, E)
        for (int i = 0; i < n_steps; ++i) {
          for (auto g : gates) {
            for (int j = 0; j < gates.get_u_pruned_size(); ++j) {
              y[idx] = g.second->get_u()->fix_pruned_data()[j];
              ++idx;
            }
          }
        }
      break;
      case 1:
        // NOTE: the following arrangement is: (G, N, E)
        for (auto g : gates) {
          for (int i = 0; i < n_steps; ++i) {
            for (int j = 0; j < gates.get_u_pruned_size(); ++j) {
              y[idx] = g.second->get_u()->fix_pruned_data()[j];
              ++idx;
            }
          }
        }
      break;
      case 2:
        // NOTE: the following arrangement is: (N, E, G)
        for (int i = 0; i < n_steps; ++i) {
          for (int j = 0; j < gates.get_u_pruned_size(); ++j) {
            for (auto g : gates) {
              y[idx] = g.second->get_u()->fix_pruned_data()[j];
              ++idx;
            }
          }
        }
      break;
      default:
        // NOTE: the following arrangement is: (N, G, E)
        for (int i = 0; i < n_steps; ++i) {
          for (auto g : gates) {
            for (int j = 0; j < gates.get_u_pruned_size(); ++j) {
              y[idx] = g.second->get_u()->fix_pruned_data()[j];
              ++idx;
            }
          }
        }
      break;
    }
  }

public:
  AcceleratorBlob(const int num_inputs, const int refinement_steps,
      const int u_cur_size, const int u_rec_size, const int v_size,
      const int num_tiles_u, const int num_zero_tiles_u, const int num_tiles_v,
      const int num_zero_tiles_v) {
    srand(time(NULL));
    this->lstm_num_inputs_ = num_inputs;
    this->lstm_input_size_ = u_cur_size;
    this->lstm_output_size_ = v_size;
    // NOTE: The following instantiation order is important and must be that.
    this->cur_gates_["o"] = new SvdVecType(num_inputs, refinement_steps, u_cur_size, v_size, num_tiles_u, num_zero_tiles_u, num_tiles_v, num_zero_tiles_v);
    this->cur_gates_["c"] = new SvdVecType(num_inputs, refinement_steps, u_cur_size, v_size, num_tiles_u, num_zero_tiles_u, num_tiles_v, num_zero_tiles_v);
    this->cur_gates_["f"] = new SvdVecType(num_inputs, refinement_steps, u_cur_size, v_size, num_tiles_u, num_zero_tiles_u, num_tiles_v, num_zero_tiles_v);
    this->cur_gates_["i"] = new SvdVecType(num_inputs, refinement_steps, u_cur_size, v_size, num_tiles_u, num_zero_tiles_u, num_tiles_v, num_zero_tiles_v);
    this->rec_gates_["o"] = new SvdVecType(num_inputs, refinement_steps, u_rec_size, v_size, num_tiles_u, num_zero_tiles_u, num_tiles_v, num_zero_tiles_v);
    this->rec_gates_["c"] = new SvdVecType(num_inputs, refinement_steps, u_rec_size, v_size, num_tiles_u, num_zero_tiles_u, num_tiles_v, num_zero_tiles_v);
    this->rec_gates_["f"] = new SvdVecType(num_inputs, refinement_steps, u_rec_size, v_size, num_tiles_u, num_zero_tiles_u, num_tiles_v, num_zero_tiles_v);
    this->rec_gates_["i"] = new SvdVecType(num_inputs, refinement_steps, u_rec_size, v_size, num_tiles_u, num_zero_tiles_u, num_tiles_v, num_zero_tiles_v);
    const int kNumGates = 8;
    const int kU_CurTotalSize = kNumGates / 2 * this->cur_gates_["i"]->get_u()->get_pruned_total_size();
    const int kU_RecTotalSize = kNumGates / 2 * this->rec_gates_["i"]->get_u()->get_pruned_total_size();
    const int kV_TotalSize = kNumGates * this->cur_gates_["i"]->get_v()->get_pruned_total_size();
    const int kS_TotalSize = kNumGates * refinement_steps;
    std::cout << "setting fix_u_cur_" << std::endl;
    this->fix_u_cur_ = svd::AllocateContiguously<FixType>(kU_CurTotalSize);
    this->fix_u_rec_ = svd::AllocateContiguously<FixType>(kU_RecTotalSize);
    this->fix_v_ = svd::AllocateContiguously<FixType>(kV_TotalSize);
    this->u_cur_size_ = kU_CurTotalSize;
    this->u_rec_size_ = kU_RecTotalSize;
    this->v_size_ = kV_TotalSize;
    this->s_size_ = kS_TotalSize;
    this->fix_nz_u_ = svd::AllocateContiguously<ap_uint<NumTilesU> >(kS_TotalSize);
    this->fix_nz_v_ = svd::AllocateContiguously<ap_uint<NumTilesV> >(kS_TotalSize);
    std::cout << "kS_TotalSize: " << kS_TotalSize << std::endl;
    std::cout << "kS_TotalSize / 8: " << kS_TotalSize / 8 << std::endl;
    // NOTE: the following arrangement is: (R, E, G)
    const int kArrangementTypeREG = 2;
    const int kArrangementTypeRGE = 0;
    const int kU_CurLengthPruned = this->cur_gates_["i"]->get_u()->get_pruned_size();
    const int kU_RecLengthPruned = this->rec_gates_["i"]->get_u()->get_pruned_size();
    const int kV_LengthPruned = this->cur_gates_["i"]->get_v()->get_pruned_size();
    std::cout << "setting ArrangeWeights" << std::endl;
    svd::ArrangeWeights(kArrangementTypeREG, refinement_steps, kU_CurLengthPruned,
      this->cur_gates_["i"]->get_u()->fix_pruned_data(),
      this->cur_gates_["f"]->get_u()->fix_pruned_data(),
      this->cur_gates_["c"]->get_u()->fix_pruned_data(),
      this->cur_gates_["o"]->get_u()->fix_pruned_data(),
      this->fix_u_cur_);
    std::cout << "setting ArrangeWeights" << std::endl;
    svd::ArrangeWeights(kArrangementTypeREG, refinement_steps, kU_RecLengthPruned,
      this->rec_gates_["i"]->get_u()->fix_pruned_data(),
      this->rec_gates_["f"]->get_u()->fix_pruned_data(),
      this->rec_gates_["c"]->get_u()->fix_pruned_data(),
      this->rec_gates_["o"]->get_u()->fix_pruned_data(),
      this->fix_u_rec_);
    std::cout << "setting ArrangeWeights S" << std::endl;
    svd::ArrangeWeights(kArrangementTypeREG, refinement_steps, kV_LengthPruned,
      kV_LengthPruned,
      this->cur_gates_["i"]->get_v()->fix_pruned_data(),
      this->cur_gates_["f"]->get_v()->fix_pruned_data(),
      this->cur_gates_["c"]->get_v()->fix_pruned_data(),
      this->cur_gates_["o"]->get_v()->fix_pruned_data(),
      this->rec_gates_["i"]->get_v()->fix_pruned_data(),
      this->rec_gates_["f"]->get_v()->fix_pruned_data(),
      this->rec_gates_["c"]->get_v()->fix_pruned_data(),
      this->rec_gates_["o"]->get_v()->fix_pruned_data(),
      this->fix_v_);
    std::cout << "setting ArrangeWeights NZu" << std::endl;
    svd::ArrangeWeights(kArrangementTypeRGE, refinement_steps, 1, 1,
      this->cur_gates_["i"]->get_u()->get_fix_nz_idx(),
      this->cur_gates_["f"]->get_u()->get_fix_nz_idx(),
      this->cur_gates_["c"]->get_u()->get_fix_nz_idx(),
      this->cur_gates_["o"]->get_u()->get_fix_nz_idx(),
      this->rec_gates_["i"]->get_u()->get_fix_nz_idx(),
      this->rec_gates_["f"]->get_u()->get_fix_nz_idx(),
      this->rec_gates_["c"]->get_u()->get_fix_nz_idx(),
      this->rec_gates_["o"]->get_u()->get_fix_nz_idx(),
      this->fix_nz_u_);
    std::cout << "setting ArrangeWeights NZv" << std::endl;
    svd::ArrangeWeights(kArrangementTypeRGE, refinement_steps, 1, 1,
      this->cur_gates_["i"]->get_v()->get_fix_nz_idx(),
      this->cur_gates_["f"]->get_v()->get_fix_nz_idx(),
      this->cur_gates_["c"]->get_v()->get_fix_nz_idx(),
      this->cur_gates_["o"]->get_v()->get_fix_nz_idx(),
      this->rec_gates_["i"]->get_v()->get_fix_nz_idx(),
      this->rec_gates_["f"]->get_v()->get_fix_nz_idx(),
      this->rec_gates_["c"]->get_v()->get_fix_nz_idx(),
      this->rec_gates_["o"]->get_v()->get_fix_nz_idx(),
      this->fix_nz_v_);
    this->fix_x_.resize(num_inputs);
    this->fix_h_.resize(num_inputs);
    this->fix_c_.resize(num_inputs);
    this->fix_h_curr_.resize(num_inputs);
    this->fix_c_curr_.resize(num_inputs);
    this->fix_h_prev_.resize(num_inputs);
    this->fix_c_prev_.resize(num_inputs);
    this->fix_bias_.resize(num_inputs);
    this->x_.resize(num_inputs, std::vector<FloatType>(this->lstm_input_size_));
    this->h_.resize(num_inputs, std::vector<FloatType>(this->lstm_output_size_));
    this->c_.resize(num_inputs, std::vector<FloatType>(this->lstm_output_size_));
    this->h_curr_.resize(num_inputs, std::vector<FloatType>(this->lstm_output_size_));
    this->c_curr_.resize(num_inputs, std::vector<FloatType>(this->lstm_output_size_));
    this->h_prev_.resize(num_inputs, std::vector<FloatType>(this->lstm_output_size_));
    this->c_prev_.resize(num_inputs, std::vector<FloatType>(this->lstm_output_size_));
    this->bias_.resize(num_inputs, std::vector<FloatType>(kNumGates / 2 * this->lstm_output_size_));
    const bool init_random = true;
    this->InitVector(init_random, num_inputs, this->lstm_input_size_, this->fix_x_, this->x_);
    this->InitVector(!init_random, num_inputs, this->lstm_output_size_, this->fix_h_, this->h_);
    this->InitVector(!init_random, num_inputs, this->lstm_output_size_, this->fix_c_, this->c_);
    this->InitVector(!init_random, num_inputs, this->lstm_output_size_, this->fix_h_curr_, this->h_curr_);
    this->InitVector(!init_random, num_inputs, this->lstm_output_size_, this->fix_c_curr_, this->c_curr_);
    this->InitVector(!init_random, num_inputs, this->lstm_output_size_, this->fix_h_prev_, this->h_prev_);
    this->InitVector(!init_random, num_inputs, this->lstm_output_size_, this->fix_c_prev_, this->c_prev_);
    this->InitVector(init_random, num_inputs, kNumGates / 2 * this->lstm_output_size_, this->fix_bias_, this->bias_);
    for (int i = 0; i < num_inputs; ++i) {
      this->fix_s_.push_back(svd::AllocateContiguously<FixType>(kS_TotalSize));
    }
    int idx = 0;
    for (int i = 0; i < num_inputs; ++i) {
      for (int j = 0; j < refinement_steps; ++j) {
        for (auto g : this->cur_gates_) {
          this->fix_s_[i][idx] = g.second->get_s(i).fix_data()[j];
          ++idx;
        }
        for (auto g : this->rec_gates_) {
          this->fix_s_[i][idx] = g.second->get_s(i).fix_data()[j];
          ++idx;
        }
      }
    }
    std::cout << "allocation done." << std::endl;
  }

  ~AcceleratorBlob() {
    svd::FreeContiguously(this->fix_u_cur_);
    svd::FreeContiguously(this->fix_u_rec_);
    svd::FreeContiguously(this->fix_v_);
    for (int i = 0; i < this->lstm_num_inputs_; ++i) {
      svd::FreeContiguously(this->fix_s_[i]);
      svd::FreeContiguously(this->fix_x_[i]);
      svd::FreeContiguously(this->fix_h_[i]);
      svd::FreeContiguously(this->fix_c_[i]);
      svd::FreeContiguously(this->fix_h_curr_[i]);
      svd::FreeContiguously(this->fix_c_curr_[i]);
      svd::FreeContiguously(this->fix_h_prev_[i]);
      svd::FreeContiguously(this->fix_c_prev_[i]);
      svd::FreeContiguously(this->fix_bias_[i]);
    }
    svd::FreeContiguously(this->fix_nz_u_);
    svd::FreeContiguously(this->fix_nz_v_);
    for (auto g : this->cur_gates_) {
      delete g.second;
    }
    for (auto g : this->rec_gates_) {
      delete g.second;
    }
  }

  void ResetLstmOutputs() {
    for (int i = 0; i < this->lstm_num_inputs_; ++i) {
      for (int j = 0; j < this->lstm_output_size_; ++j) {
        this->h_[i][j] = 0;
        this->c_[i][j] = 0;
        this->h_curr_[i][j] = 0;
        this->c_curr_[i][j] = 0;
        this->h_prev_[i][j] = 0;
        this->c_prev_[i][j] = 0;
        this->fix_h_[i][j] = 0;
        this->fix_c_[i][j] = 0;
        this->fix_h_curr_[i][j] = 0;
        this->fix_c_curr_[i][j] = 0;
        this->fix_h_prev_[i][j] = 0;
        this->fix_c_prev_[i][j] = 0;
      }
    }
  }

  int get_lstm_input_size() {
    return this->lstm_input_size_;
  }

  int get_lstm_output_size() {
    return this->lstm_output_size_;
  }


  FixType* get_fix_u_cur() {
    return this->fix_u_cur_;
  }

  FixType* get_fix_u_rec() {
    return this->fix_u_rec_;
  }

  FixType* get_fix_v() {
    return this->fix_v_;
  }

  FixType* get_fix_s(const int i) {
    return this->fix_s_[i];
  }

  std::unordered_map<std::string, SvdVecType*> get_cur_gates() {
    return this->cur_gates_;
  }

  std::unordered_map<std::string, SvdVecType*> get_rec_gates() {
    return this->rec_gates_;
  }

  SvdVecType* get_cur_gates(const std::string g) {
    return this->cur_gates_[g];
  }

  SvdVecType* get_rec_gates(const std::string g) {
    return this->rec_gates_[g];
  }

  FixType* get_fix_x(const int i) {
    return this->fix_x_[i];
  }

  FixType* get_fix_h(const int i) {
    return this->fix_h_[i];
  }

  FloatType* get_h(const int i) {
    return this->h_[i].data();
  }

  FixType* get_fix_h_curr(const int i) {
    return this->fix_h_curr_[i];
  }

  FixType* get_fix_h_prev(const int i) {
    return this->fix_h_prev_[i];
  }

  FixType* get_fix_c(const int i) {
    return this->fix_c_[i];
  }

  FixType* get_fix_c_curr(const int i) {
    return this->fix_c_curr_[i];
  }

  FixType* get_fix_c_prev(const int i) {
    return this->fix_c_prev_[i];
  }

  FixType* get_fix_bias(const int i) {
    return this->fix_bias_[i];
  }

  FloatType* get_bias(const int i) {
    return this->bias_[i].data();
  }

  ap_uint<NumTilesU>* get_fix_nz_u() {
    return this->fix_nz_u_;
  }

  ap_uint<NumTilesV>* get_fix_nz_v() {
    return this->fix_nz_v_;
  }

  int get_u_cur_size() {
    return this->u_cur_size_;
  }

  int get_u_rec_size() {
    return this->u_rec_size_;
  }

  int get_v_size() {
    return this->v_size_;
  }

  int get_s_size() {
    return this->s_size_;
  }

  FloatType* get_x(const int i) {
    return this->x_[i].data();
  }

  int CountMismatches(FixType** x, const int verbose = 0) {
    int num_errors = 0;
    for (int i = 0; i < this->lstm_num_inputs_; ++i) {
      for (int j = 0; j < this->lstm_output_size_; ++j) {
        if (verbose > 0) {
          std:: cout << j << ") hls/emulator: " << this->fix_h_curr_[i][j] << " / " << x[i][j];
        }
        if (this->fix_h_curr_[i][j] != x[i][j]) {
          ++num_errors;
          if (verbose > 0) {
            std:: cout << " <-- ERROR" << std::endl;
          }
        } else {
          if (verbose > 0) {
          std:: cout << std::endl;
        }
        }
      }
    }
    return num_errors;
  }

};

} // svd

#endif // end LAYERS_LSTM_DATA_HANDLER_H_