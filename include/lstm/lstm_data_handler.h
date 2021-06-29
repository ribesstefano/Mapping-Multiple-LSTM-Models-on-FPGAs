#ifndef LSTM_DATA_HANDLER_H_
#define LSTM_DATA_HANDLER_H_

#include "math_utils/data_handler.h"
#include "hls_utils/hls_metaprogramming.h"

#include "ap_int.h"

#ifdef SDS_DESIGN
#include <stdint.h>
#include "sds_lib.h"
#endif

#ifndef ALLOC
# ifdef SDS_DESIGN
#   define ALLOC(x) sds_alloc(x)
# else
#   define ALLOC(x) malloc(x)
# endif
#endif

#ifndef FREE
# ifdef SDS_DESIGN
#   define FREE(x) sds_free(x)
# else
#   define FREE(x) free(x)
# endif
#endif

namespace lstm {

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
void ArrangeWeights(const int arrange_type, const int n_steps,
                    const int input_size, const int output_size, 
                    const Tin* cur_i, const Tin* cur_f, const Tin* cur_c,
                    const Tin* cur_o, const Tin* rec_i, const Tin* rec_f,
                    const Tin* rec_c, const Tin* rec_o, Tout* y) {
  int idx = 0;
  switch (arrange_type) {
    case 0:
      // NOTE: the following arrangement is: (N, G, E)
      for (int i = 0; i < n_steps; ++i) {
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
    case 1:
      // NOTE: the following arrangement is: (G, N, E)
      for (int i = 0; i < n_steps; ++i) {
        for (int j = 0; j < input_size; ++j) {
          y[idx] = cur_i[i * input_size + j];
          idx++;
        }
      }
      for (int i = 0; i < n_steps; ++i) {
        for (int j = 0; j < input_size; ++j) {
          y[idx] = cur_f[i * input_size + j];
          idx++;
        }
      }
      for (int i = 0; i < n_steps; ++i) {
        for (int j = 0; j < input_size; ++j) {
          y[idx] = cur_c[i * input_size + j];
          idx++;
        }
      }
      for (int i = 0; i < n_steps; ++i) {
        for (int j = 0; j < input_size; ++j) {
          y[idx] = cur_o[i * input_size + j];
          idx++;
        }
      }
      for (int i = 0; i < n_steps; ++i) {
        for (int j = 0; j < output_size; ++j) {
          y[idx] = rec_i[i * output_size + j];
          idx++;
        }
      }
      for (int i = 0; i < n_steps; ++i) {
        for (int j = 0; j < output_size; ++j) {
          y[idx] = rec_f[i * output_size + j];
          idx++;
        }
      }
      for (int i = 0; i < n_steps; ++i) {
        for (int j = 0; j < output_size; ++j) {
          y[idx] = rec_c[i * output_size + j];
          idx++;
        }
      }
      for (int i = 0; i < n_steps; ++i) {
        for (int j = 0; j < output_size; ++j) {
          y[idx] = rec_o[i * output_size + j];
          idx++;
        }
      }
    break;
    case 2:
      assert(input_size == output_size);
      // NOTE: the following arrangement is: (N, E, G)
      for (int i = 0; i < n_steps; ++i) {
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
      // NOTE: the following arrangement is: (N, G, E)
      for (int i = 0; i < n_steps; ++i) {
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
class GateBlob {
private:
  int num_gates_;
  svd::SvdComponents<FloatType, FixType, NumTilesU, NumTilesV>* i_;
  svd::SvdComponents<FloatType, FixType, NumTilesU, NumTilesV>* f_;
  svd::SvdComponents<FloatType, FixType, NumTilesU, NumTilesV>* c_;
  svd::SvdComponents<FloatType, FixType, NumTilesU, NumTilesV>* o_;
public:
  GateBlob(const int num_inputs, const int refinement_steps,
      const int u_size, const int v_size, const int num_tiles_u,
      const int num_zero_tiles_u, const int num_tiles_v,
      const int num_zero_tiles_v) {
    this->num_gates_ = 4;
    this->i_ = new svd::SvdComponents<FloatType, FixType, NumTilesU, NumTilesV>(num_inputs, refinement_steps, u_size, v_size, num_tiles_u, num_zero_tiles_u, num_tiles_v, num_zero_tiles_v);
    this->f_ = new svd::SvdComponents<FloatType, FixType, NumTilesU, NumTilesV>(num_inputs, refinement_steps, u_size, v_size, num_tiles_u, num_zero_tiles_u, num_tiles_v, num_zero_tiles_v);
    this->c_ = new svd::SvdComponents<FloatType, FixType, NumTilesU, NumTilesV>(num_inputs, refinement_steps, u_size, v_size, num_tiles_u, num_zero_tiles_u, num_tiles_v, num_zero_tiles_v);
    this->o_ = new svd::SvdComponents<FloatType, FixType, NumTilesU, NumTilesV>(num_inputs, refinement_steps, u_size, v_size, num_tiles_u, num_zero_tiles_u, num_tiles_v, num_zero_tiles_v);
  }

  ~GateBlob() {
    delete[] this->i_;
    delete[] this->f_;
    delete[] this->c_;
    delete[] this->o_;
  }

  svd::SvdComponents<FloatType, FixType, NumTilesU, NumTilesV>* get_i() {
    return this->i_;
  }

  svd::SvdComponents<FloatType, FixType, NumTilesU, NumTilesV>* get_f() {
    return this->f_;
  }

  svd::SvdComponents<FloatType, FixType, NumTilesU, NumTilesV>* get_c() {
    return this->c_;
  }

  svd::SvdComponents<FloatType, FixType, NumTilesU, NumTilesV>* get_o() {
    return this->o_;
  }
};

template <typename FloatType, typename FixType, int NumTilesU = 1, int NumTilesV = 1>
class AcceleratorBlob {
private:
  GateBlob<FloatType, FixType, NumTilesU, NumTilesV>* cur_gates_;
  GateBlob<FloatType, FixType, NumTilesU, NumTilesV>* rec_gates_;
  FixType* fix_u_cur_;
  FixType* fix_u_rec_;
  FixType* fix_v_;
  std::vector<FixType*> fix_s_;
public:
  AcceleratorBlob(const int num_inputs, const int refinement_steps,
      const int u_cur_size, const int u_rec_size, const int v_size,
      const int num_tiles_u, const int num_zero_tiles_u, const int num_tiles_v,
      const int num_zero_tiles_v) {
    this->cur_gates_ = new GateBlob<FloatType, FixType, NumTilesU, NumTilesV>(num_inputs, refinement_steps, u_cur_size, v_size, num_tiles_u, num_zero_tiles_u, num_tiles_v, num_zero_tiles_v);
    this->rec_gates_ = new GateBlob<FloatType, FixType, NumTilesU, NumTilesV>(num_inputs, refinement_steps, u_rec_size, v_size, num_tiles_u, num_zero_tiles_u, num_tiles_v, num_zero_tiles_v);
    const int kNumInputs = this->cur_gates_->get_i()->get_num_inputs();
    const int kNumGates = 8;
    const int kU_CurTotalSize = kNumGates / 2 * this->cur_gates_->get_i()->get_u()->get_pruned_total_size();
    const int kU_RecTotalSize = kNumGates / 2 * this->rec_gates_->get_i()->get_u()->get_pruned_total_size();
    const int kV_TotalSize = kNumGates * this->cur_gates_->get_i()->get_v()->get_pruned_total_size();
    const int kS_TotalSize = kNumGates * refinement_steps;
    for (int i = 0; i < kNumInputs; ++i) {
      this->fix_s_.push_back((FixType*)ALLOC(kS_TotalSize * sizeof(FixType)));
    }
    this->fix_u_cur_ = (FixType*)ALLOC(kU_CurTotalSize * sizeof(FixType));
    this->fix_u_rec_ = (FixType*)ALLOC(kU_RecTotalSize * sizeof(FixType));
    this->fix_v_ = (FixType*)ALLOC(kV_TotalSize * sizeof(FixType));
    // NOTE: the following arrangement is: (R, E, G)
    const int kArrangementType = 2;
    const int kU_CurLengthPruned = this->cur_gates_->get_i()->get_u()->get_pruned_size();
    const int kU_RecLengthPruned = this->rec_gates_->get_i()->get_u()->get_pruned_size();
    const int kV_LengthPruned = this->cur_gates_->get_i()->get_v()->get_pruned_size();
    lstm::ArrangeWeights(kArrangementType, refinement_steps, kU_CurLengthPruned,
      this->cur_gates_->get_i()->get_u()->pruned_data(),
      this->cur_gates_->get_f()->get_u()->pruned_data(),
      this->cur_gates_->get_c()->get_u()->pruned_data(),
      this->cur_gates_->get_o()->get_u()->pruned_data(),
      this->fix_u_cur_);
    lstm::ArrangeWeights(kArrangementType, refinement_steps, kU_RecLengthPruned,
      this->rec_gates_->get_i()->get_u()->pruned_data(),
      this->rec_gates_->get_f()->get_u()->pruned_data(),
      this->rec_gates_->get_c()->get_u()->pruned_data(),
      this->rec_gates_->get_o()->get_u()->pruned_data(),
      this->fix_u_rec_);
    lstm::ArrangeWeights(kArrangementType, refinement_steps, kV_LengthPruned,
      kV_LengthPruned,
      this->cur_gates_->get_i()->get_v()->pruned_data(),
      this->cur_gates_->get_f()->get_v()->pruned_data(),
      this->cur_gates_->get_c()->get_v()->pruned_data(),
      this->cur_gates_->get_o()->get_v()->pruned_data(),
      this->rec_gates_->get_i()->get_v()->pruned_data(),
      this->rec_gates_->get_f()->get_v()->pruned_data(),
      this->rec_gates_->get_c()->get_v()->pruned_data(),
      this->rec_gates_->get_o()->get_v()->pruned_data(),
      this->fix_v_);
    for (int i = 0; i < kNumInputs; ++i) {
      lstm::ArrangeWeights(kArrangementType, refinement_steps, 1, 1,
        this->cur_gates_->get_i()->get_s(i).data(),
        this->cur_gates_->get_f()->get_s(i).data(),
        this->cur_gates_->get_c()->get_s(i).data(),
        this->cur_gates_->get_o()->get_s(i).data(),
        this->rec_gates_->get_i()->get_s(i).data(),
        this->rec_gates_->get_f()->get_s(i).data(),
        this->rec_gates_->get_c()->get_s(i).data(),
        this->rec_gates_->get_o()->get_s(i).data(),
        this->fix_s_[i]);
    }

    // // (sizeof(FixType) * 8) * 4
    // // (sizeof(FixType) * 8) * 8
    // ap_uint<64>* u_cur_uint = reinterpret_cast<ap_uint<64>*>(this->fix_u_cur_);
    // ap_uint<64>* u_rec_uint = reinterpret_cast<ap_uint<64>*>(this->fix_u_rec_);
    // ap_uint<128>* v_uint = reinterpret_cast<ap_uint<128>*>(this->fix_v_);
    // ap_uint<128>* s1_uint = reinterpret_cast<ap_uint<128>*>(this->fix_s_[0]);
    // ap_uint<128>* s2_uint = reinterpret_cast<ap_uint<128>*>(this->fix_s_[1]);

  }

  ~AcceleratorBlob() {
    FREE(this->fix_u_cur_);
    FREE(this->fix_u_rec_);
    FREE(this->fix_v_);
    for (int i = 0; i < this->cur_gates_->get_i()->get_num_inputs(); ++i) {
      FREE(this->fix_s_[i]);
    }
    delete[] this->cur_gates_;
    delete[] this->rec_gates_;
  }

};

} // lstm

#endif // end LSTM_DATA_HANDLER_H_