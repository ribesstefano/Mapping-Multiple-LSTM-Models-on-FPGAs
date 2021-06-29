#ifndef MATH_UTILS_DATA_HANDLER_H_
#define MATH_UTILS_DATA_HANDLER_H_

#include "hls_utils/hls_metaprogramming.h"

#include "ap_int.h"

#include <vector>

namespace svd {

template<typename FloatType, typename FixType, int NumTiles = 1>
class VectorBlob {
private:
  typedef ap_uint<hls_utils::log2<(NumTiles > 1) ? NumTiles : 2>::value> IdxType;
  int num_tile_elems_;
  int size_;
  int pruned_size_;
  int total_size_;
  int pruned_total_size_;
  int refinement_steps_;
  int num_tiles_;
  int num_zero_tiles_;
  std::vector<FloatType> data_;
  std::vector<FloatType> pruned_data_;
  std::vector<FixType> fix_data_;
  std::vector<FixType> fix_pruned_data_;
  std::vector<int> z_idx_;
  std::vector<int> nz_idx_;
  std::vector<IdxType> z_fix_idx_;
  std::vector<IdxType> nz_fix_idx_;

public:
  VectorBlob(const int refinement_steps, const int vector_size,
      const int num_tiles, const int num_zero_tiles) {
    this->num_tile_elems_ = vector_size / num_tiles;
    this->size_ = vector_size;
    this->pruned_size_ = this->num_tile_elems_ * (num_tiles - num_zero_tiles);
    this->total_size_ = refinement_steps * this->size_;
    this->pruned_total_size_ = refinement_steps * this->pruned_size_;
    this->refinement_steps_ = refinement_steps;
    this->num_tiles_ = num_tiles;
    this->num_zero_tiles_ = num_zero_tiles;
    // NOTE: If num_tiles matches the number of tiles specified in the template,
    // then populate the fix vectors.
    if (num_tiles == NumTiles) {
      for (int i = 0; i < refinement_steps; ++i) {
        this->nz_fix_idx_.push_back(~IdxType(0));
        this->z_fix_idx_.push_back(~IdxType(0));
      }
    }
    if (num_zero_tiles > 0) {
      for (int i = 0; i < refinement_steps; ++i) {
        std::vector<int> rand_idx;
        for (int j = 0; j < num_tiles; ++j) {
          rand_idx.push_back(1);
        }
        for (int j = 0; j < num_zero_tiles; ++j) {
          rand_idx[j] = 0;
        }
        std::random_shuffle(rand_idx.begin(), rand_idx.end());
        // Set the bits
        for (int j = 0; j < num_tiles; ++j) {
          if (num_tiles == NumTiles) {
            this->nz_fix_idx_[i][j] = rand_idx[j];
            this->z_fix_idx_[i][j] = rand_idx[j] == 0 ? 1 : 0;
          }
          if (rand_idx[j] == 0) {
            for (int k = 0; k < this->num_tile_elems_; ++k) {
              this->data_.push_back(0);
              this->fix_data_.push_back(0);
            }
            this->z_idx_.push_back(j);
          } else {
            for (int k = 0; k < this->num_tile_elems_; ++k) {
              FloatType tmp = rand();
              this->data_.push_back(tmp);
              this->pruned_data_.push_back(tmp);
              this->fix_data_.push_back(FixType(tmp));
              this->fix_pruned_data_.push_back(FixType(tmp));
            }
            this->nz_idx_.push_back(j);
          }
        }
      }
    } else {
      for (int i = 0; i < this->total_size_; ++i) {
        FloatType tmp = rand();
        this->data_.push_back(tmp);
        this->pruned_data_.push_back(tmp);
      }
    }
  }
  ~VectorBlob() {};

  FloatType* data() {
    return this->data_.data();
  }

  FloatType* pruned_data() {
    return this->pruned_data_.data();
  }

  FixType* fix_data() {
    return this->fix_data_.data();
  }

  FixType* fix_pruned_data() {
    return this->fix_pruned_data_.data();
  }

  int get_total_size() {
    return this->total_size_;
  }

  int get_pruned_total_size() {
    return this->total_pruned_size_;
  }

  int get_size() {
    return this->size_;
  }

  int get_pruned_size() {
    return this->pruned_size_;
  }

  int* get_z_idx() {
    return this->z_idx_.data();
  }

  int get_z_idx(const int i) {
    return this->z_idx_[i];
  }

  int* get_nz_idx() {
    return this->nz_idx_.data();
  }

  int get_nz_idx(const int i) {
    return this->nz_idx_[i];
  }

  IdxType* get_z_fix_idx() {
    return this->z_fix_idx_.data();
  }

  IdxType get_z_fix_idx(const int refinement_step) {
    return this->z_fix_idx_[refinement_step];
  }

  IdxType* get_nz_fix_idx() {
    return this->nz_fix_idx_.data();
  }

  IdxType get_nz_fix_idx(const int refinement_step) {
    return this->nz_fix_idx_[refinement_step];
  }
};


template <typename FloatType, typename FixType, int NumTilesU = 1, int NumTilesV = 1>
class SvdComponents {
private:
  int num_inputs_;
  VectorBlob<FloatType, FixType, NumTilesU> u_;
  std::vector<VectorBlob<FloatType, FixType, 1> > s_;
  VectorBlob<FloatType, FixType, NumTilesV> v_;
public:
  SvdComponents(const int num_inputs, const int refinement_steps,
      const int u_size, const int v_size, const int num_tiles_u,
      const int num_zero_tiles_u, const int num_tiles_v,
      const int num_zero_tiles_v) {
    this->num_inputs_ = num_inputs;
    this->u_ = VectorBlob<FloatType, FixType, NumTilesU>(refinement_steps, u_size, num_tiles_u, num_zero_tiles_u);
    this->v_ = VectorBlob<FloatType, FixType, NumTilesV>(refinement_steps, v_size, num_tiles_v, num_zero_tiles_v);
    for (int i = 0; i < num_inputs; ++i) {
      this->s_.push_back(VectorBlob<FloatType, FixType, 1>(refinement_steps, 1, 1, 0));
    }
  }

  ~SvdComponents() {};
  
  VectorBlob<FloatType, FixType, NumTilesU> get_u() {
    return this->u_;
  }

  VectorBlob<FloatType, FixType, NumTilesV> get_v() {
    return this->v_;
  }

  std::vector<VectorBlob<FloatType, FixType, 1> > get_s() {
    return this->s_;
  }

  VectorBlob<FloatType, FixType, 1> get_s(const int i) {
    return this->s_[i];
  }

  int get_num_inputs() {
    return this->num_inputs_;
  }
};

} // svd

#endif // MATH_UTILS_DATA_HANDLER_H_