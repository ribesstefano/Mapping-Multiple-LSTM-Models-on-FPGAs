#ifndef MATH_UTILS_DATA_HANDLER_H_
#define MATH_UTILS_DATA_HANDLER_H_

#include "hls_utils/hls_metaprogramming.h"

#include "ap_int.h"

#include <vector>
#include <exception>
#include <new>
#include <iostream>
#include <cassert>
#include <algorithm>

#ifdef SDS_DESIGN
#include <stdint.h>
#include "sds_lib.h"
#else
#include <cstdlib>
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

namespace svd {

template <typename T>
T* AllocateContiguously(const int size) {
#ifndef __SYNTHESIS__
  T* tmp;
  try {
    tmp = (T*)ALLOC(size * sizeof(T));
  } catch(...) {
    std::cout << "[ERROR] Exception occurred while contiguously allocating." << std::endl;
    throw;
  }
  if (!tmp) {
    std::cout << "[ERROR] Contiguous allocation failed." << std::endl;
    std::bad_alloc except_alloc;
    throw except_alloc;
  }
  return tmp;
#else
  T* tmp = (T*)ALLOC(size * sizeof(T));
  if (!tmp) {
    std::cout << "[ERROR] Contiguous allocation failed." << std::endl;
    exit(1);
  }
  return tmp;
#endif
}

template <typename T>
void FreeContiguously(T** x) {
#ifdef SDS_DESIGN
  std::cout << "[INFO] Calling FreeContiguously sds_free()." << std::endl;
#else
  std::cout << "[INFO] Calling FreeContiguously free()." << std::endl;
#endif
  FREE(*x);
  *x = nullptr;
}

template<typename FloatType, typename FixType, int NumTiles = 1>
class VectorBlob {
private:
  typedef ap_uint<NumTiles> IdxType;
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
  std::vector<IdxType> fix_z_idx_;
  std::vector<IdxType> fix_nz_idx_;

public:
  VectorBlob(const int refinement_steps, const int vector_size,
      const int num_tiles, const int num_zero_tiles) {
    assert(refinement_steps > 0);
    assert(vector_size > 0);
    assert(num_tiles > 0);
    assert(vector_size % num_tiles == 0);
    this->num_tile_elems_ = vector_size / num_tiles;
    this->size_ = vector_size;
    this->pruned_size_ = this->num_tile_elems_ * (num_tiles - num_zero_tiles);
    this->total_size_ = refinement_steps * this->size_;
    this->pruned_total_size_ = refinement_steps * this->pruned_size_;
    this->refinement_steps_ = refinement_steps;
    this->num_tiles_ = num_tiles;
    this->num_zero_tiles_ = num_zero_tiles;
    for (int i = 0; i < refinement_steps; ++i) {
      this->fix_nz_idx_.push_back(~IdxType(0));
      this->fix_z_idx_.push_back(~IdxType(0));
      this->nz_idx_.push_back(-1);
      this->z_idx_.push_back(-1);
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
            this->fix_nz_idx_[i][j] = rand_idx[j];
            this->fix_z_idx_[i][j] = rand_idx[j] == 0 ? 1 : 0;
          }
          if (rand_idx[j] == 0) {
            // Pruned tile
            for (int k = 0; k < this->num_tile_elems_; ++k) {
              this->data_.push_back(0);
              this->fix_data_.push_back(0);
            }
            this->z_idx_.push_back(j);
          } else {
            // Non-pruned tile
            for (int k = 0; k < this->num_tile_elems_; ++k) {
              FloatType tmp = 0.00001 * rand();
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
        FloatType tmp = 0.00001 * rand();
        this->data_.push_back(tmp);
        this->pruned_data_.push_back(tmp);
        this->fix_data_.push_back(FixType(tmp));
        this->fix_pruned_data_.push_back(FixType(tmp));
      }
      std::cout << "this->data_.size(): " << this->data_.size() << std::endl;
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
    assert(this->pruned_total_size_ == this->fix_pruned_data_.size());
    return this->pruned_total_size_;
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
    return this->z_idx_.at(i);
  }


  /**
   * @brief      Gets the nz index.
   *
   * @return     The nz index. Shape: (R, NZ-Tiles)
   */
  int* get_nz_idx() {
    return this->nz_idx_.data();
  }

  int get_nz_idx(const int i) {
    return this->nz_idx_.at(i);
  }

  int get_nz_idx(const int r, const int t) {
    return this->nz_idx_.at(r * this->num_tiles_ + t);
  }

  IdxType* get_fix_z_idx() {
    return this->fix_z_idx_.data();
  }

  IdxType get_fix_z_idx(const int refinement_step) {
    return this->fix_z_idx_[refinement_step];
  }

  IdxType* get_fix_nz_idx() {
    return this->fix_nz_idx_.data();
  }

  IdxType get_fix_nz_idx(const int refinement_step) {
    return this->fix_nz_idx_.at(refinement_step);
  }

  int get_refinement_steps() {
    return this->refinement_steps_;
  }
};


template <typename FloatType, typename FixType, int NumTilesU = 1, int NumTilesV = 1>
class SvdComponents {
private:
  int num_inputs_;
  VectorBlob<FloatType, FixType, NumTilesU>* u_;
  std::vector<VectorBlob<FloatType, FixType, 1> > s_;
  VectorBlob<FloatType, FixType, NumTilesV>* v_;
public:
  SvdComponents(const int num_inputs, const int refinement_steps,
      const int u_size, const int v_size, const int num_tiles_u,
      const int num_zero_tiles_u, const int num_tiles_v,
      const int num_zero_tiles_v) {
    assert(num_inputs > 0);
    this->num_inputs_ = num_inputs;
    this->u_ = new VectorBlob<FloatType, FixType, NumTilesU>(refinement_steps,
      u_size, num_tiles_u, num_zero_tiles_u);
    this->v_ = new VectorBlob<FloatType, FixType, NumTilesV>(refinement_steps,
      v_size, num_tiles_v, num_zero_tiles_v);
    for (int i = 0; i < num_inputs; ++i) {
      this->s_.push_back(VectorBlob<FloatType, FixType, 1>(refinement_steps, 1, 1, 0));
    }
  }

  ~SvdComponents() {
    delete this->u_;
    delete this->v_;
  }
  
  VectorBlob<FloatType, FixType, NumTilesU>* get_u() {
    return this->u_;
  }

  VectorBlob<FloatType, FixType, NumTilesV>* get_v() {
    return this->v_;
  }

  int get_u_size() {
    return this->u_->get_size();
  }

  int get_v_size() {
    return this->v_->get_size();
  }

  int get_u_pruned_size() {
    return this->u_->get_pruned_size();
  }

  int get_v_pruned_size() {
    return this->v_->get_pruned_size();
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

  int get_refinement_steps() {
    return this->s_[0].get_refinement_steps();
  }
};

} // svd

#endif // MATH_UTILS_DATA_HANDLER_H_