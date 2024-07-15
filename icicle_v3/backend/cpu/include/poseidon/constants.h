#pragma once

#include <cstdint>

namespace poseidon {
#define FIRST_FULL_ROUNDS  true
#define SECOND_FULL_ROUNDS false

  /**
   * For most of the Poseidon configurations this is the case
   * TODO: Add support for different full rounds numbers
   */
  const int FULL_ROUNDS_DEFAULT = 4;

  /**
   * @struct PoseidonConstants
   * This constants are enough to define a Poseidon instantce
   * @param round_constants A pointer to round constants allocated on the device
   * @param mds_matrix A pointer to an mds matrix allocated on the device
   * @param non_sparse_matrix A pointer to non sparse matrix allocated on the device
   * @param sparse_matrices A pointer to sparse matrices allocated on the device
   */
  template <typename S>
  struct PoseidonConstants {
    unsigned int arity;
    unsigned int alpha;
    unsigned int partial_rounds;
    unsigned int full_rounds_half;
    S* round_constants = nullptr;
    S* mds_matrix = nullptr;
    S* non_sparse_matrix = nullptr;
    S* sparse_matrices = nullptr;
    S domain_tag = S::zero();

    PoseidonConstants() = default;
    PoseidonConstants(const PoseidonConstants& other) = default;

    PoseidonConstants<S>& operator=(PoseidonConstants<S> const& other)
    {
      this->arity = other.arity;
      this->alpha = other.alpha;
      this->partial_rounds = other.partial_rounds;
      this->full_rounds_half = other.full_rounds_half;
      this->round_constants = other.round_constants;
      this->mds_matrix = other.mds_matrix;
      this->non_sparse_matrix = other.non_sparse_matrix;
      this->sparse_matrices = other.sparse_matrices;
      this->domain_tag = other.domain_tag;

      return *this;
    }
  };

  // /**
  //  * @class PoseidonKernelsConfiguration
  //  * Describes the logic of deriving CUDA kernels parameters
  //  * such as the number of threads and the number of blocks
  //  */
  // class PoseidonKernelsConfiguration
  // {
  // public:
  //   // The logic behind this is that 1 thread only works on 1 element
  //   // We have {width} elements in each state, and {number_of_states} states total
  //   static int number_of_threads(unsigned int width) { return 256 / width * width; }

  //   // The partial rounds operates on the whole state, so we define
  //   // the parallelism params for processing a single hash preimage per thread
  //   static const int singlehash_block_size = 128;

  //   static int hashes_per_block(unsigned int width) { return number_of_threads(width) / width; }

  //   static int number_of_full_blocks(unsigned int width, size_t number_of_states)
  //   {
  //     int total_number_of_threads = number_of_states * width;
  //     return total_number_of_threads / number_of_threads(width) +
  //            static_cast<bool>(total_number_of_threads % number_of_threads(width));
  //   }

  //   static int number_of_singlehash_blocks(size_t number_of_states)
  //   {
  //     return number_of_states / singlehash_block_size + static_cast<bool>(number_of_states % singlehash_block_size);
  //   }
  // };

  // using PKC = PoseidonKernelsConfiguration;

  template <typename S>
  IcicleError_t create_optimized_poseidon_constants(
    unsigned int arity,
    unsigned int alpha,
    unsigned int partial_rounds,
    unsigned int full_rounds_half,
    const S* round_constants,
    const S* mds_matrix,
    const S* non_sparse_matrix,
    const S* sparse_matrices,
    const S domain_tag,
    PoseidonConstants<S>* poseidon_constants,
    device_context::DeviceContext& ctx);

  /**
   * Loads pre-calculated optimized constants, moves them to the device
   */
  template <typename S>
  IcicleError_t
  init_optimized_poseidon_constants(int arity, device_context::DeviceContext& ctx, PoseidonConstants<S>* constants);

  template <typename S>
  IcicleError_t release_optimized_poseidon_constants(PoseidonConstants<S>* constants, device_context::DeviceContext& ctx);
} // namespace poseidon

#endif