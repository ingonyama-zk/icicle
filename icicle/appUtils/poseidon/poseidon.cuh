#pragma once
#ifndef POSEIDON_H
#define POSEIDON_H

#include <cstdint>
#include <stdexcept>

namespace poseidon {
  #define FIRST_FULL_ROUNDS true
  #define SECOND_FULL_ROUNDS false

  uint32_t partial_rounds_number_from_arity(const uint32_t arity)
  {
    switch (arity) {
    case 2:
      return 55;
    case 4:
      return 56;
    case 8:
      return 57;
    case 11:
      return 57;
    default:
      throw std::invalid_argument("unsupported arity");
    }
  };

  // TO-DO: change to mapping
  const uint32_t FULL_ROUNDS_DEFAULT = 4;

  template <typename S>
  struct PoseidonConfiguration {
    uint32_t partial_rounds, full_rounds_half, t;
    S *round_constants, *mds_matrix, *non_sparse_matrix, *sparse_matrices;
  };

  /// This class describes the logic of calculating CUDA kernels parameters
  /// such as the number of threads and the number of blocks
  class ParallelPoseidonConfiguration
  {
    uint32_t t;
  public:
    int number_of_threads, hashes_per_block, singlehash_block_size;

    ParallelPoseidonConfiguration(const uint32_t t) {
      this->t = t;
      // The logic behind this is that 1 thread only works on 1 element
      // We have {t} elements in each state, and {number_of_states} states total
      number_of_threads = (256 / t) * t;
      hashes_per_block = number_of_threads / t;
      this->t = t;

      // The partial rounds operates on the whole state, so we define
      // the parallelism params for processing a single hash preimage per thread
      singlehash_block_size = 128;
    }

    int number_of_full_blocks(size_t number_of_states) {
      int total_number_of_threads = number_of_states * t;
      return total_number_of_threads / number_of_threads + static_cast<bool>(total_number_of_threads % number_of_threads);
    }

    int number_of_singlehash_blocks(size_t number_of_states) {
      return number_of_states / singlehash_block_size + static_cast<bool>(number_of_states % singlehash_block_size);
    }
  };

  /// Interface class
  template <typename S>
  class Poseidon
  {
  public:
    enum HashType {
      ConstInputLen,
      MerkleTree,
    };
    uint32_t t, arity;

    Poseidon(uint32_t arity) {
      this->arity = arity;
      this->t = arity + 1;
    }

    /// This function will apply a single Poseidon permutation to mulitple states in parallel 
    virtual void permute_many(S * states, size_t number_of_states, cudaStream_t stream) {}

    /// This function will copy input from host and copy the result from device
    void hash_blocks(const S * inp, size_t number_of_states, S * out, HashType hash_type, cudaStream_t stream) {
        S * states, * out_device;
        // allocate memory for {number_of_states} states of {t} scalars each
        if (cudaMallocAsync(&states, number_of_states * t * sizeof(S), stream) != cudaSuccess) {
            throw std::runtime_error("Failed memory allocation on the device");
        }
        if (cudaMallocAsync(&out_device, number_of_states * sizeof(S), stream) != cudaSuccess) {
            throw std::runtime_error("Failed memory allocation on the device");
        }

        // This is where the input matrix of size Arity x NumberOfBlocks is
        // padded and coppied to device in a T x NumberOfBlocks matrix
        cudaMemcpy2DAsync(states, t * sizeof(S),  // Device pointer and device pitch
                      inp, (t - 1) * sizeof(S),    // Host pointer and pitch
                      (t - 1) * sizeof(S), number_of_states, // Size of the source matrix (Arity x NumberOfBlocks)
                      cudaMemcpyHostToDevice, stream);

        poseidon_hash(states, number_of_states, out_device, hash_type, stream, false, false);

        cudaFreeAsync(states, stream);
        cudaMemcpyAsync(out, out_device, number_of_states * sizeof(S), cudaMemcpyDeviceToHost, stream);
        cudaFreeAsync(out_device, stream);
    }

    // Compute the poseidon hash over a sequence of preimages
    ///
    ///=====================================================
    /// # Arguments
    /// * `states`  - a device pointer to the states memory. Expected to be of size `number_of_states * t` elements. States should contain the leaves values
    /// * `number_of_states`  - number of preimages number_of_states. Each block is of size t
    /// * `out` - a device pointer to the digests memory. Expected to be of size `sum(arity ^ (i)) for i in [0..height-1]`
    /// * `hash_type`  - this will determine the domain_tag value
    /// * `stream` - a cuda stream to run the kernels
    /// * `aligned` - if set to `true`, the algorithm expects the states to contain leaves in an aligned form
    /// * `loop_results` - if set to `true`, the resulting hash will be also copied into the states memory in aligned form.
    ///
    /// Aligned form (for arity = 2):
    /// [0, X1, X2, 0, X3, X4, ...]
    ///
    /// Not aligned form (for arity = 2) (you will get this format
    ///                                   after copying leaves with cudaMemcpy2D):
    /// [X1, X2, 0, X3, X4, 0]
    /// Note: elements denoted by 0 doesn't need to be set to 0, the algorithm
    /// will replace them with domain tags.
    ///
    /// # Algorithm
    /// The function will split large trees into many subtrees of size that will fit `STREAM_CHUNK_SIZE`.
    /// The subtrees will be constructed in streams pool. Each stream will handle a subtree
    /// After all subtrees are constructed - the function will combine the resulting sub-digests into the final top-tree
    ///======================================================
    void poseidon_hash(S * states, size_t number_of_states, S * out, Poseidon<S>::HashType hash_type, cudaStream_t stream, bool aligned, bool loop_results) {
      // Pick the domain_tag accordinaly
      S domain_tag;
      switch (hash_type) {
      case HashType::ConstInputLen:
        // Temporary solution
        domain_tag = S::zero();
        break;

      case HashType::MerkleTree:
        uint32_t tree_domain_tag_value = 1;
        tree_domain_tag_value = (tree_domain_tag_value << arity) - tree_domain_tag_value;
        domain_tag = S::from(tree_domain_tag_value);
      }

      prepare_states(states, number_of_states, domain_tag, aligned);

      permute_many(states, number_of_states, stream);

      process_results(states, number_of_states, out, loop_results);
    }

  private:
    virtual void prepare_states(S * states, size_t number_of_states, S domain_tag, bool aligned) {}
    virtual void process_results(S * states, size_t number_of_states, S * out, bool loop_results) {}
  };
} // namespace poseidon

#endif