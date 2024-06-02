/**
 * Warning!
 * Contents of this file are deprecated and will be removed in the next minor version
 */

#include "fields/field_config.cuh"

using namespace field_config;

#include "poseidon/poseidon.cuh"

#include "gpu-utils/device_context.cuh"
#include "utils/utils.h"

namespace poseidon {
  /**
   * @struct PoseidonConfig
   * Struct that encodes various Poseidon parameters.
   */
  struct PoseidonConfig {
    device_context::DeviceContext ctx; /**< Details related to the device such as its id and stream id. */
    bool are_inputs_on_device;  /**< True if inputs are on device and false if they're on host. Default value: false. */
    bool are_outputs_on_device; /**< If true, output is preserved on device, otherwise on host. Default value: false. */
    bool input_is_a_state;      /**< If true, input is considered to be a states vector, holding the preimages
                                 * in aligned or not aligned format. Memory under the input pointer will be used for states
                                 * If false, fresh states memory will be allocated and input will be copied into it */
    bool aligned;               /**< If true - input should be already aligned for poseidon permutation.
                                 * Aligned format: [0, A, B, 0, C, D, ...] (as you might get by using loop_state)
                                 * not aligned format: [A, B, 0, C, D, 0, ...] (as you might get from cudaMemcpy2D) */
    bool loop_state;            /**< If true, hash results will also be copied in the input pointer in aligned format */
    bool is_async; /**< Whether to run the Poseidon asynchronously. If set to `true`, the poseidon_hash function will be
                    *   non-blocking and you'd need to synchronize it explicitly by running
                    *   `cudaStreamSynchronize` or `cudaDeviceSynchronize`. If set to false, the poseidon_hash
                    *   function will block the current CPU thread. */
  };

  /**
   * Extern "C" version of [poseidon_hash_cuda] function with the following
   * value of template parameter (where the field is given by `-DFIELD` env variable during build):
   *  - `S` is the [field](@ref scalar_t) - either a scalar field of the elliptic curve or a
   * stand-alone "STARK field";
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, poseidon_hash_cuda)(
    scalar_t* input,
    scalar_t* output,
    int number_of_states,
    int arity,
    const PoseidonConstants<scalar_t>& constants,
    PoseidonConfig& config)
  {
    Poseidon<scalar_t> poseidon(arity, config.ctx);
    SpongeConfig cfg{config.ctx, config.are_inputs_on_device, config.are_outputs_on_device, config.is_async};
    return poseidon.hash_many(input, output, number_of_states, arity, 1, cfg);
  }

  extern "C" cudaError_t CONCAT_EXPAND(FIELD, create_optimized_poseidon_constants_cuda)(
    int arity,
    int full_rounds_half,
    int partial_rounds,
    const scalar_t* constants,
    device_context::DeviceContext& ctx,
    PoseidonConstants<scalar_t>* poseidon_constants)
  {
    unsigned int width = arity + 1;
    unsigned int round_constants_len = width * full_rounds_half * 2 + partial_rounds;
    unsigned int mds_matrix_len = width * width;

    const scalar_t* round_constants = constants;
    const scalar_t* mds_matrix = round_constants + round_constants_len;
    const scalar_t* non_sparse_matrix = mds_matrix + mds_matrix_len;
    const scalar_t* sparse_matrices = non_sparse_matrix + mds_matrix_len;

    uint32_t tree_domain_tag_value = 1;
    tree_domain_tag_value = (tree_domain_tag_value << arity) - tree_domain_tag_value;
    scalar_t domain_tag = scalar_t::from(tree_domain_tag_value);
    return create_optimized_poseidon_constants<scalar_t>(
      arity, 5, partial_rounds, full_rounds_half, round_constants, mds_matrix, non_sparse_matrix, sparse_matrices,
      domain_tag, poseidon_constants, ctx);
  }

  extern "C" cudaError_t CONCAT_EXPAND(FIELD, init_optimized_poseidon_constants_cuda)(
    int arity, device_context::DeviceContext& ctx, PoseidonConstants<scalar_t>* constants)
  {
    return init_optimized_poseidon_constants<scalar_t>(arity, ctx, constants);
  }
} // namespace poseidon