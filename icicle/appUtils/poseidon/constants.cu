#include "poseidon.cuh"

namespace poseidon {
  int partial_rounds_number_from_arity(const int arity)
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

  template <typename S>
  S* precalculated_optimized_constants(const int arity)
  {
    unsigned char* constants;
    switch (arity) {
    case 2:
      constants = poseidon_constants_2;
      break;
    case 4:
      constants = poseidon_constants_4;
      break;
    case 8:
      constants = poseidon_constants_8;
      break;
    case 11:
      constants = poseidon_constants_11;
      break;
    default:
      throw std::invalid_argument("unsupported arity");
    }
    return reinterpret_cast<S*>(constants);
  }

  template <typename S>
  cudaError_t create_optimized_poseidon_constants(
    int arity,
    int full_rounds_half,
    int partial_rounds,
    const S* constants,
    device_context::DeviceContext& ctx,
    PoseidonConstants<S>* poseidon_constants)
  {
    cudaStream_t& stream = ctx.stream;
    int width = arity + 1;
    int round_constants_len = width * full_rounds_half * 2 + partial_rounds;
    int mds_matrix_len = width * width;
    int sparse_matrices_len = (width * 2 - 1) * partial_rounds;
    int constants_len = round_constants_len + mds_matrix_len * 2 + sparse_matrices_len;

    // Malloc memory for copying constants
    S* d_constants;
    CHK_IF_RETURN(cudaMallocAsync(&d_constants, sizeof(S) * constants_len, stream));

    // Copy constants
    CHK_IF_RETURN(cudaMemcpyAsync(d_constants, constants, sizeof(S) * constants_len, cudaMemcpyHostToDevice, stream));

    S* round_constants = d_constants;
    S* mds_matrix = round_constants + round_constants_len;
    S* non_sparse_matrix = mds_matrix + mds_matrix_len;
    S* sparse_matrices = non_sparse_matrix + mds_matrix_len;

    // Pick the domain_tag accordinaly
    // For now, we only support Merkle tree mode
    uint32_t tree_domain_tag_value = 1;
    tree_domain_tag_value = (tree_domain_tag_value << (width - 1)) - tree_domain_tag_value;
    S domain_tag = S::from(tree_domain_tag_value);

    // Make sure all the constants have been copied
    CHK_IF_RETURN(cudaStreamSynchronize(stream));
    *poseidon_constants = {arity,      partial_rounds,    full_rounds_half, round_constants,
                           mds_matrix, non_sparse_matrix, sparse_matrices,  domain_tag};

    return CHK_LAST();
  }

  template <typename S>
  cudaError_t
  init_optimized_poseidon_constants(int arity, device_context::DeviceContext& ctx, PoseidonConstants<S>* constants)
  {
    CHK_INIT_IF_RETURN();
    int width = arity + 1;
    int full_rounds_half = FULL_ROUNDS_DEFAULT;
    int partial_rounds = partial_rounds_number_from_arity(arity);

    // All the constants are stored in a single file
    S* h_constants = precalculated_optimized_constants<S>(width - 1);

    create_optimized_poseidon_constants(arity, full_rounds_half, partial_rounds, h_constants, ctx, constants);

    return CHK_LAST();
  }

  extern "C" cudaError_t CONCAT_EXPAND(CURVE, CreateOptimizedPoseidonConstants)(
    int arity,
    int full_rounds_half,
    int partial_rounds,
    const curve_config::scalar_t* constants,
    device_context::DeviceContext& ctx,
    PoseidonConstants<curve_config::scalar_t>* poseidon_constants)
  {
    return create_optimized_poseidon_constants<curve_config::scalar_t>(
      arity, full_rounds_half, partial_rounds, constants, ctx, poseidon_constants);
  }

  extern "C" cudaError_t CONCAT_EXPAND(CURVE, InitOptimizedPoseidonConstants)(
    int arity, device_context::DeviceContext& ctx, PoseidonConstants<curve_config::scalar_t>* constants)
  {
    return init_optimized_poseidon_constants<curve_config::scalar_t>(arity, ctx, constants);
  }
} // namespace poseidon