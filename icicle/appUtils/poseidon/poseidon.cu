#include "poseidon.cuh"
#include "kernels.cu"

namespace poseidon {
  PoseidonConfig default_poseidon_config(int t)
  {
    device_context::DeviceContext ctx = device_context::get_default_device_context();
    PoseidonConfig config = {
      ctx,   // ctx
      false, // are_inputes_on_device
      false, // are_outputs_on_device
      false, // input_is_a_state
      false, // aligned
      false, // loop_state
      false, // is_async
    };
    return config;
  }

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
  S* optimized_constants(const int arity)
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

  template <typename S, int T>
  cudaError_t
  permute_many(S* states, size_t number_of_states, const PoseidonConstants<S>& constants, cudaStream_t& stream)
  {
    size_t rc_offset = 0;

    full_rounds<S, T><<<
      PKC<T>::number_of_full_blocks(number_of_states), PKC<T>::number_of_threads,
      sizeof(S) * PKC<T>::hashes_per_block * T, stream>>>(
      states, number_of_states, rc_offset, FIRST_FULL_ROUNDS, constants);
    rc_offset += T * (constants.full_rounds_half + 1);

    partial_rounds<S, T>
      <<<PKC<T>::number_of_singlehash_blocks(number_of_states), PKC<T>::singlehash_block_size, 0, stream>>>(
        states, number_of_states, rc_offset, constants);
    rc_offset += constants.partial_rounds;

    full_rounds<S, T><<<
      PKC<T>::number_of_full_blocks(number_of_states), PKC<T>::number_of_threads,
      sizeof(S) * PKC<T>::hashes_per_block * T, stream>>>(
      states, number_of_states, rc_offset, SECOND_FULL_ROUNDS, constants);
    return CHK_LAST();
  }

  template <typename S, int T>
  cudaError_t poseidon_hash(
    S* input, S* output, size_t number_of_states, const PoseidonConstants<S>& constants, const PoseidonConfig& config)
  {
    CHK_INIT_IF_RETURN();
    cudaStream_t& stream = config.ctx.stream;
    S* states;
    if (config.input_is_a_state) {
      states = input;
    } else {
      // allocate memory for {number_of_states} states of {t} scalars each
      CHK_IF_RETURN(cudaMallocAsync(&states, number_of_states * T * sizeof(S), stream))

      // This is where the input matrix of size Arity x NumberOfBlocks is
      // padded and copied to device in a T x NumberOfBlocks matrix
      CHK_IF_RETURN(cudaMemcpy2DAsync(
        states, T * sizeof(S),                 // Device pointer and device pitch
        input, (T - 1) * sizeof(S),            // Host pointer and pitch
        (T - 1) * sizeof(S), number_of_states, // Size of the source matrix (Arity x NumberOfBlocks)
        cudaMemcpyHostToDevice, stream));
    }

    S* output_device;
    if (config.are_outputs_on_device) {
      output_device = output;
    } else {
      CHK_IF_RETURN(cudaMallocAsync(&output_device, number_of_states * sizeof(S), stream))
    }

    prepare_poseidon_states<S, T>
      <<<PKC<T>::number_of_full_blocks(number_of_states), PKC<T>::number_of_threads, 0, stream>>>(
        states, number_of_states, constants.domain_tag, config.aligned);

    cudaError_t hash_error = permute_many<S, T>(states, number_of_states, constants, stream);
    CHK_IF_RETURN(hash_error);

    get_hash_results<S, T>
      <<<PKC<T>::number_of_singlehash_blocks(number_of_states), PKC<T>::singlehash_block_size, 0, stream>>>(
        states, number_of_states, output_device);

    if (config.loop_state) {
      copy_recursive<S, T>
        <<<PKC<T>::number_of_singlehash_blocks(number_of_states), PKC<T>::singlehash_block_size, 0, stream>>>(
          states, number_of_states, output_device);
    }

    if (!config.input_is_a_state) CHK_IF_RETURN(cudaFreeAsync(states, stream));

    if (!config.are_outputs_on_device) {
      CHK_IF_RETURN(
        cudaMemcpyAsync(output, output_device, number_of_states * sizeof(S), cudaMemcpyDeviceToHost, stream));
      CHK_IF_RETURN(cudaFreeAsync(output_device, stream));
    }

    if (!config.is_async) return CHK_STICKY(cudaStreamSynchronize(stream));
    return CHK_LAST();
  }

  template <typename S>
  cudaError_t
  init_optimized_poseidon_constants(int arity, device_context::DeviceContext& ctx, PoseidonConstants<S>* constants)
  {
    CHK_INIT_IF_RETURN();
    cudaStream_t& stream = ctx.stream;
    int width = arity + 1;
    int full_rounds_half = FULL_ROUNDS_DEFAULT;
    int partial_rounds = partial_rounds_number_from_arity(arity);

    int round_constants_len = width * full_rounds_half * 2 + partial_rounds;
    int mds_matrix_len = width * width;
    int sparse_matrices_len = (width * 2 - 1) * partial_rounds;

    // All the constants are stored in a single file
    S* h_constants = optimized_constants<S>(width - 1);
    int constants_len = round_constants_len + mds_matrix_len * 2 + sparse_matrices_len;

    // Malloc memory for copying constants
    S* d_constants;
    CHK_IF_RETURN(cudaMallocAsync(&d_constants, sizeof(S) * constants_len, stream));

    // Copy constants
    CHK_IF_RETURN(cudaMemcpyAsync(d_constants, h_constants, sizeof(S) * constants_len, cudaMemcpyHostToDevice, stream));

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
    *constants = {arity,      partial_rounds,    full_rounds_half, round_constants,
                  mds_matrix, non_sparse_matrix, sparse_matrices,  domain_tag};
    return CHK_LAST();
  }

  extern "C" cudaError_t CONCAT_EXPAND(CURVE, InitOptimizedPoseidonConstants)(
    int arity, device_context::DeviceContext& ctx, PoseidonConstants<curve_config::scalar_t>* constants)
  {
    return init_optimized_poseidon_constants<curve_config::scalar_t>(arity, ctx, constants);
  }

  extern "C" cudaError_t CONCAT_EXPAND(CURVE, PoseidonHash)(
    curve_config::scalar_t* input,
    curve_config::scalar_t* output,
    int number_of_states,
    int arity,
    const PoseidonConstants<curve_config::scalar_t>& constants,
    PoseidonConfig& config)
  {
    switch (arity) {
    case 2:
      return poseidon_hash<curve_config::scalar_t, 3>(input, output, number_of_states, constants, config);
    case 4:
      return poseidon_hash<curve_config::scalar_t, 5>(input, output, number_of_states, constants, config);
    case 8:
      return poseidon_hash<curve_config::scalar_t, 9>(input, output, number_of_states, constants, config);
    case 11:
      return poseidon_hash<curve_config::scalar_t, 12>(input, output, number_of_states, constants, config);
    default:
      throw std::runtime_error("invalid arity");
    }
  }
} // namespace poseidon