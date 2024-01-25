#include "poseidon.cuh"
#include "kernels.cu"

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
  cudaError_t permute_many(
    S* states,
    size_t number_of_states,
    const PoseidonConstants<S, T>& constants,
    cudaStream_t& stream)
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
  cudaError_t
  poseidon_hash(S* input, S* output, size_t number_of_states, const PoseidonConstants<S, T>& constants, const PoseidonConfig& config)
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
      // padded and coppied to device in a T x NumberOfBlocks matrix
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

    #if !defined(__CUDA_ARCH__) && defined(DEBUG)
    cudaDeviceSynchronize();
    std::cout << "Domain separation: " << std::endl;
    print_buffer_from_cuda<S, T>(states, number_of_states * T);
    #endif

    cudaError_t hash_error = permute_many<S, T>(states, number_of_states, constants, stream);
    CHK_IF_RETURN(hash_error);

    #if !defined(__CUDA_ARCH__) && defined(DEBUG)
    cudaDeviceSynchronize();
    std::cout << "Permutations: " << std::endl;
    print_buffer_from_cuda<S, T>(states, number_of_states * T);
    #endif

    get_hash_results<S, T><<<
      PKC<T>::number_of_singlehash_blocks(number_of_states), PKC<T>::singlehash_block_size, 0,
      stream>>>(states, number_of_states, output_device);

    if (config.loop_state) {
      copy_recursive<S, T><<<
        PKC<T>::number_of_singlehash_blocks(number_of_states), PKC<T>::singlehash_block_size, 0,
        stream>>>(states, number_of_states, output_device);

      #if !defined(__CUDA_ARCH__) && defined(DEBUG)
      cudaDeviceSynchronize();
      std::cout << "Looping: " << std::endl;
      print_buffer_from_cuda<S, T>(states, number_of_states / (T - 1) * T);
      #endif
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

  template <typename S, int T>
  cudaError_t init_optimized_poseidon_constants(device_context::DeviceContext ctx)
  {
    CHK_INIT_IF_RETURN();
    cudaStream_t& stream = ctx.stream;
    int full_rounds_half = FULL_ROUNDS_DEFAULT;
    int partial_rounds = partial_rounds_number_from_arity(T - 1);

    int round_constants_len = T * full_rounds_half * 2 + partial_rounds;
    int mds_matrix_len = T * T;
    int sparse_matrices_len = (T * 2 - 1) * partial_rounds;

    // All the constants are stored in a single file
    S* constants = optimized_constants<S>(T - 1);
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
    tree_domain_tag_value = (tree_domain_tag_value << (T - 1)) - tree_domain_tag_value;
    S domain_tag = S::from(tree_domain_tag_value);

    // Make sure all the constants have been copied
    CHK_IF_RETURN(cudaStreamSynchronize(stream));
    preloaded_constants<S, T> = {partial_rounds,    full_rounds_half, round_constants, mds_matrix,
                                 non_sparse_matrix, sparse_matrices,  domain_tag};
    return CHK_LAST();
  }
} // namespace poseidon