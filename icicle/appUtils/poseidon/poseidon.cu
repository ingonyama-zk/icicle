#include "poseidon.cuh"
#include "constants.cu"
#include "kernels.cu"

namespace poseidon {
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
      THROW_ICICLE_ERR(IcicleError_t::InvalidArgument, "PoseidonHash: #arity must be one of [2, 4, 8, 11]");
    }
    return CHK_LAST();
  }
} // namespace poseidon