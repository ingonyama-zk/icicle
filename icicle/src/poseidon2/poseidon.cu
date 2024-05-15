#include "poseidon2/poseidon2.cuh"
#include "constants.cu"
#include "kernels.cu"

namespace poseidon2 {
  static int poseidon_block_size = 128;

  template <typename S, int T>
  int poseidon_number_of_blocks(size_t number_of_states)
  {
    return number_of_states / poseidon_block_size + static_cast<bool>(number_of_states % poseidon_block_size);
  }

  template <typename S, int T>
  cudaError_t permute_many(
    const S* states,
    S* states_out,
    size_t number_of_states,
    const Poseidon2Constants<S>& constants,
    cudaStream_t& stream)
  {
    poseidon2_permutation_kernel<S, T>
      <<<poseidon_number_of_blocks<S, T>(number_of_states), poseidon_block_size, 0, stream>>>(
        states, states_out, number_of_states, constants);
    CHK_IF_RETURN(cudaPeekAtLastError());
    return CHK_LAST();
  }

  template <typename S, int T>
  cudaError_t poseidon2_permutation(
    const S* states,
    S* output,
    size_t number_of_states,
    const Poseidon2Constants<S>& constants,
    const Poseidon2Config& config
  ) {
    CHK_INIT_IF_RETURN();
    cudaStream_t& stream = config.ctx.stream;
    S* d_states;
    if (config.are_states_on_device && config.in_place) {
      d_states = const_cast<S*>(states);
    } else {
      // allocate memory for {number_of_states} states of {t} scalars each
      CHK_IF_RETURN(cudaMallocAsync(&d_states, number_of_states * T * sizeof(S), stream))
      CHK_IF_RETURN(cudaMemcpyAsync(d_states, states, number_of_states * T * sizeof(S),
                                    config.are_states_on_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice,
                                    stream))
    }
  }

  template <typename S, int T>
  cudaError_t poseidon2_hash(
    const S* states,
    S* output,
    size_t number_of_states,
    const Poseidon2Constants<S>& constants,
    const Poseidon2Config& config,
    S* auxiliary=nullptr)
  {
    CHK_INIT_IF_RETURN();
    cudaStream_t& stream = config.ctx.stream;
    S* d_states;
    if (config.are_states_on_device) {
      d_states = const_cast<S*>(states);
    } else {
      // allocate memory for {number_of_states} states of {t} scalars each
      CHK_IF_RETURN(cudaMallocAsync(&d_states, number_of_states * T * sizeof(S), stream))
      CHK_IF_RETURN(cudaMemcpyAsync(d_states, states, number_of_states * T * sizeof(S), cudaMemcpyHostToDevice, stream))
    }

    // --- permutation ---
    cudaError_t hash_error = permute_many<S, T>(d_states, d_states, number_of_states, constants, stream);
    CHK_IF_RETURN(hash_error);
    // === permutation ===

    if (config.mode == PoseidonMode::COMPRESSION) {
      S* output_device;
      if (config.are_outputs_on_device) {
        output_device = output;
      } else {
        CHK_IF_RETURN(cudaMallocAsync(&output_device, number_of_states * sizeof(S), stream))
      }

      get_hash_results<S, T><<<poseidon_number_of_blocks<S, T>(number_of_states), poseidon_block_size, 0, stream>>>(
        d_states, number_of_states, config.output_index, output_device);
      CHK_IF_RETURN(cudaPeekAtLastError());

      if (!config.are_outputs_on_device) {
        CHK_IF_RETURN(
          cudaMemcpyAsync(output, output_device, number_of_states * sizeof(S), cudaMemcpyDeviceToHost, stream));
        CHK_IF_RETURN(cudaFreeAsync(output_device, stream));
      }
    } else {
      if (!config.are_states_on_device || !config.are_outputs_on_device) {
        CHK_IF_RETURN(
          cudaMemcpyAsync(output, d_states, number_of_states * T * sizeof(S), cudaMemcpyDeviceToHost, stream));
      }
    }

    if (!config.are_states_on_device) CHK_IF_RETURN(cudaFreeAsync(d_states, stream));

    if (!config.is_async) return CHK_STICKY(cudaStreamSynchronize(stream));
    return CHK_LAST();
  }

  template <typename S, int WIDTH>
  Poseidon2<S, WIDTH>::Poseidon2(MdsType mds_type, DiffusionStrategy diffusion, device_context::DeviceContext& ctx) {
    Poseidon2Constants<S> constants;
    init_poseidon2_constants(WIDTH, mds_type, diffusion, ctx, &constants);
    this->constants = constants;
  }
  
  template <typename S, int WIDTH>
  Poseidon2<S, WIDTH>::~Poseidon2() {
    release_poseidon2_constants(&this->constants);
  }

  template <typename S, int WIDTH>
  cudaError_t Poseidon2Compression<S, WIDTH>::compress_many(
      const S* states,
      S* output,
      unsigned int number_of_states,
      device_context::DeviceContext& ctx,
      bool is_async,
      S* perm_output=nullptr
  ) {
    CHK_INIT_IF_RETURN();
    cudaStream_t& stream = config.ctx.stream;

    bool states_on_device = is_host_ptr(states, ctx.device_id);
    bool output_on_device = is_host_ptr(states, ctx.device_id);
    if (states_on_device != output_on_device) {
      THROW_ICICLE_ERR(IcicleError_t::InvalidArgument, "States and output should be both on-device or on-host");
    }

    // Allocate memory for states if needed
    bool need_allocate_perm_output = perm_output == nullptr;
    if (need_allocate_perm_output) {
      CHK_IF_RETURN(cudaMallocAsync(&perm_output, number_of_states * WIDTH * sizeof(S), stream))
    }

    // Copy states from host if they are on host
    if (!states_on_device) {
      CHK_IF_RETURN(
        cudaMemcpyAsync(perm_output, states, number_of_states * WIDTH * sizeof(S), cudaMemcpyHostToDevice, stream));
    }

    // Run the permutation
    cudaError_t hash_error = permute_many<S, T>(
      states_on_device ? states : perm_output,
      perm_output,
      number_of_states,
      this->constants,
      stream
    );
    CHK_IF_RETURN(hash_error);
    
    if (need_allocate_perm_output) CHK_IF_RETURN(cudaFreeAsync(perm_output, stream))

    if (!is_async) return CHK_STICKY(cudaStreamSynchronize(stream));
    return CHK_LAST();
  }
} // namespace poseidon2