#include "poseidon2/poseidon2.cuh"
#include "constants.cu"
#include "kernels.cu"

namespace poseidon2 {
  static int poseidon_block_size = 128;

  template <typename S>
  int poseidon_number_of_blocks(size_t number_of_states)
  {
    return number_of_states / poseidon_block_size + static_cast<bool>(number_of_states % poseidon_block_size);
  }

  template <typename S>
  Poseidon2<S>::Poseidon2(int width, MdsType mds_type, DiffusionStrategy diffusion, device_context::DeviceContext& ctx) {
    Poseidon2Constants<S> constants;
    init_poseidon2_constants(width, mds_type, diffusion, ctx, &constants);
    this->constants = constants;
  }

  template <typename S>
  Poseidon2<S>::~Poseidon2() {
    std::cout << "Destructor called" << std::endl;
    auto ctx = device_context::get_default_device_context();
    release_poseidon2_constants<S>(&this->constants, ctx);
  }

  template <typename S>
  cudaError_t Poseidon2<S>::squeeze_states(
    const S* states,
    unsigned int number_of_states,
    unsigned int rate,
    S* output,
    device_context::DeviceContext& ctx,
    unsigned int offset
    )
  {
#define P2_SQUEEZE_T(width)                                                                                               \
  case width:                                                                                                          \
    squeeze_states_kernel<S, width, 1, 0><<<poseidon_number_of_blocks<S>(number_of_states), poseidon_block_size, 0, ctx.stream>>>(\
      states, number_of_states, output);\
    break;

    switch (this->constants.width) {
      P2_SQUEEZE_T(2)
      P2_SQUEEZE_T(3)
      P2_SQUEEZE_T(4)
      P2_SQUEEZE_T(8)
      P2_SQUEEZE_T(12)
      P2_SQUEEZE_T(16)
      P2_SQUEEZE_T(20)
      P2_SQUEEZE_T(24)
    default:
      THROW_ICICLE_ERR(
        IcicleError_t::InvalidArgument, "PoseidonSqueeze: #width must be one of [2, 3, 4, 8, 12, 16, 20, 24]");
    }
    // Squeeze states to get results
    CHK_IF_RETURN(cudaPeekAtLastError());
    return CHK_LAST();
  }

  template <typename S>
  cudaError_t Poseidon2<S>::run_permutation_kernel(
        const S* states,
        S* output,
        unsigned int number_of_states,
        device_context::DeviceContext& ctx
    ) {
// #define P2_PERM_T(width)                                                                                               \
//   case width:                                                                                                          \
//     poseidon2_permutation_kernel<S, width>  \
//       <<<poseidon_number_of_blocks<S>(number_of_states), poseidon_block_size, 0, ctx.stream>>>( \
//         states, output, number_of_states, this->constants);\
//     break;

//     switch (this->constants.width) {
//       P2_PERM_T(2)
//       P2_PERM_T(3)
//       P2_PERM_T(4)
//       P2_PERM_T(8)
//       P2_PERM_T(12)
//       P2_PERM_T(16)
//       P2_PERM_T(20)
//       P2_PERM_T(24)
//     default:
//       THROW_ICICLE_ERR(
//         IcicleError_t::InvalidArgument, "PoseidonPermutation: #width must be one of [2, 3, 4, 8, 12, 16, 20, 24]");
//     }

    cudaStream_t& stream = ctx.stream;
    Poseidon2Constants<S> constants;
    init_poseidon2_constants(3, MdsType::DEFAULT_MDS, DiffusionStrategy::DEFAULT_DIFFUSION, ctx, &constants);
    poseidon2_permutation_kernel<S, 3>  \
      <<<1, 1, 0, stream>>>( \
        states, output, number_of_states, constants);\
    CHK_IF_RETURN(cudaPeekAtLastError());

    return CHK_LAST();
  }
  
  template <typename S>
  cudaError_t Poseidon2<S>::permute_many(
        const S* states,
        S* output,
        unsigned int number_of_states,
        device_context::DeviceContext& ctx) {
    CHK_INIT_IF_RETURN();

    std::cout << "Inside" << std::endl;

    bool states_on_device = device_context::is_host_ptr(states, ctx.device_id);
    bool output_on_device = device_context::is_host_ptr(output, ctx.device_id);
    if (states_on_device != output_on_device) {
      THROW_ICICLE_ERR(IcicleError_t::InvalidArgument, "States and output should be both on-device or on-host");
    }

    S* d_states = nullptr;
    if (!states_on_device)
      CHK_IF_RETURN(cudaMallocAsync(&d_states, number_of_states * this->constants.width * sizeof(S), ctx.stream))

    cudaError_t permutation_error = run_permutation_kernel(
      states_on_device ? d_states : states,
      states_on_device ? d_states : output,
      number_of_states,
      ctx
    );
    CHK_IF_RETURN(permutation_error);

    if (!states_on_device) {
      CHK_IF_RETURN(
        cudaMemcpyAsync(output, d_states, number_of_states * this->constants.width * sizeof(S), cudaMemcpyDeviceToHost, ctx.stream));
      CHK_IF_RETURN(cudaFreeAsync(d_states, ctx.stream))
    }

    return CHK_LAST();
  }

  template <typename S>
  cudaError_t Poseidon2<S>::compress_many(
      const S* states,
      S* output,
      unsigned int number_of_states,
      unsigned int rate,
      device_context::DeviceContext& ctx,
      unsigned int offset,
      S* perm_output
  ) {
    CHK_INIT_IF_RETURN();
    bool states_on_device = device_context::is_host_ptr(states, ctx.device_id);
    bool output_on_device = device_context::is_host_ptr(output, ctx.device_id);

    // Allocate memory for states if needed
    bool need_allocate_perm_output = perm_output == nullptr;
    if (need_allocate_perm_output) {
      CHK_IF_RETURN(cudaMallocAsync(&perm_output, number_of_states * this->constants.width * sizeof(S), ctx.stream))
    }

    // Copy states from host if they are on host
    if (!states_on_device) {
      CHK_IF_RETURN(
        cudaMemcpyAsync(perm_output, states, number_of_states * this->constants.width * sizeof(S), cudaMemcpyHostToDevice, ctx.stream));
    }

    // Run the permutation
    cudaError_t hash_error = run_permutation_kernel(
      states_on_device ? states : perm_output,
      perm_output,
      number_of_states,
      ctx
    );
    CHK_IF_RETURN(hash_error);
    
    // Squeeze states to get results
    cudaError_t squeeze_error = squeeze_states(perm_output, number_of_states, rate, perm_output, ctx, offset);
    CHK_IF_RETURN(squeeze_error);

    if (output != perm_output) {
      CHK_IF_RETURN(
        cudaMemcpyAsync(output, perm_output, number_of_states * this->constants.width * sizeof(S),
                        output_on_device ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost,
                        ctx.stream));
    }

    if (need_allocate_perm_output) CHK_IF_RETURN(cudaFreeAsync(perm_output, ctx.stream))

    return CHK_LAST();
  }
} // namespace poseidon2