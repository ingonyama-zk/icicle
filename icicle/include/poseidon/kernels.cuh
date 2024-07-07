#pragma once
#ifndef POSEIDON_KERNELS_H
#define POSEIDON_KERNELS_H

#include "gpu-utils/modifiers.cuh"
#include "poseidon/constants.cuh"

namespace poseidon {
  template <typename S, int T>
  __global__ void prepare_poseidon_states(const S* input, S* states, unsigned int number_of_states, const S domain_tag)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int state_number = idx / T;
    if (state_number >= number_of_states) { return; }
    int element_number = idx % T;

    S prepared_element;

    // Domain separation
    if (element_number == 0) {
      prepared_element = domain_tag;
    } else {
      prepared_element = input[idx - state_number - 1];
    }

    // Store element in state
    states[idx] = prepared_element;
  }

  template <typename S>
  DEVICE_INLINE S sbox_el(S element, const int alpha)
  {
    S result2 = S::sqr(element);
    switch (alpha) {
    case 3:
      return result2 * element;
    case 5:
      return S::sqr(result2) * element;
    case 7:
      return S::sqr(result2) * result2 * element;
    case 11:
      return S::sqr(S::sqr(result2)) * result2 * element;
    }
  }

  template <typename S, int T>
  __device__ S vecs_mul_matrix(S element, S* matrix, int element_number, int vec_number, S* shared_states)
  {
    __syncthreads();
    shared_states[threadIdx.x] = element;
    __syncthreads();

    typename S::Wide element_wide = S::mul_wide(shared_states[vec_number * T], matrix[element_number]);
    UNROLL
    for (int i = 1; i < T; i++) {
      element_wide = element_wide + S::mul_wide(shared_states[vec_number * T + i], matrix[i * T + element_number]);
    }

    return S::reduce(element_wide);
  }

  template <typename S, int T>
  __device__ S full_round(
    S element,
    size_t rc_offset,
    int local_state_number,
    int element_number,
    bool multiply_by_mds,
    bool add_pre_round_constants,
    bool skip_rc,
    S* shared_states,
    const PoseidonConstants<S>& constants)
  {
    if (add_pre_round_constants) {
      element = element + constants.round_constants[rc_offset + element_number];
      rc_offset += T;
    }
    element = sbox_el(element, constants.alpha);
    if (!skip_rc) { element = element + constants.round_constants[rc_offset + element_number]; }

    // Multiply all the states by mds matrix
    S* matrix = multiply_by_mds ? constants.mds_matrix : constants.non_sparse_matrix;
    return vecs_mul_matrix<S, T>(element, matrix, element_number, local_state_number, shared_states);
  }

  template <typename S, int T>
  __global__ void full_rounds(
    S* states, size_t number_of_states, size_t rc_offset, bool first_half, const PoseidonConstants<S> constants)
  {
    extern __shared__ S shared_states[];

    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int state_number = idx / T;
    if (state_number >= number_of_states) { return; }
    int local_state_number = threadIdx.x / T;
    int element_number = idx % T;

    S new_el = states[idx];
    bool add_pre_round_constants = first_half;
    for (int i = 0; i < constants.full_rounds_half; i++) {
      new_el = full_round<S, T>(
        new_el, rc_offset, local_state_number, element_number, !first_half || (i < (constants.full_rounds_half - 1)),
        add_pre_round_constants, !first_half && (i == constants.full_rounds_half - 1), shared_states, constants);
      rc_offset += T;

      if (add_pre_round_constants) {
        rc_offset += T;
        add_pre_round_constants = false;
      }
    }
    states[idx] = new_el;
  }

  template <typename S, int T>
  __device__ S partial_round(S state[T], size_t rc_offset, int round_number, const PoseidonConstants<S>& constants)
  {
    S element = state[0];
    element = sbox_el(element, constants.alpha);
    element = element + constants.round_constants[rc_offset];

    S* sparse_matrix = &constants.sparse_matrices[(T * 2 - 1) * round_number];

    typename S::Wide state_0_wide = S::mul_wide(element, sparse_matrix[0]);

    UNROLL
    for (int i = 1; i < T; i++) {
      state_0_wide = state_0_wide + S::mul_wide(state[i], sparse_matrix[i]);
    }

    state[0] = S::reduce(state_0_wide);

    UNROLL
    for (int i = 1; i < T; i++) {
      state[i] = state[i] + (element * sparse_matrix[T + i - 1]);
    }
  }

  template <typename S, int T>
  __global__ void
  partial_rounds(S* states, size_t number_of_states, size_t rc_offset, const PoseidonConstants<S> constants)
  {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= number_of_states) { return; }

    S state[T];
    UNROLL
    for (int i = 0; i < T; i++) {
      state[i] = states[idx * T + i];
    }

    for (int i = 0; i < constants.partial_rounds; i++) {
      partial_round<S, T>(state, rc_offset, i, constants);
      rc_offset++;
    }

    UNROLL
    for (int i = 0; i < T; i++) {
      states[idx * T + i] = state[i];
    }
  }

  template <typename S, int T>
  __global__ void
  squeeze_states_kernel(const S* states, unsigned int number_of_states, unsigned int rate, unsigned int offset, S* out)
  {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= number_of_states) { return; }

    for (int i = 0; i < rate; i++) {
      out[idx * rate + i] = states[idx * T + offset + i];
    }
  }

  template <typename S, int T>
  cudaError_t poseidon_permutation_kernel(
    const S* input,
    S* out,
    unsigned int number_of_states,
    unsigned int input_len,
    unsigned int output_len,
    const PoseidonConstants<S>& constants,
    cudaStream_t& stream)
  {
    S* states;
    CHK_IF_RETURN(cudaMallocAsync(&states, number_of_states * T * sizeof(S), stream));

    prepare_poseidon_states<S, T>
      <<<PKC::number_of_full_blocks(T, number_of_states), PKC::number_of_threads(T), 0, stream>>>(
        input, states, number_of_states, constants.domain_tag);

    size_t rc_offset = 0;
    full_rounds<S, T><<<
      PKC::number_of_full_blocks(T, number_of_states), PKC::number_of_threads(T),
      sizeof(S) * PKC::hashes_per_block(T) * T, stream>>>(
      states, number_of_states, rc_offset, FIRST_FULL_ROUNDS, constants);
    rc_offset += T * (constants.full_rounds_half + 1);

    partial_rounds<S, T><<<PKC::number_of_singlehash_blocks(number_of_states), PKC::singlehash_block_size, 0, stream>>>(
      states, number_of_states, rc_offset, constants);
    rc_offset += constants.partial_rounds;

    full_rounds<S, T><<<
      PKC::number_of_full_blocks(T, number_of_states), PKC::number_of_threads(T),
      sizeof(S) * PKC::hashes_per_block(T) * T, stream>>>(
      states, number_of_states, rc_offset, SECOND_FULL_ROUNDS, constants);

    squeeze_states_kernel<S, T>
      <<<PKC::number_of_singlehash_blocks(number_of_states), PKC::singlehash_block_size, 0, stream>>>(
        states, number_of_states, output_len, 1, out);

    CHK_IF_RETURN(cudaFreeAsync(states, stream));
    return CHK_LAST();
  }
} // namespace poseidon

#endif