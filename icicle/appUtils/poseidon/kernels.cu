#include "poseidon.cuh"

namespace poseidon {
  template <typename S, int T>
  __global__ void prepare_poseidon_states(S* states, size_t number_of_states, S domain_tag, bool aligned)
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
      if (aligned) {
        prepared_element = states[idx];
      } else {
        prepared_element = states[idx - 1];
      }
    }

    // We need __syncthreads here if the state is not aligned
    // because then we need to shift the vector [A, B, 0] -> [D, A, B]
    if (!aligned) { __syncthreads(); }

    // Store element in state
    states[idx] = prepared_element;
  }

  template <typename S>
  __device__ __forceinline__ S sbox_alpha_five(S element)
  {
    S result = S::sqr(element);
    result = S::sqr(result);
    return result * element;
  }

  template <typename S, int T>
  __device__ S vecs_mul_matrix(S element, S* matrix, int element_number, int vec_number, S* shared_states)
  {
    __syncthreads();
    shared_states[threadIdx.x] = element;
    __syncthreads();

    typename S::Wide element_wide = S::mul_wide(shared_states[vec_number * T], matrix[element_number]);
#pragma unroll
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
    element = sbox_alpha_five(element);
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
    element = sbox_alpha_five(element);
    element = element + constants.round_constants[rc_offset];

    S* sparse_matrix = &constants.sparse_matrices[(T * 2 - 1) * round_number];

    typename S::Wide state_0_wide = S::mul_wide(element, sparse_matrix[0]);

#pragma unroll
    for (int i = 1; i < T; i++) {
      state_0_wide = state_0_wide + S::mul_wide(state[i], sparse_matrix[i]);
    }

    state[0] = S::reduce(state_0_wide);

#pragma unroll
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
#pragma unroll
    for (int i = 0; i < T; i++) {
      state[i] = states[idx * T + i];
    }

    for (int i = 0; i < constants.partial_rounds; i++) {
      partial_round<S, T>(state, rc_offset, i, constants);
      rc_offset++;
    }

#pragma unroll
    for (int i = 0; i < T; i++) {
      states[idx * T + i] = state[i];
    }
  }

  // These function is just doing copy from the states to the output
  template <typename S, int T>
  __global__ void get_hash_results(S* states, size_t number_of_states, S* out)
  {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= number_of_states) { return; }

    out[idx] = states[idx * T + 1];
  }

  template <typename S, int T>
  __global__ void copy_recursive(S* state, size_t number_of_states, S* out)
  {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= number_of_states) { return; }

    state[(idx / (T - 1) * T) + (idx % (T - 1)) + 1] = out[idx];
  }
} // namespace poseidon