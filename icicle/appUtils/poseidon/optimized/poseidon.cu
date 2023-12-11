#include "poseidon.cuh"

namespace poseidon {
    template <typename S>
    __global__ void
    prepare_poseidon_states(S* states, size_t number_of_states, S domain_tag, const PoseidonConfiguration<S> config, bool aligned)
    {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int state_number = idx / config.t;
    if (state_number >= number_of_states) { return; }
    int element_number = idx % config.t;

    S prepared_element;

    // Domain separation
    if (element_number == 0) {
        prepared_element = domain_tag;
    } else {
        if (aligned) {
            prepared_element = states[idx];
        } else {
            prepared_element = states[state_number * config.t + element_number - 1];
        }
    }

    if (!aligned) {
        __syncthreads();
    }

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

    template <typename S>
    __device__ S vecs_mul_matrix(S element, S* matrix, int element_number, int vec_number, int size, S* shared_states)
    {
    shared_states[threadIdx.x] = element;
    __syncthreads();

    typename S::Wide element_wide = S::mul_wide(shared_states[vec_number * size], matrix[element_number]);
    for (int i = 1; i < size; i++) {
        element_wide = element_wide + S::mul_wide(shared_states[vec_number * size + i], matrix[i * size + element_number]);
    }
    __syncthreads();

    return S::reduce(element_wide);
    }

    template <typename S>
    __device__ S full_round(
    S element,
    size_t rc_offset,
    int local_state_number,
    int element_number,
    bool multiply_by_mds,
    bool add_pre_round_constants,
    bool skip_rc,
    S* shared_states,
    const PoseidonConfiguration<S> config)
    {
    if (add_pre_round_constants) {
        element = element + config.round_constants[rc_offset + element_number];
        rc_offset += config.t;
    }
    element = sbox_alpha_five(element);
    if (!skip_rc) {
        element = element + config.round_constants[rc_offset + element_number];
    }

    // Multiply all the states by mds matrix
    S* matrix = multiply_by_mds ? config.mds_matrix : config.non_sparse_matrix;
    return vecs_mul_matrix(element, matrix, element_number, local_state_number, config.t, shared_states);
    }

    // Execute full rounds
    template <typename S>
    __global__ void full_rounds(
    S* states, size_t number_of_states, size_t rc_offset, bool first_half, const PoseidonConfiguration<S> config)
    {
    extern __shared__ S shared_states[];

    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int state_number = idx / config.t;
    if (state_number >= number_of_states) { return; }
    int local_state_number = threadIdx.x / config.t;
    int element_number = idx % config.t;

    bool add_pre_round_constants = first_half;
    for (int i = 0; i < config.full_rounds_half; i++) {
        states[idx] =
        full_round(states[idx], rc_offset, local_state_number,
                    element_number, !first_half || (i < (config.full_rounds_half - 1)),
                    add_pre_round_constants, !first_half && (i == config.full_rounds_half - 1), shared_states, config);
        rc_offset += config.t;

        if (add_pre_round_constants) {
        rc_offset += config.t;
        add_pre_round_constants = false;
        }
    }
    }

    template <typename S>
    __device__ S partial_round(S* state, size_t rc_offset, int round_number, const PoseidonConfiguration<S> config)
    {
    S element = state[0];
    element = sbox_alpha_five(element);
    element = element + config.round_constants[rc_offset];

    S* sparse_matrix = &config.sparse_matrices[(config.t * 2 - 1) * round_number];

    typename S::Wide state_0_wide = S::mul_wide(element, sparse_matrix[0]);
    for (int i = 1; i < config.t; i++) {
        state_0_wide = state_0_wide + S::mul_wide(state[i], sparse_matrix[i]);
    }
    state[0] = S::reduce(state_0_wide);

    for (int i = 1; i < config.t; i++) {
        state[i] = state[i] + (element * sparse_matrix[config.t + i - 1]);
    }
    }

    // Execute partial rounds
    template <typename S>
    __global__ void
    partial_rounds(S* states, size_t number_of_states, size_t rc_offset, const PoseidonConfiguration<S> config)
    {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= number_of_states) { return; }

    S* state = &states[idx * config.t];

    for (int i = 0; i < config.partial_rounds; i++) {
        partial_round(state, rc_offset, i, config);
        rc_offset++;
    }
    }

    // These function is just doing copy from the states to the output
    template <typename S>
    __global__ void get_hash_results(S* states, size_t number_of_states, S* out, int t)
    {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= number_of_states) { return; }

    out[idx] = states[idx * t + 1];
    }

    template <typename S>
    __global__ void copy_recursive(S * state, size_t number_of_states, S * out, int t) {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= number_of_states) {
        return;
    }

    state[(idx / (t - 1) * t) + (idx % (t - 1)) + 1] = out[idx];
    }
}

template class poseidon::OptimizedPoseidon<curve_config::scalar_t>;