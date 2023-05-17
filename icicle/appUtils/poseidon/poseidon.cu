#include "poseidon.cuh"

// Used in matrix multiplication
extern __shared__ scalar_t shared_states[];

__global__ void prepare_poseidon_states(scalar_t * inp, scalar_t * states, size_t number_of_states, scalar_t domain_tag, const PoseidonConfiguration config) {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int state_number = idx / config.t;
    if (state_number >= number_of_states) {
        return;
    }
    int element_number = idx % config.t;

    scalar_t prepared_element;

    // Domain separation
    if (element_number == 0) {
        prepared_element = domain_tag;
    } else {
        prepared_element = inp[state_number * (config.t - 1) + element_number - 1];
    }

    // Add pre-round constant
    prepared_element = prepared_element + config.round_constants[element_number];

    // Store element in state
    states[idx] = prepared_element;
}

__device__ __forceinline__ scalar_t sbox_alpha_five(scalar_t element) {
    scalar_t result = element * element;
    result = result * result;
    return result * element;
}

__device__ scalar_t vecs_mul_matrix(scalar_t element, scalar_t * matrix, int element_number, int vec_number, int size) {
    shared_states[threadIdx.x] = element;
    __syncthreads();

    element = scalar_t::zero();
    for (int i = 0; i < size; i++) {
        element = element + (shared_states[vec_number * size + i] * matrix[i * size + element_number]);
    }
    __syncthreads();
    return element;
}

__device__ scalar_t full_round(scalar_t element,
                               size_t rc_offset,
                               int local_state_number,
                               int element_number,
                               bool multiply_by_mds,
                               bool add_round_constant,
                               const PoseidonConfiguration config) {
    element = sbox_alpha_five(element);
    if (add_round_constant) {
        element = element + config.round_constants[rc_offset + element_number];
    }

    // Multiply all the states by mds matrix
    scalar_t * matrix = multiply_by_mds ? config.mds_matrix : config.non_sparse_matrix;
    return vecs_mul_matrix(element, matrix, element_number, local_state_number, config.t);
}

// Execute full rounds
__global__ void full_rounds(scalar_t * states, size_t number_of_states, size_t rc_offset, bool first_half, const PoseidonConfiguration config) {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int state_number = idx / config.t;
    if (state_number >= number_of_states) {
        return;
    }
    int local_state_number = threadIdx.x / config.t;
    int element_number = idx % config.t;

    for (int i = 0; i < config.full_rounds_half - 1; i++) {
        states[idx] = full_round(states[idx],
                                 rc_offset,
                                 local_state_number,
                                 element_number,
                                 true,
                                 true,
                                 config);
        rc_offset += config.t;
    }

    states[idx] = full_round(states[idx],
                             rc_offset,
                             local_state_number,
                             element_number,
                             !first_half,
                             first_half,
                             config);
}

__device__ scalar_t partial_round(scalar_t * state,
                                  size_t rc_offset,
                                  int round_number,
                                  const PoseidonConfiguration config) {
    scalar_t element = state[0];
    element = sbox_alpha_five(element);
    element = element + config.round_constants[rc_offset];

    scalar_t * sparse_matrix = &config.sparse_matrices[(config.t * 2 - 1) * round_number];

    state[0] = element * sparse_matrix[0];
    for (int i = 1; i < config.t; i++) {
        state[0] = state[0] + (state[i] * sparse_matrix[i]);
    }

    for (int i = 1; i < config.t; i++) {
        state[i] = state[i] + (element * sparse_matrix[config.t + i - 1]);
    }
}

// Execute partial rounds
__global__ void partial_rounds(scalar_t * states, size_t number_of_states, size_t rc_offset, const PoseidonConfiguration config) {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= number_of_states) {
        return;
    }

    scalar_t * state = &states[idx * config.t];

    for (int i = 0; i < config.partial_rounds; i++) {
        partial_round(state, rc_offset, i, config);
        rc_offset++;
    }
}

// These function is just doing copy from the states to the output
__global__ void get_hash_results(scalar_t * states, size_t number_of_states, scalar_t * out, int t) {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= number_of_states) {
        return;
    }

    out[idx] = states[idx * t + 1];
}

#ifndef __CUDA_ARCH__
void Poseidon::hash_blocks(const scalar_t * inp, size_t blocks, scalar_t * out, HashType hash_type) {
    scalar_t * states, * inp_device;

    // allocate memory for {blocks} states of {t} scalars each
    cudaMalloc(&states, blocks * this->t * sizeof(scalar_t));

    // Move input to cuda
    cudaMalloc(&inp_device, blocks * (this->t - 1) * sizeof(scalar_t));
    cudaMemcpy(inp_device, inp, blocks * (this->t - 1) * sizeof(scalar_t), cudaMemcpyHostToDevice);

    size_t rc_offset = 0;

    // The logic behind this is that 1 thread only works on 1 element
    // We have {t} elements in each state, and {blocks} states total
    int number_of_threads = (256 / this->t) * this->t;
    int hashes_per_block = number_of_threads / this->t;
    int total_number_of_threads = blocks * this->t;
    int number_of_blocks = total_number_of_threads / number_of_threads +
        static_cast<bool>(total_number_of_threads % number_of_threads);

    // The partial rounds operates on the whole state, so we define
    // the parallelism params for processing a single hash preimage per thread
    int singlehash_block_size = 128;
    int number_of_singlehash_blocks = blocks / singlehash_block_size + static_cast<bool>(blocks % singlehash_block_size);

    // Pick the domain_tag accordinaly
    scalar_t domain_tag;
    switch (hash_type) {
        case HashType::ConstInputLen:
            domain_tag = this->const_input_no_pad_domain_tag;
            break;

        case HashType::MerkleTree:
            domain_tag = this->tree_domain_tag;
    }

    #ifdef DEBUG
    auto start_time = std::chrono::high_resolution_clock::now();
    #endif

    // Domain separation and adding pre-round constants
    prepare_poseidon_states <<< number_of_blocks, number_of_threads >>> (inp_device, states, blocks, domain_tag, this->config);
    rc_offset += this->t;
    cudaFree(inp_device);

    #ifdef DEBUG
    cudaThreadSynchronize();
    std::cout << "Domain separation: " << rc_offset << std::endl;
    // print_buffer_from_cuda(states, blocks * this->t);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Elapsed time: " << elapsed_time.count() << " ms" << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    #endif

    // execute half full rounds
    full_rounds <<< number_of_blocks, number_of_threads, sizeof(scalar_t) * hashes_per_block * this->t >>> (states, blocks, rc_offset, true, this->config);
    rc_offset += this->t * this->config.full_rounds_half;

    #ifdef DEBUG
    cudaThreadSynchronize();
    std::cout << "Full rounds 1. RCOFFSET: " << rc_offset << std::endl;
    // print_buffer_from_cuda(states, blocks * this->t);

    end_time = std::chrono::high_resolution_clock::now();
    elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Elapsed time: " << elapsed_time.count() << " ms" << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    #endif

    // execute partial rounds
    partial_rounds <<< number_of_singlehash_blocks, singlehash_block_size >>> (states, blocks, rc_offset, this->config);
    rc_offset += this->config.partial_rounds;

    #ifdef DEBUG
    cudaThreadSynchronize();
    std::cout << "Partial rounds. RCOFFSET: " << rc_offset << std::endl;
    // print_buffer_from_cuda(states, blocks * this->t);

    end_time = std::chrono::high_resolution_clock::now();
    elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Elapsed time: " << elapsed_time.count() << " ms" << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    #endif

    // execute half full rounds
    full_rounds <<< number_of_blocks, number_of_threads, sizeof(scalar_t) * hashes_per_block * this->t >>> (states, blocks, rc_offset, false, this->config);

    #ifdef DEBUG
    cudaThreadSynchronize();
    std::cout << "Full rounds 2. RCOFFSET: " << rc_offset << std::endl;
    // print_buffer_from_cuda(states, blocks * this->t);
    end_time = std::chrono::high_resolution_clock::now();
    elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Elapsed time: " << elapsed_time.count() << " ms" << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    #endif

    // get output
    scalar_t * out_device;
    cudaMalloc(&out_device, blocks * sizeof(scalar_t));
    get_hash_results <<< number_of_singlehash_blocks, singlehash_block_size >>> (states, blocks, out_device, this->config.t);

    #ifdef DEBUG
    cudaThreadSynchronize();
    std::cout << "Get hash results" << std::endl;
    end_time = std::chrono::high_resolution_clock::now();
    elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Elapsed time: " << elapsed_time.count() << " ms" << std::endl;
    #endif
    cudaMemcpy(out, out_device, blocks * sizeof(scalar_t), cudaMemcpyDeviceToHost);
    cudaFree(out_device);
    cudaFree(states);
}
#endif