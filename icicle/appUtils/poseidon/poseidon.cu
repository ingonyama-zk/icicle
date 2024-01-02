#include "../../utils/error_handler.cuh"
#include "poseidon.cuh"

template <typename S>
__global__ void
prepare_poseidon_states(S* states, size_t number_of_states, S domain_tag, const PoseidonConfiguration<S> config)
{
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int state_number = idx / config.t;
  if (state_number >= number_of_states) { return; }
  int element_number = idx % config.t;

  S prepared_element;

  // Domain separation
  if (element_number == 0) {
    prepared_element = domain_tag;
  } else {
    prepared_element = states[state_number * config.t + element_number - 1];
  }

  // Add pre-round constant
  prepared_element = prepared_element + config.round_constants[element_number];

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

  element = S::zero();
  for (int i = 0; i < size; i++) {
    element = element + (shared_states[vec_number * size + i] * matrix[i * size + element_number]);
  }
  __syncthreads();
  return element;
}

template <typename S>
__device__ S full_round(
  S element,
  size_t rc_offset,
  int local_state_number,
  int element_number,
  bool multiply_by_mds,
  bool add_round_constant,
  S* shared_states,
  const PoseidonConfiguration<S> config)
{
  element = sbox_alpha_five(element);
  if (add_round_constant) { element = element + config.round_constants[rc_offset + element_number]; }

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

  for (int i = 0; i < config.full_rounds_half - 1; i++) {
    states[idx] =
      full_round(states[idx], rc_offset, local_state_number, element_number, true, true, shared_states, config);
    rc_offset += config.t;
  }

  states[idx] = full_round(
    states[idx], rc_offset, local_state_number, element_number, !first_half, first_half, shared_states, config);
}

template <typename S>
__device__ S partial_round(S* state, size_t rc_offset, int round_number, const PoseidonConfiguration<S> config)
{
  S element = state[0];
  element = sbox_alpha_five(element);
  element = element + config.round_constants[rc_offset];

  S* sparse_matrix = &config.sparse_matrices[(config.t * 2 - 1) * round_number];

  state[0] = element * sparse_matrix[0];
  for (int i = 1; i < config.t; i++) {
    state[0] = state[0] + (state[i] * sparse_matrix[i]);
  }

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
__host__ cudaError_t
Poseidon<S>::hash_blocks(const S* inp, size_t blocks, S* out, HashType hash_type, cudaStream_t stream)
{
  S* states;

  // allocate memory for {blocks} states of {t} scalars each
  CHK_IF_RETURN(cudaMallocAsync(&states, blocks * this->t * sizeof(S), stream));

  // This is where the input matrix of size Arity x NumberOfBlocks is
  // padded and coppied to device in a T x NumberOfBlocks matrix
  cudaMemcpy2DAsync(
    states, this->t * sizeof(S),       // Device pointer and device pitch
    inp, (this->t - 1) * sizeof(S),    // Host pointer and pitch
    (this->t - 1) * sizeof(S), blocks, // Size of the source matrix (Arity x NumberOfBlocks)
    cudaMemcpyHostToDevice, stream);

  size_t rc_offset = 0;

  // The logic behind this is that 1 thread only works on 1 element
  // We have {t} elements in each state, and {blocks} states total
  int number_of_threads = (256 / this->t) * this->t;
  int hashes_per_block = number_of_threads / this->t;
  int total_number_of_threads = blocks * this->t;
  int number_of_blocks =
    total_number_of_threads / number_of_threads + static_cast<bool>(total_number_of_threads % number_of_threads);

  // The partial rounds operates on the whole state, so we define
  // the parallelism params for processing a single hash preimage per thread
  int singlehash_block_size = 128;
  int number_of_singlehash_blocks = blocks / singlehash_block_size + static_cast<bool>(blocks % singlehash_block_size);

  // Pick the domain_tag accordinaly
  S domain_tag;
  switch (hash_type) {
  case HashType::ConstInputLen:
    domain_tag = this->const_input_no_pad_domain_tag;
    break;

  case HashType::MerkleTree:
    domain_tag = this->tree_domain_tag;
  }

#if !defined(__CUDA_ARCH__) && defined(DEBUG)
  auto start_time = std::chrono::high_resolution_clock::now();
#endif

  // Domain separation and adding pre-round constants
  prepare_poseidon_states<<<number_of_blocks, number_of_threads, 0, stream>>>(states, blocks, domain_tag, this->config);
  rc_offset += this->t;

#if !defined(__CUDA_ARCH__) && defined(DEBUG)
  cudaStreamSynchronize(stream);
  std::cout << "Domain separation: " << rc_offset << std::endl;
  // print_buffer_from_cuda<S>(states, blocks * this->t);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  std::cout << "Elapsed time: " << elapsed_time.count() << " ms" << std::endl;
  start_time = std::chrono::high_resolution_clock::now();
#endif

  // execute half full rounds
  full_rounds<<<number_of_blocks, number_of_threads, this->t * sizeof(S) * hashes_per_block, stream>>>(
    states, blocks, rc_offset, true, this->config);
  rc_offset += this->t * this->config.full_rounds_half;

#if !defined(__CUDA_ARCH__) && defined(DEBUG)
  cudaStreamSynchronize(stream);
  std::cout << "Full rounds 1. RCOFFSET: " << rc_offset << std::endl;
  // print_buffer_from_cuda<S>(states, blocks * this->t);

  end_time = std::chrono::high_resolution_clock::now();
  elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  std::cout << "Elapsed time: " << elapsed_time.count() << " ms" << std::endl;
  start_time = std::chrono::high_resolution_clock::now();
#endif

  // execute partial rounds
  partial_rounds<<<number_of_singlehash_blocks, singlehash_block_size, 0, stream>>>(
    states, blocks, rc_offset, this->config);
  rc_offset += this->config.partial_rounds;

#if !defined(__CUDA_ARCH__) && defined(DEBUG)
  cudaStreamSynchronize(stream);
  std::cout << "Partial rounds. RCOFFSET: " << rc_offset << std::endl;
  // print_buffer_from_cuda<S>(states, blocks * this->t);

  end_time = std::chrono::high_resolution_clock::now();
  elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  std::cout << "Elapsed time: " << elapsed_time.count() << " ms" << std::endl;
  start_time = std::chrono::high_resolution_clock::now();
#endif

  // execute half full rounds
  full_rounds<<<number_of_blocks, number_of_threads, this->t * sizeof(S) * hashes_per_block, stream>>>(
    states, blocks, rc_offset, false, this->config);

#if !defined(__CUDA_ARCH__) && defined(DEBUG)
  cudaStreamSynchronize(stream);
  std::cout << "Full rounds 2. RCOFFSET: " << rc_offset << std::endl;
  // print_buffer_from_cuda<S>(states, blocks * this->t);
  end_time = std::chrono::high_resolution_clock::now();
  elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  std::cout << "Elapsed time: " << elapsed_time.count() << " ms" << std::endl;
  start_time = std::chrono::high_resolution_clock::now();
#endif

  // get output
  S* out_device;
  cudaMalloc(&out_device, blocks * sizeof(S));
  get_hash_results<<<number_of_singlehash_blocks, singlehash_block_size, 0, stream>>>(
    states, blocks, out_device, this->config.t);

#if !defined(__CUDA_ARCH__) && defined(DEBUG)
  cudaStreamSynchronize(stream);
  std::cout << "Get hash results" << std::endl;
  end_time = std::chrono::high_resolution_clock::now();
  elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  std::cout << "Elapsed time: " << elapsed_time.count() << " ms" << std::endl;
#endif
  cudaMemcpyAsync(out, out_device, blocks * sizeof(S), cudaMemcpyDeviceToHost, stream);
  cudaFreeAsync(out_device, stream);
  cudaFreeAsync(states, stream);

#if !defined(__CUDA_ARCH__) && defined(DEBUG)
  cudaDeviceReset();
#endif
  return CHK_LAST();
}