#include "ntt.cuh"

#include "../../curves/curve_config.cuh"
#include "../../utils/sharedmem.cuh"
#include "../../utils/utils_kernels.cuh"

namespace ntt {

namespace {

const uint32_t MAX_NUM_THREADS = 512;
const uint32_t MAX_THREADS_BATCH = 512;          // TODO: allows 100% occupancy for scalar NTT for sm_86..sm_89
const uint32_t MAX_SHARED_MEM_ELEMENT_SIZE = 32; // TODO: occupancy calculator, hardcoded for sm_86..sm_89
const uint32_t MAX_SHARED_MEM = MAX_SHARED_MEM_ELEMENT_SIZE * MAX_NUM_THREADS;

/**
  * Computes the twiddle factors.
  * Outputs: d_twiddles[i] = omega^i.
  * @param d_twiddles input empty array.
  * @param n_twiddles number of twiddle factors.
  * @param omega multiplying factor.
  */
template <typename S>
__global__ void twiddle_factors_kernel(S* d_twiddles, int n_twiddles, S omega)
{
  d_twiddles[0] = S::one();
  for (int i = 0; i < n_twiddles - 1; i++) {
    d_twiddles[i + 1] = omega * d_twiddles[i];
  }
}

template <typename E>
__global__ void reverse_order_kernel(E* arr, E* arr_reversed, uint32_t n, uint32_t logn, uint32_t batch_size)
{
  int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (threadId < n * batch_size) {
    int idx = threadId % n;
    int batch_idx = threadId / n;
    int idx_reversed = __brev(idx) >> (32 - logn);
    arr_reversed[batch_idx * n + idx_reversed] = arr[batch_idx * n + idx];
  }
}

/**
  * Bit-reverses a batch of input arrays in-place inside GPU.
  * for example: on input array ([a[0],a[1],a[2],a[3]], 4, 2) it returns
  * [a[0],a[3],a[2],a[1]] (elements at indices 3 and 1 swhich places).
  * @param arr batch of arrays of some object of type T. Should be on GPU.
  * @param n length of `arr`.
  * @param logn log(n).
  * @param batch_size the size of the batch.
  */
template <typename E>
void reverse_order_batch(E* arr, uint32_t n, uint32_t logn, uint32_t batch_size, cudaStream_t stream)
{
  E* arr_reversed;
  cudaMallocAsync(&arr_reversed, n * batch_size * sizeof(E), stream);
  int number_of_threads = MAX_THREADS_BATCH;
  int number_of_blocks = (n * batch_size + number_of_threads - 1) / number_of_threads;
  reverse_order_kernel<<<number_of_blocks, number_of_threads, 0, stream>>>(arr, arr_reversed, n, logn, batch_size);
  cudaMemcpyAsync(arr, arr_reversed, n * batch_size * sizeof(E), cudaMemcpyDefault, stream);
  cudaFreeAsync(arr_reversed, stream);
}

/**
  * Bit-reverses an input array in-place inside GPU.
  * for example: on array ([a[0],a[1],a[2],a[3]], 4, 2) it returns
  * [a[0],a[3],a[2],a[1]] (elements at indices 3 and 1 swhich places).
  * @param arr array of some object of type T of size which is a power of 2. Should be on GPU.
  * @param n length of `arr`.
  * @param logn log(n).
  */
template <typename E>
void reverse_order(E* arr, uint32_t n, uint32_t logn, cudaStream_t stream)
{
  reverse_order_batch(arr, n, logn, 1, stream);
}

/**
  * Cooley-Tuckey NTT.
  * NOTE! this function assumes that d_twiddles are located in the device memory.
  * @param arr input array of type E (elements).
  * @param n length of d_arr.
  * @param twiddles twiddle factors of type S (scalars) array allocated on the device memory (must be a power of 2).
  * @param n_twiddles length of twiddles.
  * @param max_task max count of parallel tasks.
  * @param s log2(n) loop index.
  */
template <typename E, typename S>
__global__ void ntt_template_kernel_shared_rev(
  E* __restrict__ arr_g,
  uint32_t n,
  const S* __restrict__ r_twiddles,
  uint32_t n_twiddles,
  uint32_t max_task,
  uint32_t ss,
  uint32_t logn)
{
  SharedMemory<E> smem;
  E* arr = smem.getPointer();

  uint32_t task = blockIdx.x;
  uint32_t loop_limit = blockDim.x;
  uint32_t chunks = n / (loop_limit * 2);
  uint32_t offset = (task / chunks) * n;
  if (task < max_task) {
    // flattened loop allows parallel processing
    uint32_t l = threadIdx.x;

    if (l < loop_limit) {
#pragma unroll
      for (; ss < logn; ss++) {
        int s = logn - ss - 1;
        bool is_beginning = ss == 0;
        bool is_end = ss == (logn - 1);

        uint32_t ntw_i = task % chunks;

        uint32_t n_twiddles_div = n_twiddles >> (s + 1);

        uint32_t shift_s = 1 << s;
        uint32_t shift2_s = 1 << (s + 1);

        l = ntw_i * loop_limit + l; // to l from chunks to full

        uint32_t j = l & (shift_s - 1);               // Equivalent to: l % (1 << s)
        uint32_t i = ((l >> s) * shift2_s) & (n - 1); // (..) % n (assuming n is power of 2)
        uint32_t oij = i + j;
        uint32_t k = oij + shift_s;

        S tw = r_twiddles[j * n_twiddles_div];

        E u = is_beginning ? arr_g[offset + oij] : arr[oij];
        E v = is_beginning ? arr_g[offset + k] : arr[k];
        if (is_end) {
          arr_g[offset + oij] = u + v;
          arr_g[offset + k] = tw * (u - v);
        } else {
          arr[oij] = u + v;
          arr[k] = tw * (u - v);
        }

        __syncthreads();
      }
    }
  }
}

/**
  * Cooley-Tuckey NTT.
  * NOTE! this function assumes that d_twiddles are located in the device memory.
  * @param arr input array of type E (elements).
  * @param n length of d_arr.
  * @param twiddles twiddle factors of type S (scalars) array allocated on the device memory (must be a power of 2).
  * @param n_twiddles length of twiddles.
  * @param max_task max count of parallel tasks.
  * @param s log2(n) loop index.
  */
template <typename E, typename S>
__global__ void ntt_template_kernel_shared(
  E* __restrict__ arr_g,
  uint32_t n,
  const S* __restrict__ r_twiddles,
  uint32_t n_twiddles,
  uint32_t max_task,
  uint32_t s,
  uint32_t logn)
{
  SharedMemory<E> smem;
  E* arr = smem.getPointer();

  uint32_t task = blockIdx.x;
  uint32_t loop_limit = blockDim.x;
  uint32_t chunks = n / (loop_limit * 2);
  uint32_t offset = (task / chunks) * n;
  if (task < max_task) {
    // flattened loop allows parallel processing
    uint32_t l = threadIdx.x;

    if (l < loop_limit) {
#pragma unroll
      for (; s < logn; s++) // TODO: this loop also can be unrolled
      {
        uint32_t ntw_i = task % chunks;

        uint32_t n_twiddles_div = n_twiddles >> (s + 1);

        uint32_t shift_s = 1 << s;
        uint32_t shift2_s = 1 << (s + 1);

        l = ntw_i * loop_limit + l; // to l from chunks to full

        uint32_t j = l & (shift_s - 1);               // Equivalent to: l % (1 << s)
        uint32_t i = ((l >> s) * shift2_s) & (n - 1); // (..) % n (assuming n is power of 2)
        uint32_t oij = i + j;
        uint32_t k = oij + shift_s;
        S tw = r_twiddles[j * n_twiddles_div];

        E u = s == 0 ? arr_g[offset + oij] : arr[oij];
        E v = s == 0 ? arr_g[offset + k] : arr[k];
        v = tw * v;
        if (s == (logn - 1)) {
          arr_g[offset + oij] = u + v;
          arr_g[offset + k] = u - v;
        } else {
          arr[oij] = u + v;
          arr[k] = u - v;
        }

        __syncthreads();
      }
    }
  }
}

/**
  * Cooley-Tukey NTT.
  * NOTE! this function assumes that d_twiddles are located in the device memory.
  * @param arr input array of type E (elements).
  * @param n length of d_arr.
  * @param twiddles twiddle factors of type S (scalars) array allocated on the device memory (must be a power of 2).
  * @param n_twiddles length of twiddles.
  * @param max_task max count of parallel tasks.
  * @param s log2(n) loop index.
  */
template <typename E, typename S>
__global__ void
ntt_template_kernel(E* arr, uint32_t n, S* twiddles, uint32_t n_twiddles, uint32_t max_task, uint32_t s, bool rev)
{
  int task = blockIdx.x;
  int chunks = n / (blockDim.x * 2);

  if (task < max_task) {
    // flattened loop allows parallel processing
    uint32_t l = threadIdx.x;
    uint32_t loop_limit = blockDim.x;

    if (l < loop_limit) {
      uint32_t ntw_i = task % chunks;

      uint32_t shift_s = 1 << s;
      uint32_t shift2_s = 1 << (s + 1);
      uint32_t n_twiddles_div = n_twiddles >> (s + 1);

      l = ntw_i * blockDim.x + l; // to l from chunks to full

      uint32_t j = l & (shift_s - 1);               // Equivalent to: l % (1 << s)
      uint32_t i = ((l >> s) * shift2_s) & (n - 1); // (..) % n (assuming n is power of 2)
      uint32_t k = i + j + shift_s;

      S tw = twiddles[j * n_twiddles_div];

      uint32_t offset = (task / chunks) * n;
      E u = arr[offset + i + j];
      E v = arr[offset + k];
      if (!rev) v = tw * v;
      arr[offset + i + j] = u + v;
      v = u - v;
      arr[offset + k] = rev ? tw * v : v;
    }
  }
}

/**
  * NTT/INTT inplace batch
  * Note: this function does not preform any bit-reverse permutations on its inputs or outputs.
  * @param d_inout Array for inplace processing
  * @param d_twiddles
  * @param n Length of `d_twiddles` array
  * @param batch_size The size of the batch; the length of `d_inout` is `n` * `batch_size`.
  * @param inverse true for iNTT
  * @param is_coset true for multiplication by coset
  * @param coset should be array of lenght n - or in case of lesser than n, right-padded with zeroes
  * @param stream CUDA stream
  * @param is_sync_needed do perform sync of the supplied CUDA stream at the end of processing
  */
template <typename E, typename S>
void ntt_inplace_batch_template(
  E* d_inout,
  S* d_twiddles,
  unsigned n,
  unsigned batch_size,
  bool inverse,
  bool is_coset,
  S* coset,
  cudaStream_t stream,
  bool is_sync_needed)
{
  const int logn = int(log(n) / log(2));
  bool is_shared_mem_enabled = sizeof(E) <= MAX_SHARED_MEM_ELEMENT_SIZE;
  const int log2_shmem_elems = is_shared_mem_enabled ? int(log(int(MAX_SHARED_MEM / sizeof(E))) / log(2)) : logn;
  int num_threads = max(min(min(n / 2, MAX_THREADS_BATCH), 1 << (log2_shmem_elems - 1)), 1);
  const int chunks = max(int((n / 2) / num_threads), 1);
  const int total_tasks = batch_size * chunks;
  int num_blocks = total_tasks;
  const int shared_mem = 2 * num_threads * sizeof(E); // TODO: calculator, as shared mem size may be more efficient
                                                      // less then max to allow more concurrent blocks on SM
  const int logn_shmem = is_shared_mem_enabled ? int(log(2 * num_threads) / log(2))
                                                : 0; // TODO: shared memory support only for types <= 32 bytes

  if (inverse) {
    if (is_shared_mem_enabled)
      ntt_template_kernel_shared<<<num_blocks, num_threads, shared_mem, stream>>>(
        d_inout, 1 << logn_shmem, d_twiddles, n, total_tasks, 0, logn_shmem);

    for (int s = logn_shmem; s < logn; s++) // TODO: this loop also can be unrolled
    {
      ntt_template_kernel<E, S>
        <<<num_blocks, num_threads, 0, stream>>>(d_inout, n, d_twiddles, n, total_tasks, s, false);
    }

    if (is_coset)
      utils_internal::batchVectorMult<E, S><<<num_blocks, num_threads, 0, stream>>>(d_inout, coset, n, batch_size);

    num_threads = max(min(n / 2, MAX_NUM_THREADS), 1);
    num_blocks = (n * batch_size + num_threads - 1) / num_threads;
    utils_internal::template_normalize_kernel<E, S>
      <<<num_blocks, num_threads, 0, stream>>>(d_inout, S::inv_log_size(logn), n * batch_size);
  } else {
    if (is_coset)
      utils_internal::batchVectorMult<E, S><<<num_blocks, num_threads, 0, stream>>>(d_inout, coset, n, batch_size);

    for (int s = logn - 1; s >= logn_shmem; s--) // TODO: this loop also can be unrolled
    {
      ntt_template_kernel<<<num_blocks, num_threads, 0, stream>>>(d_inout, n, d_twiddles, n, total_tasks, s, true);
    }

    if (is_shared_mem_enabled)
      ntt_template_kernel_shared_rev<<<num_blocks, num_threads, shared_mem, stream>>>(
        d_inout, 1 << logn_shmem, d_twiddles, n, total_tasks, 0, logn_shmem);
  }

  if (!is_sync_needed) return;

  cudaStreamSynchronize(stream);
}

} // namespace

template <typename S>
cudaError_t GenerateTwiddleFactors(S* d_twiddles, int n_twiddles, S omega, device_context::DeviceContext ctx)
{
  twiddle_factors_kernel<S><<<1, 1, 0, ctx.stream>>>(d_twiddles, n_twiddles, omega);
  cudaStreamSynchronize(ctx.stream);
  return cudaSuccess;
}

template <typename E, typename S>
cudaError_t NTT(NTTConfig<E, S>* config)
{
  CHECK_LAST_CUDA_ERROR();

  cudaStream_t stream = config->ctx.stream;
  int size = config->size;
  int batch_size = config->batch_size;
  bool is_inverse = config->is_inverse;
  int n_twiddles = size;
  int logn = int(log(size) / log(2));
  int input_size_bytes = size * batch_size * sizeof(E);
  bool is_input_on_device = config->are_inputs_on_device;
  bool is_output_on_device = config->is_output_on_device;
  bool is_forward_twiddle_empty = config->twiddles == nullptr;
  bool is_inverse_twiddle_empty = config->inv_twiddles == nullptr;
  bool is_generating_twiddles = (is_forward_twiddle_empty && is_inverse_twiddle_empty) ||
                                (is_forward_twiddle_empty && !is_inverse) || (is_inverse_twiddle_empty && is_inverse);

  S* d_twiddles;
  if (is_generating_twiddles) {
    cudaMallocAsync(&d_twiddles, n_twiddles * sizeof(S), stream);
    S omega = is_inverse ? S::omega_inv(logn) : S::omega(logn);
    GenerateTwiddleFactors(d_twiddles, n_twiddles, omega, config->ctx);
  } else {
    d_twiddles = is_inverse ? config->inv_twiddles : config->twiddles;
  }

  E* d_inout;
  if (is_input_on_device) {
    d_inout = config->inout;
  } else {
    cudaMallocAsync(&d_inout, input_size_bytes, stream);
    cudaMemcpyAsync(d_inout, config->inout, input_size_bytes, cudaMemcpyHostToDevice, stream);
  }

  bool reverse_input;
  bool reverse_output;
  switch (config->ordering) {
  case Ordering::kNN:
    reverse_input = is_inverse;
    reverse_output = !is_inverse;
    break;
  case Ordering::kNR:
    reverse_input = is_inverse;
    reverse_output = is_inverse;
    break;
  case Ordering::kRN:
    reverse_input = !is_inverse;
    reverse_output = !is_inverse;
    break;
  case Ordering::kRR:
    reverse_input = !is_inverse;
    reverse_output = is_inverse;
    break;
  }
  CHECK_LAST_CUDA_ERROR();

  if (reverse_input) reverse_order_batch(d_inout, size, logn, config->batch_size, stream);
  CHECK_LAST_CUDA_ERROR();

  ntt_inplace_batch_template(
    d_inout, d_twiddles, size, batch_size, is_inverse, config->is_coset, config->coset_gen, stream, false);
  CHECK_LAST_CUDA_ERROR();

  if (reverse_output) reverse_order_batch(d_inout, size, logn, batch_size, stream);
  CHECK_LAST_CUDA_ERROR();

  if (is_output_on_device) {
    // free(config->inout); // TODO: ? or callback?+
    config->inout = d_inout;
  } else {
    if (is_input_on_device) {
      E* h_output = (E*)malloc(input_size_bytes); // TODO: caller responsible for memory management
      cudaMemcpyAsync(h_output, d_inout, input_size_bytes, cudaMemcpyDeviceToHost, stream);
      config->inout = h_output;
      CHECK_LAST_CUDA_ERROR();
    } else {
      cudaMemcpyAsync(config->inout, d_inout, input_size_bytes, cudaMemcpyDeviceToHost, stream);
      CHECK_LAST_CUDA_ERROR();
    }
    cudaFreeAsync(d_inout, stream); // TODO: make it optional? so can be reused
  }
  CHECK_LAST_CUDA_ERROR();

  if (is_generating_twiddles && !config->is_preserving_twiddles) { cudaFreeAsync(d_twiddles, stream); }

  if (config->is_preserving_twiddles) {
    if (is_inverse)
      config->inv_twiddles = d_twiddles;
    else {
      config->twiddles = d_twiddles;
    }
  }

  cudaStreamSynchronize(stream);

  CHECK_LAST_CUDA_ERROR();

  return cudaSuccess;
}

/**
  * Extern version of [ntt](@ref ntt) function with the following values of template parameters
  * (where the curve is given by `-DCURVE` env variable during build):
  *  - `S` and `E` are both the [scalar field](@ref scalar_t) of the curve;
  * @return `cudaSuccess` if the execution was successful and an error code otherwise.
  */
extern "C" cudaError_t NTTCuda(NTTConfig<curve_config::scalar_t, curve_config::scalar_t>* config)
{
  return NTT<curve_config::scalar_t, curve_config::scalar_t>(config);
}

/**
  * Extern version of [ntt](@ref ntt) function with the following values of template parameters
  * (where the curve is given by `-DCURVE` env variable during build):
  *  - `S` and `E` are both the [scalar field](@ref scalar_t) of the curve;
  * @return `cudaSuccess` if the execution was successful and an error code otherwise.
  */
template <typename E, typename S>
cudaError_t NTTDefaultContext(NTTConfig<E, S>* config)
{
  // TODO: if empty - create default
  cudaMemPool_t mempool;
  cudaDeviceGetDefaultMemPool(&mempool, config->ctx.device_id);

  device_context::DeviceContext context = {
    config->ctx.device_id,
    0, // default stream
    mempool};

  config->ctx = context;

  return NTT<E, S>(config);
}

/**
  * Extern version of [ntt](@ref ntt) function with the following values of template parameters
  * (where the curve is given by `-DCURVE` env variable during build):
  *  - `S` and `E` are both the [scalar field](@ref scalar_t) of the curve;
  * @return `cudaSuccess` if the execution was successful and an error code otherwise.
  */
extern "C" cudaError_t NTTDefaultContextCuda(NTTConfig<curve_config::scalar_t, curve_config::scalar_t>* config)
{
  return NTTDefaultContext(config);
}

#if defined(ECNTT_DEFINED)

/**
  * Extern version of [NTT](@ref NTT) function with the following values of template parameters
  * (where the curve is given by `-DCURVE` env variable during build):
  *  - `S` is the [projective representation](@ref projective_t) of the curve (i.e. EC NTT is computed);
  *  - `E` is the [scalar field](@ref scalar_t) of the curve;
  * @return `cudaSuccess` if the execution was successful and an error code otherwise.
  */
extern "C" cudaError_t ECNTTCuda(NTTConfig<curve_config::projective_t, curve_config::scalar_t>* config)
{
  return NTT<curve_config::projective_t, curve_config::scalar_t>(config);
}

#endif

} // namespace ntt
