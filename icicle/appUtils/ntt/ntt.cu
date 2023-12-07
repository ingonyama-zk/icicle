#include "ntt.cuh"

#include <vector>

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
     * Bit-reverses a batch of input arrays out-of-place inside GPU.
     * for example: on input array ([a[0],a[1],a[2],a[3]], 4, 2) it returns
     * [a[0],a[3],a[2],a[1]] (elements at indices 3 and 1 swhich places).
     * @param arr_in batch of arrays of some object of type T. Should be on GPU.
     * @param n length of `arr`.
     * @param logn log(n).
     * @param batch_size the size of the batch.
     * @param arr_out buffer of the same size as `arr_in` on the GPU to write the bit-permuted array into.
     */
    template <typename E>
    void reverse_order_batch(E* arr_in, uint32_t n, uint32_t logn, uint32_t batch_size, cudaStream_t stream, E* arr_out)
    {
      int number_of_threads = MAX_THREADS_BATCH;
      int number_of_blocks = (n * batch_size + number_of_threads - 1) / number_of_threads;
      reverse_order_kernel<<<number_of_blocks, number_of_threads, 0, stream>>>(arr_in, arr_out, n, logn, batch_size);
    }

    /**
     * Bit-reverses an input array out-of-place inside GPU.
     * for example: on array ([a[0],a[1],a[2],a[3]], 4, 2) it returns
     * [a[0],a[3],a[2],a[1]] (elements at indices 3 and 1 swhich places).
     * @param arr_in array of some object of type T of size which is a power of 2. Should be on GPU.
     * @param n length of `arr`.
     * @param logn log(n).
     * @param arr_out buffer of the same size as `arr_in` on the GPU to write the bit-permuted array into.
     */
    template <typename E>
    void reverse_order(E* arr_in, uint32_t n, uint32_t logn, cudaStream_t stream, E* arr_out)
    {
      reverse_order_batch(arr_in, n, logn, 1, stream, arr_out);
    }

    /**
     * Cooley-Tuckey NTT.
     * NOTE! this function assumes that d_twiddles are located in the device memory.
     * @param arr_in input array of type E (elements).
     * @param n length of d_arr.
     * @param twiddles twiddle factors of type S (scalars) array allocated on the device memory (must be a power of 2).
     * @param n_twiddles length of twiddles.
     * @param max_task max count of parallel tasks.
     * @param s log2(n) loop index.
     * @param arr_out buffer for the output.
     */
    template <typename E, typename S>
    __global__ void ntt_template_kernel_shared_rev(
      E* __restrict__ arr_in,
      uint32_t n,
      const S* __restrict__ r_twiddles,
      uint32_t n_twiddles,
      uint32_t max_task,
      uint32_t ss,
      uint32_t logn,
      E* __restrict__ arr_out)
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

            E u = is_beginning ? arr_in[offset + oij] : arr[oij];
            E v = is_beginning ? arr_in[offset + k] : arr[k];
            if (is_end) {
              arr_out[offset + oij] = u + v;
              arr_out[offset + k] = tw * (u - v);
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
     * @param arr_in input array of type E (elements).
     * @param n length of d_arr.
     * @param twiddles twiddle factors of type S (scalars) array allocated on the device memory (must be a power of 2).
     * @param n_twiddles length of twiddles.
     * @param max_task max count of parallel tasks.
     * @param s log2(n) loop index.
     * @param arr_out buffer for the output.
     */
    template <typename E, typename S>
    __global__ void ntt_template_kernel_shared(
      E* __restrict__ arr_in,
      uint32_t n,
      const S* __restrict__ r_twiddles,
      uint32_t n_twiddles,
      uint32_t max_task,
      uint32_t s,
      uint32_t logn,
      E* __restrict__ arr_out)
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

            E u = s == 0 ? arr_in[offset + oij] : arr[oij];
            E v = s == 0 ? arr_in[offset + k] : arr[k];
            v = tw * v;
            if (s == (logn - 1)) {
              arr_out[offset + oij] = u + v;
              arr_out[offset + k] = u - v;
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
     * @param d_input Input array
     * @param d_twiddles Twiddles
     * @param n Length of `d_twiddles` array
     * @param batch_size The size of the batch; the length of `d_inout` is `n` * `batch_size`.
     * @param inverse true for iNTT
     * @param is_coset true for multiplication by coset
     * @param coset should be array of lenght n - or in case of lesser than n, right-padded with zeroes
     * @param stream CUDA stream
     * @param is_async if false, perform sync of the supplied CUDA stream at the end of processing
     * @param d_output Output array
     */
    template <typename E, typename S>
    void ntt_inplace_batch_template(
      E* d_input,
      S* d_twiddles,
      unsigned n,
      unsigned batch_size,
      bool inverse,
      bool is_coset,
      S* coset,
      cudaStream_t stream,
      bool is_async,
      E* d_output)
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
            d_input, 1 << logn_shmem, d_twiddles, n, total_tasks, 0, logn_shmem, d_output);

        for (int s = logn_shmem; s < logn; s++) // TODO: this loop also can be unrolled
        {
          ntt_template_kernel<E, S><<<num_blocks, num_threads, 0, stream>>>(
            (s == 0) ? d_input : d_output, n, d_twiddles, n, total_tasks, s, false, d_output);
        }

        if (is_coset)
          utils_internal::BatchMulKernel<E, S>
            <<<num_blocks, num_threads, 0, stream>>>(d_output, coset, n, batch_size, d_output);

        num_threads = max(min(n / 2, MAX_NUM_THREADS), 1);
        num_blocks = (n * batch_size + num_threads - 1) / num_threads;
        utils_internal::NormalizeKernel<E, S>
          <<<num_blocks, num_threads, 0, stream>>>(d_output, S::inv_log_size(logn), n * batch_size);
      } else {
        if (is_coset)
          utils_internal::BatchMulKernel<E, S>
            <<<num_blocks, num_threads, 0, stream>>>(d_input, coset, n, batch_size, d_output);

        for (int s = logn - 1; s >= logn_shmem; s--) // TODO: this loop also can be unrolled
        {
          ntt_template_kernel<<<num_blocks, num_threads, 0, stream>>>(
            is_coset ? d_output : d_input, n, d_twiddles, n, total_tasks, s, true, d_output);
        }

        if (is_shared_mem_enabled)
          ntt_template_kernel_shared_rev<<<num_blocks, num_threads, shared_mem, stream>>>(
            (is_coset || (logn > logn_shmem)) ? d_output : d_input, 1 << logn_shmem, d_twiddles, n, total_tasks, 0,
            logn_shmem, d_output);
      }

      if (is_async) return;

      cudaStreamSynchronize(stream);
    }

  } // namespace

  template <typename S>
  cudaError_t GenerateDomain(S primitive_root, device_context::DeviceContext& ctx, Domain<S>& domain)
  {
    S inv_primitive_root = S::inverse(primitive_root);
    std::vector<S> twiddles;
    twiddles.push_back(S::one());
    std::vector<S> inv_twiddles;
    inv_twiddles.push_back(S::one());
    int n = 1;
    do {
      twiddles.push_back(twiddles.at(n - 1) * primitive_root);
      inv_twiddles.push_back(inv_twiddles.at(n - 1) * inv_primitive_root);
    } while (twiddles.at(n++) != S::one);
    cudaMemcpyAsync(domain.twiddles, &twiddles.front(), cudaMemcpyHostToDevice, ctx.stream);
    cudaMemcpyAsync(domain.inv_twiddles, &inv_twiddles.front(), cudaMemcpyHostToDevice, ctx.stream);
    domain.max_size = n;
    domain.log_max_size = int(log(n) / log(2));
    domain.coset_gen = S::one();
    return cudaSuccess;
  }

  template <typename E, typename S>
  cudaError_t NTT(E* input, int size, bool is_inverse, Domain<S>& domain, NTTConfig& config, E* output)
  {
    CHECK_LAST_CUDA_ERROR();

    cudaStream_t stream = config.ctx.stream;
    int batch_size = config.batch_size;
    int n_twiddles = size;
    int logn = int(log(size) / log(2));
    int input_size_bytes = size * batch_size * sizeof(E);
    bool is_input_on_device = config.are_inputs_on_device;
    bool is_output_on_device = config.are_outputs_on_device;
    S* d_twiddles = is_inverse ? domain.inv_twiddles : domain.twiddles;

    E* d_input;
    if (is_input_on_device) {
      d_input = input;
    } else {
      cudaMallocAsync(&d_input, input_size_bytes, stream);
      cudaMemcpyAsync(d_input, input, input_size_bytes, cudaMemcpyHostToDevice, stream);
    }
    E* d_output;
    if (is_input_on_device) {
      d_output = output;
    } else {
      cudaMallocAsync(&d_output, input_size_bytes, stream);
    }

    bool reverse_input;
    bool reverse_output;
    switch (config.ordering) {
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

    if (reverse_input) reverse_order_batch(d_input, size, logn, batch_size, stream, d_output);
    CHECK_LAST_CUDA_ERROR();

    ntt_inplace_batch_template(
      reverse_input ? d_output : d_input, d_twiddles, size, batch_size, is_inverse, domain.coset_gen != S::one(),
      domain.coset_gen, stream, !config.is_async, reverse_output ? d_input : d_output);
    CHECK_LAST_CUDA_ERROR();

    // it's assumed that reverse_input and reverse_output can't both be true at the same time
    // which should be guaranteed by Ordering
    if (reverse_output) reverse_order_batch(d_input, size, logn, batch_size, stream, d_output);
    CHECK_LAST_CUDA_ERROR();

    if (is_output_on_device) {
      // free(config->inout); // TODO: ? or callback?+
      output = d_output;
    } else {
      cudaMemcpyAsync(output, d_output, input_size_bytes, cudaMemcpyDeviceToHost, stream);
      CHECK_LAST_CUDA_ERROR();
    }
    CHECK_LAST_CUDA_ERROR();

    if (!config.is_async) cudaStreamSynchronize(stream);

    CHECK_LAST_CUDA_ERROR();

    return cudaSuccess;
  }

//   /**
//    * Extern "C" version of [NTT](@ref NTT) function with the following values of template parameters
//    * (where the curve is given by `-DCURVE` env variable during build):
//    *  - `S` and `E` are both the [scalar field](@ref scalar_t) of the curve;
//    * @return `cudaSuccess` if the execution was successful and an error code otherwise.
//    */
//   extern "C" cudaError_t NTTCuda(NTTConfig<curve_config::scalar_t, curve_config::scalar_t>* config)
//   {
//     return NTT<curve_config::scalar_t, curve_config::scalar_t>(config);
//   }

//   /**
//    * Extern "C" version of [NTT](@ref NTT) function with the following values of template parameters
//    * (where the curve is given by `-DCURVE` env variable during build):
//    *  - `S` and `E` are both the [scalar field](@ref scalar_t) of the curve;
//    * @return `cudaSuccess` if the execution was successful and an error code otherwise.
//    */
//   template <typename E, typename S>
//   cudaError_t NTTDefaultContext(NTTConfig<E, S>* config)
//   {
//     // TODO: if empty - create default
//     cudaMemPool_t mempool;
//     cudaDeviceGetDefaultMemPool(&mempool, config->ctx.device_id);

//     device_context::DeviceContext context = {
//       config->ctx.device_id,
//       0, // default stream
//       mempool};

//     config->ctx = context;

//     return NTT<E, S>(config);
//   }

//   /**
//    * Extern "C" version of [NTT](@ref NTT) function with the following values of template parameters
//    * (where the curve is given by `-DCURVE` env variable during build):
//    *  - `S` and `E` are both the [scalar field](@ref scalar_t) of the curve;
//    * @return `cudaSuccess` if the execution was successful and an error code otherwise.
//    */
//   extern "C" cudaError_t NTTDefaultContextCuda(NTTConfig<curve_config::scalar_t, curve_config::scalar_t>* config)
//   {
//     return NTTDefaultContext(config);
//   }

// #if defined(ECNTT_DEFINED)

//   /**
//    * Extern "C" version of [NTT](@ref NTT) function with the following values of template parameters
//    * (where the curve is given by `-DCURVE` env variable during build):
//    *  - `S` is the [projective representation](@ref projective_t) of the curve (i.e. EC NTT is computed);
//    *  - `E` is the [scalar field](@ref scalar_t) of the curve;
//    * @return `cudaSuccess` if the execution was successful and an error code otherwise.
//    */
//   extern "C" cudaError_t ECNTTCuda(NTTConfig<curve_config::projective_t, curve_config::scalar_t>* config)
//   {
//     return NTT<curve_config::projective_t, curve_config::scalar_t>(config);
//   }

// #endif

} // namespace ntt