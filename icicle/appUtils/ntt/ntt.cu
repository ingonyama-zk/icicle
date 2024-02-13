#include "ntt.cuh"

#include <unordered_map>
#include <vector>

#include "curves/curve_config.cuh"
#include "utils/sharedmem.cuh"
#include "utils/utils_kernels.cuh"
#include "utils/utils.h"
#include "appUtils/ntt/ntt_impl.cuh"

namespace ntt {

  namespace {

    const uint32_t MAX_NUM_THREADS = 512;   // TODO: hotfix - should be 1024, currently limits shared memory size
    const uint32_t MAX_THREADS_BATCH = 512; // TODO: allows 100% occupancy for scalar NTT for sm_86..sm_89
    const uint32_t MAX_SHARED_MEM_ELEMENT_SIZE = 32; // TODO: occupancy calculator, hardcoded for sm_86..sm_89
    const uint32_t MAX_SHARED_MEM = MAX_SHARED_MEM_ELEMENT_SIZE * MAX_NUM_THREADS;

    template <typename E>
    __global__ void reverse_order_kernel(E* arr, E* arr_reversed, uint32_t n, uint32_t logn, uint32_t batch_size)
    {
      int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
      if (threadId < n * batch_size) {
        int idx = threadId % n;
        int batch_idx = threadId / n;
        int idx_reversed = __brev(idx) >> (32 - logn);

        E val = arr[batch_idx * n + idx];
        if (arr == arr_reversed) { __syncthreads(); } // for in-place (when pointers arr==arr_reversed)
        arr_reversed[batch_idx * n + idx_reversed] = val;
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
     * @param n_twiddles length of twiddles, should be negative for intt.
     * @param max_task max count of parallel tasks.
     * @param s log2(n) loop index.
     * @param arr_out buffer for the output.
     */
    template <typename E, typename S>
    __global__ void ntt_template_kernel_shared_rev(
      E* __restrict__ arr_in,
      int n,
      const S* __restrict__ r_twiddles,
      int n_twiddles,
      int max_task,
      int ss,
      int logn,
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

            S tw = *(r_twiddles + (int)(j * n_twiddles_div));

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
     * @param n_twiddles length of twiddles, should be negative for intt.
     * @param max_task max count of parallel tasks.
     * @param s log2(n) loop index.
     * @param arr_out buffer for the output.
     */
    template <typename E, typename S>
    __global__ void ntt_template_kernel_shared(
      E* __restrict__ arr_in,
      int n,
      const S* __restrict__ r_twiddles,
      int n_twiddles,
      int max_task,
      int s,
      int logn,
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
            S tw = *(r_twiddles + (int)(j * n_twiddles_div));

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
     * @param n_twiddles length of twiddles, should be negative for intt.
     * @param max_task max count of parallel tasks.
     * @param s log2(n) loop index.
     */
    template <typename E, typename S>
    __global__ void
    ntt_template_kernel(E* arr_in, int n, S* twiddles, int n_twiddles, int max_task, int s, bool rev, E* arr_out)
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

          S tw = *(twiddles + (int)(j * n_twiddles_div));

          uint32_t offset = (task / chunks) * n;
          E u = arr_in[offset + i + j];
          E v = arr_in[offset + k];
          if (!rev) v = tw * v;
          arr_out[offset + i + j] = u + v;
          v = u - v;
          arr_out[offset + k] = rev ? tw * v : v;
        }
      }
    }

    /**
     * NTT/INTT inplace batch
     * Note: this function does not perform any bit-reverse permutations on its inputs or outputs.
     * @param d_input Input array
     * @param n Size of `d_input`
     * @param d_twiddles Twiddles
     * @param n_twiddles Size of `d_twiddles`
     * @param batch_size The size of the batch; the length of `d_inout` is `n` * `batch_size`.
     * @param inverse true for iNTT
     * @param coset should be array of length n or a nullptr if NTT is not computed on a coset
     * @param stream CUDA stream
     * @param is_async if false, perform sync of the supplied CUDA stream at the end of processing
     * @param d_output Output array
     */
    template <typename E, typename S>
    cudaError_t ntt_inplace_batch_template(
      E* d_input,
      int n,
      S* d_twiddles,
      int n_twiddles,
      int batch_size,
      int logn,
      bool inverse,
      bool ct_buttterfly,
      S* arbitrary_coset,
      int coset_gen_index,
      cudaStream_t stream,
      E* d_output)
    {
      CHK_INIT_IF_RETURN();

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
      int num_threads_coset = max(min(n / 2, MAX_NUM_THREADS), 1);
      int num_blocks_coset = (n * batch_size + num_threads_coset - 1) / num_threads_coset;

      if (inverse) {
        d_twiddles = d_twiddles + n_twiddles;
        n_twiddles = -n_twiddles;
      }

      bool is_on_coset = (coset_gen_index != 0) || arbitrary_coset;
      bool direct_coset = (!inverse && is_on_coset);
      if (direct_coset)
        utils_internal::BatchMulKernel<E, S><<<num_blocks_coset, num_threads_coset, 0, stream>>>(
          d_input, n, batch_size, arbitrary_coset ? arbitrary_coset : d_twiddles, arbitrary_coset ? 1 : coset_gen_index,
          n_twiddles, logn, ct_buttterfly, d_output);

      if (ct_buttterfly) {
        if (is_shared_mem_enabled)
          ntt_template_kernel_shared<<<num_blocks, num_threads, shared_mem, stream>>>(
            direct_coset ? d_output : d_input, 1 << logn_shmem, d_twiddles, n_twiddles, total_tasks, 0, logn_shmem,
            d_output);

        for (int s = logn_shmem; s < logn; s++) // TODO: this loop also can be unrolled
        {
          ntt_template_kernel<E, S><<<num_blocks, num_threads, 0, stream>>>(
            (direct_coset || (s > 0)) ? d_output : d_input, n, d_twiddles, n_twiddles, total_tasks, s, false, d_output);
        }
      } else {
        for (int s = logn - 1; s >= logn_shmem; s--) // TODO: this loop also can be unrolled
        {
          ntt_template_kernel<<<num_blocks, num_threads, 0, stream>>>(
            (direct_coset || (s < logn - 1)) ? d_output : d_input, n, d_twiddles, n_twiddles, total_tasks, s, true,
            d_output);
        }

        if (is_shared_mem_enabled)
          ntt_template_kernel_shared_rev<<<num_blocks, num_threads, shared_mem, stream>>>(
            (direct_coset || (logn > logn_shmem)) ? d_output : d_input, 1 << logn_shmem, d_twiddles, n_twiddles,
            total_tasks, 0, logn_shmem, d_output);
      }

      if (inverse) {
        if (is_on_coset)
          utils_internal::BatchMulKernel<E, S><<<num_blocks_coset, num_threads_coset, 0, stream>>>(
            d_output, n, batch_size, arbitrary_coset ? arbitrary_coset : d_twiddles,
            arbitrary_coset ? 1 : -coset_gen_index, -n_twiddles, logn, !ct_buttterfly, d_output);

        utils_internal::NormalizeKernel<E, S>
          <<<num_blocks_coset, num_threads_coset, 0, stream>>>(d_output, S::inv_log_size(logn), n * batch_size);
      }

      return CHK_LAST();
    }

  } // namespace

  /**
   * @struct Domain
   * Struct containing information about the domain on which (i)NTT is evaluated i.e. twiddle factors.
   * Twiddle factors are private, static and can only be set using [InitDomain](@ref InitDomain) function.
   * The internal representation of twiddles is prone to change in accordance with changing [NTT](@ref NTT) algorithm.
   * @tparam S The type of twiddle factors \f$ \{ \omega^i \} \f$. Must be a field.
   */
  template <typename S>
  class Domain
  {
    static inline int max_size = 0;
    static inline int max_log_size = 0;
    static inline S* twiddles = nullptr;
    static inline std::unordered_map<S, int> coset_index = {};

    static inline S* internal_twiddles = nullptr; // required by mixed-radix NTT
    static inline S* basic_twiddles = nullptr;    // required by mixed-radix NTT

  public:
    template <typename U>
    friend cudaError_t InitDomain<U>(U primitive_root, device_context::DeviceContext& ctx);

    static cudaError_t ReleaseDomain(device_context::DeviceContext& ctx);

    template <typename U, typename E>
    friend cudaError_t NTT<U, E>(E* input, int size, NTTDir dir, NTTConfig<U>& config, E* output);
  };

  template <typename S>
  cudaError_t InitDomain(S primitive_root, device_context::DeviceContext& ctx)
  {
    CHK_INIT_IF_RETURN();

    // only generate twiddles if they haven't been generated yet
    // please note that this is not thread-safe at all,
    // but it's a singleton that is supposed to be initialized once per device per program lifetime
    if (!Domain<S>::twiddles) {
      bool found_logn = false;
      S omega = primitive_root;
      unsigned omegas_count = S::get_omegas_count();
      for (int i = 0; i < omegas_count; i++) {
        omega = S::sqr(omega);
        if (!found_logn) {
          ++Domain<S>::max_log_size;
          found_logn = omega == S::one();
          if (found_logn) break;
        }
      }
      Domain<S>::max_size = (int)pow(2, Domain<S>::max_log_size);
      if (omega != S::one()) {
        throw IcicleError(
          IcicleError_t::InvalidArgument, "Primitive root provided to the InitDomain function is not in the subgroup");
      }

      // allocate and calculate twiddles on GPU
      // Note: radix-2 INTT needs ONE in last element (in addition to first element), therefore have n+1 elements
      // Managed allocation allows host to read the elements (logn) without copying all (n) TFs back to host
      CHK_IF_RETURN(cudaMallocManaged(&Domain<S>::twiddles, (Domain<S>::max_size + 1) * sizeof(S)));
      CHK_IF_RETURN(generate_external_twiddles_generic(
        primitive_root, Domain<S>::twiddles, Domain<S>::internal_twiddles, Domain<S>::basic_twiddles,
        Domain<S>::max_log_size, ctx.stream));
      CHK_IF_RETURN(cudaStreamSynchronize(ctx.stream));

      const bool is_map_only_powers_of_primitive_root = true;
      if (is_map_only_powers_of_primitive_root) {
        // populate the coset_index map. Note that only powers of the primitive-root are stored (1, PR, PR^2, PR^4, PR^8
        // etc.)
        Domain<S>::coset_index[S::one()] = 0;
        for (int i = 0; i < Domain<S>::max_log_size; ++i) {
          const int index = (int)pow(2, i);
          Domain<S>::coset_index[Domain<S>::twiddles[index]] = index;
        }
      } else {
        // populate all values
        for (int i = 0; i < Domain<S>::max_size; ++i) {
          Domain<S>::coset_index[Domain<S>::twiddles[i]] = i;
        }
      }
    }

    return CHK_LAST();
  }

  template <typename S>
  cudaError_t Domain<S>::ReleaseDomain(device_context::DeviceContext& ctx)
  {
    CHK_INIT_IF_RETURN();

    max_size = 0;
    max_log_size = 0;
    cudaFreeAsync(twiddles, ctx.stream);
    twiddles = nullptr;
    cudaFreeAsync(internal_twiddles, ctx.stream);
    internal_twiddles = nullptr;
    cudaFreeAsync(basic_twiddles, ctx.stream);
    basic_twiddles = nullptr;
    coset_index.clear();

    return CHK_LAST();
  }

  template <typename S, typename E>
  cudaError_t NTT(E* input, int size, NTTDir dir, NTTConfig<S>& config, E* output)
  {
    CHK_INIT_IF_RETURN();
    if (size > Domain<S>::max_size) { // TODO: proper error handling
      std::cerr
        << "NTT size is too large for the domain. Consider generating your domain with a higher order root of unity"
        << '\n';
      throw -1;
    }

    cudaStream_t& stream = config.ctx.stream;
    int batch_size = config.batch_size;
    int logn = int(log(size) / log(2));
    int input_size_bytes = size * batch_size * sizeof(E);
    bool are_inputs_on_device = config.are_inputs_on_device;
    bool are_outputs_on_device = config.are_outputs_on_device;

    E* d_input;
    if (are_inputs_on_device) {
      d_input = input;
    } else {
      CHK_IF_RETURN(cudaMallocAsync(&d_input, input_size_bytes, stream));
      CHK_IF_RETURN(cudaMemcpyAsync(d_input, input, input_size_bytes, cudaMemcpyHostToDevice, stream));
    }
    E* d_output;
    if (are_outputs_on_device) {
      d_output = output;
    } else {
      CHK_IF_RETURN(cudaMallocAsync(&d_output, input_size_bytes, stream));
    }

    S* coset = nullptr;
    int coset_index = 0;
    try {
      coset_index = Domain<S>::coset_index.at(config.coset_gen);
    } catch (...) {
      // if coset index is not found in the subgroup, compute coset powers on CPU and move them to device
      std::vector<S> h_coset;
      h_coset.push_back(S::one());
      S coset_gen = (dir == NTTDir::kInverse) ? S::inverse(config.coset_gen) : config.coset_gen;
      for (int i = 1; i < size; i++) {
        h_coset.push_back(h_coset.at(i - 1) * coset_gen);
      }
      CHK_IF_RETURN(cudaMallocAsync(&coset, size * sizeof(S), stream));
      CHK_IF_RETURN(cudaMemcpyAsync(coset, &h_coset.front(), size * sizeof(S), cudaMemcpyHostToDevice, stream));
      h_coset.clear();
    }

    const bool is_small_ntt = logn < 16;                  // cutoff point where mixed-radix is faster than radix-2
    const bool is_on_coset = (coset_index != 0) || coset; // coset not supported by mixed-radix algorithm yet
    const bool is_batch_ntt = batch_size > 1;             // batch not supported by mixed-radidx algorithm yet
    const bool is_NN = config.ordering == Ordering::kNN;  // TODO Yuval: relax this limitation
    const bool is_radix2_algorithm = config.is_force_radix2 || is_batch_ntt || is_small_ntt || is_on_coset || !is_NN;

    if (is_radix2_algorithm) {
      bool ct_butterfly = true;
      bool reverse_input = false;
      switch (config.ordering) {
      case Ordering::kNN:
        reverse_input = true;
        break;
      case Ordering::kNR:
        ct_butterfly = false;
        break;
      case Ordering::kRR:
        reverse_input = true;
        ct_butterfly = false;
        break;
      }

      if (reverse_input) reverse_order_batch(d_input, size, logn, batch_size, stream, d_output);

      CHK_IF_RETURN(ntt_inplace_batch_template(
        reverse_input ? d_output : d_input, size, Domain<S>::twiddles, Domain<S>::max_size, batch_size, logn,
        dir == NTTDir::kInverse, ct_butterfly, coset, coset_index, stream, d_output));

      if (coset) CHK_IF_RETURN(cudaFreeAsync(coset, stream));
    } else { // mixed-radix algorithm
      CHK_IF_RETURN(ntt::mixed_radix_ntt(
        d_input, d_output, Domain<S>::twiddles, Domain<S>::internal_twiddles, Domain<S>::basic_twiddles, size,
        Domain<S>::max_log_size, dir == NTTDir::kInverse, config.ordering, stream));
    }

    if (!are_outputs_on_device)
      CHK_IF_RETURN(cudaMemcpyAsync(output, d_output, input_size_bytes, cudaMemcpyDeviceToHost, stream));

    if (!are_inputs_on_device) CHK_IF_RETURN(cudaFreeAsync(d_input, stream));
    if (!are_outputs_on_device) CHK_IF_RETURN(cudaFreeAsync(d_output, stream));
    if (!config.is_async) return CHK_STICKY(cudaStreamSynchronize(stream));

    return CHK_LAST();
  }

  template <typename S>
  NTTConfig<S> DefaultNTTConfig()
  {
    device_context::DeviceContext ctx = device_context::get_default_device_context();
    NTTConfig<S> config = {
      ctx,           // ctx
      S::one(),      // coset_gen
      1,             // batch_size
      Ordering::kNN, // ordering
      false,         // are_inputs_on_device
      false,         // are_outputs_on_device
      false,         // is_async
      false,         // is_force_radix2
    };
    return config;
  }

  /**
   * Extern "C" version of [InitDomain](@ref InitDomain) function with the following
   * value of template parameter (where the curve is given by `-DCURVE` env variable during build):
   *  - `S` is the [scalar field](@ref scalar_t) of the curve;
   */
  extern "C" cudaError_t
    CONCAT_EXPAND(CURVE, InitializeDomain)(curve_config::scalar_t primitive_root, device_context::DeviceContext& ctx)
  {
    return InitDomain(primitive_root, ctx);
  }

  /**
   * Extern "C" version of [NTT](@ref NTT) function with the following values of template parameters
   * (where the curve is given by `-DCURVE` env variable during build):
   *  - `S` and `E` are both the [scalar field](@ref scalar_t) of the curve;
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(CURVE, NTTCuda)(
    curve_config::scalar_t* input,
    int size,
    NTTDir dir,
    NTTConfig<curve_config::scalar_t>& config,
    curve_config::scalar_t* output)
  {
    return NTT<curve_config::scalar_t, curve_config::scalar_t>(input, size, dir, config, output);
  }

#if defined(ECNTT_DEFINED)

  /**
   * Extern "C" version of [NTT](@ref NTT) function with the following values of template parameters
   * (where the curve is given by `-DCURVE` env variable during build):
   *  - `S` is the [projective representation](@ref projective_t) of the curve (i.e. EC NTT is computed);
   *  - `E` is the [scalar field](@ref scalar_t) of the curve;
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(CURVE, ECNTTCuda)(
    curve_config::projective_t* input,
    int size,
    NTTDir dir,
    NTTConfig<curve_config::scalar_t>& config,
    curve_config::projective_t* output)
  {
    return NTT<curve_config::scalar_t, curve_config::projective_t>(input, size, dir, config, output);
  }

#endif

} // namespace ntt