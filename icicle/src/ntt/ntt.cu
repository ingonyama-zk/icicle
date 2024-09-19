#include "fields/field_config.cuh"

using namespace field_config;

#include "ntt/ntt.cuh"

#include <unordered_map>
#include <vector>
#include <type_traits>

#include "gpu-utils/sharedmem.cuh"
#include "utils/utils_kernels.cuh"
#include "utils/utils.h"
#include "ntt/ntt_impl.cuh"

#include <mutex>

#ifdef CURVE_ID
#include "curves/curve_config.cuh"
using namespace curve_config;
#define IS_ECNTT std::is_same_v<E, projective_t>
#else
#define IS_ECNTT false
#endif

namespace ntt {

  namespace {
    // TODO: Set MAX THREADS based on GPU arch
    const uint32_t MAX_NUM_THREADS = 512; // TODO: hotfix - should be 1024, currently limits shared memory size
    const uint32_t MAX_THREADS_BATCH = 512;
    const uint32_t MAX_THREADS_BATCH_ECNTT =
      128; // TODO: hardcoded - allows (2^18 x 64) ECNTT for sm86, decrease this to allow larger ecntt length, batch
           // size limited by on-device memory
    const uint32_t MAX_SHARED_MEM_ELEMENT_SIZE = 32; // TODO: occupancy calculator, hardcoded for sm_86..sm_89
    const uint32_t MAX_SHARED_MEM = MAX_SHARED_MEM_ELEMENT_SIZE * MAX_NUM_THREADS;

    template <typename E>
    __global__ void reverse_order_kernel(const E* arr, E* arr_reversed, uint32_t n, uint32_t logn, uint32_t batch_size)
    {
      int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
      if (threadId < n * batch_size) {
        int idx = threadId % n;
        int batch_idx = threadId / n;
        int idx_reversed = __brev(idx) >> (32 - logn);

        if (arr == arr_reversed) { // for in-place (when pointers arr==arr_reversed)
          if (idx < idx_reversed) {
            E val = arr[batch_idx * n + idx];
            arr_reversed[batch_idx * n + idx] = arr[batch_idx * n + idx_reversed];
            arr_reversed[batch_idx * n + idx_reversed] = val;
          }
        } else
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
    void reverse_order_batch(
      const E* arr_in, uint32_t n, uint32_t logn, uint32_t batch_size, cudaStream_t stream, E* arr_out)
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
    void reverse_order(const E* arr_in, uint32_t n, uint32_t logn, cudaStream_t stream, E* arr_out)
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
      const E* __restrict__ arr_in,
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
          UNROLL
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
      const E* __restrict__ arr_in,
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
          UNROLL
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
    ntt_template_kernel(const E* arr_in, int n, S* twiddles, int n_twiddles, int max_task, int s, bool rev, E* arr_out)
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
      const E* d_input,
      int n,
      S* d_twiddles,
      int n_twiddles,
      int batch_size,
      int logn,
      bool inverse,
      bool dit,
      S* arbitrary_coset,
      int coset_gen_index,
      cudaStream_t stream,
      E* d_output)
    {
      CHK_INIT_IF_RETURN();

      bool is_shared_mem_enabled = sizeof(E) <= MAX_SHARED_MEM_ELEMENT_SIZE;
      const int log2_shmem_elems = is_shared_mem_enabled ? int(log(int(MAX_SHARED_MEM / sizeof(E))) / log(2)) : logn;
      int max_threads_batch = IS_ECNTT ? MAX_THREADS_BATCH_ECNTT : MAX_THREADS_BATCH;
      int num_threads = max(min(min(n / 2, max_threads_batch), 1 << (log2_shmem_elems - 1)), 1);
      const int chunks = max(int((n / 2) / num_threads), 1);
      const int total_tasks = batch_size * chunks;
      int num_blocks = total_tasks;
      const int shared_mem = 2 * num_threads * sizeof(E); // TODO: calculator, as shared mem size may be more efficient
                                                          // less then max to allow more concurrent blocks on SM
      const int logn_shmem = is_shared_mem_enabled ? int(log(2 * num_threads) / log(2))
                                                   : 0; // TODO: shared memory support only for types <= 32 bytes
      // Note: for ecntt we limit block size (=#threads per block) since otherwise it doesn't fit the SM resources.
      int num_threads_coset = max(min(n / 2, IS_ECNTT ? MAX_THREADS_BATCH_ECNTT : MAX_NUM_THREADS), 1);
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
          n_twiddles, logn, dit, d_output);

      if (dit) {
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
            arbitrary_coset ? 1 : -coset_gen_index, -n_twiddles, logn, !dit, d_output);

        utils_internal::NormalizeKernel<E, S>
          <<<num_blocks_coset, num_threads_coset, 0, stream>>>(d_output, S::inv_log_size(logn), n * batch_size);
      }

      return CHK_LAST();
    }

  } // namespace

  /**
   * @struct Domain
   * Struct containing information about the domain on which (i)NTT is evaluated i.e. twiddle factors.
   * Twiddle factors are private, static and can only be set using [init_domain](@ref init_domain) function.
   * The internal representation of twiddles is prone to change in accordance with changing [NTT](@ref NTT) algorithm.
   * @tparam S The type of twiddle factors \f$ \{ \omega^i \} \f$. Must be a field.
   */
  template <typename S>
  class Domain
  {
    // Mutex for protecting access to the domain/device container array
    static inline std::mutex device_domain_mutex;
    // The domain-per-device container - assumption is init_domain is called once per device per program.

    int max_size = 0;
    int max_log_size = 0;
    S* twiddles = nullptr;
    bool initialized = false; // protection for multi-threaded case
    std::unordered_map<S, int> coset_index = {};

    S* internal_twiddles = nullptr; // required by mixed-radix NTT
    S* basic_twiddles = nullptr;    // required by mixed-radix NTT

    // mixed-radix NTT supports a fast-twiddle option at the cost of additional 4N memory (where N is max NTT size)
    S* fast_external_twiddles = nullptr;     // required by mixed-radix NTT (fast-twiddles mode)
    S* fast_internal_twiddles = nullptr;     // required by mixed-radix NTT (fast-twiddles mode)
    S* fast_basic_twiddles = nullptr;        // required by mixed-radix NTT (fast-twiddles mode)
    S* fast_external_twiddles_inv = nullptr; // required by mixed-radix NTT (fast-twiddles mode)
    S* fast_internal_twiddles_inv = nullptr; // required by mixed-radix NTT (fast-twiddles mode)
    S* fast_basic_twiddles_inv = nullptr;    // required by mixed-radix NTT (fast-twiddles mode)

  public:
    template <typename U, typename R>
    friend cudaError_t
    init_domain<U, R>(R primitive_root, device_context::DeviceContext& ctx, bool fast_tw);

    template <typename U>
    friend cudaError_t release_domain(device_context::DeviceContext& ctx);

    template <typename U>
    friend U get_root_of_unity<U>(uint64_t logn, device_context::DeviceContext& ctx);

    template <typename U>
    friend U get_root_of_unity_from_domain<U>(uint64_t logn, device_context::DeviceContext& ctx);

    template <typename U, typename E>
    friend cudaError_t ntt<U, E>(const E* input, int size, NTTDir dir, NTTConfig<U>& config, E* output);
  };

  template <typename S>
  static inline Domain<S> domains_for_devices[device_context::MAX_DEVICES] = {};

  template <typename S, typename R>
  cudaError_t init_domain(R primitive_root, device_context::DeviceContext& ctx, bool fast_twiddles_mode)
  {
    CHK_INIT_IF_RETURN();

    Domain<S>& domain = domains_for_devices<S>[ctx.device_id];

    // only generate twiddles if they haven't been generated yet
    // please note that this offers just basic thread-safety,
    // it's assumed a singleton (non-enforced) that is supposed
    // to be initialized once per device per program lifetime
    if (!domain.initialized) {
      // Mutex is automatically released when lock goes out of scope, even in case of exceptions
      std::lock_guard<std::mutex> lock(Domain<S>::device_domain_mutex);
      // double check locking
      if (domain.initialized) return CHK_LAST(); // another thread is already initializing the domain

      bool found_logn = false;
      R omega = primitive_root;
      unsigned omegas_count = R::get_omegas_count();
      for (int i = 0; i < omegas_count; i++) {
        omega = R::sqr(omega);
        if (!found_logn) {
          ++domain.max_log_size;
          found_logn = omega == R::one();
          if (found_logn) break;
        }
      }

#ifdef DCCT
      domain.max_log_size--;
#endif

      domain.max_size = (int)pow(2, domain.max_log_size);
      if (omega != R::one()) {
        THROW_ICICLE_ERR(
          IcicleError_t::InvalidArgument, "Primitive root provided to the InitDomain function is not in the subgroup");
      }

// #define DCCT
#ifdef DCCT
      // allocate and calculate twiddles on GPU
      // N * (2 ** (N - 1))
      size_t number_of_twiddles = domain.max_log_size * (1 << (domain.max_log_size - 1));
      CHK_IF_RETURN(cudaMalloc(&domain.basic_twiddles, number_of_twiddles * sizeof(S)));

      CHK_IF_RETURN(mxntt::generate_twiddles_dcct(
        primitive_root, domain.basic_twiddles, domain.max_log_size, ctx.stream));

      // S* tmp = static_cast<S*>(malloc(number_of_twiddles * sizeof(S)));
      // cudaMemcpy(tmp, domain.basic_twiddles, number_of_twiddles * sizeof(S), cudaMemcpyDeviceToHost);
      // for (size_t i = 0; i < number_of_twiddles; i++) {
      //   std::cout << tmp[i] << std::endl;
      // }
      domain.coset_index[S::one()] = 0;
#else
      // allocate and calculate twiddles on GPU
      // Note: radix-2 INTT needs ONE in last element (in addition to first element), therefore have n+1 elements
      // Managed allocation allows host to read the elements (logn) without copying all (n) TFs back to host
      CHK_IF_RETURN(cudaMallocManaged(&domain.twiddles, (domain.max_size + 1) * sizeof(S)));

      CHK_IF_RETURN(mxntt::generate_external_twiddles_generic(
        primitive_root, domain.twiddles, domain.internal_twiddles, domain.basic_twiddles, domain.max_log_size,
        ctx.stream));

      if (fast_twiddles_mode) {
        // generating fast-twiddles (note that this cost 4N additional memory)
        CHK_IF_RETURN(cudaMallocAsync(&domain.fast_external_twiddles, domain.max_size * sizeof(S) * 2, ctx.stream));
        CHK_IF_RETURN(cudaMallocAsync(&domain.fast_external_twiddles_inv, domain.max_size * sizeof(S) * 2, ctx.stream));

        // fast-twiddles forward NTT
        CHK_IF_RETURN(mxntt::generate_external_twiddles_fast_twiddles_mode(
          primitive_root, domain.fast_external_twiddles, domain.fast_internal_twiddles, domain.fast_basic_twiddles,
          domain.max_log_size, ctx.stream));

        // fast-twiddles inverse NTT
        S primitive_root_inv;
        CHK_IF_RETURN(cudaMemcpyAsync(
          &primitive_root_inv, &domain.twiddles[domain.max_size - 1], sizeof(S), cudaMemcpyDeviceToHost, ctx.stream));
        CHK_IF_RETURN(mxntt::generate_external_twiddles_fast_twiddles_mode(
          primitive_root_inv, domain.fast_external_twiddles_inv, domain.fast_internal_twiddles_inv,
          domain.fast_basic_twiddles_inv, domain.max_log_size, ctx.stream));
      }
      CHK_IF_RETURN(cudaStreamSynchronize(ctx.stream));

      const bool is_map_only_powers_of_primitive_root = true;
      if (is_map_only_powers_of_primitive_root) {
        // populate the coset_index map. Note that only powers of the primitive-root are stored (1, PR, PR^2, PR^4, PR^8
        // etc.)
        domain.coset_index[S::one()] = 0;
        for (int i = 0; i < domain.max_log_size; ++i) {
          const int index = (int)pow(2, i);
          domain.coset_index[domain.twiddles[index]] = index;
        }
      } else {
        // populate all values
        for (int i = 0; i < domain.max_size; ++i) {
          domain.coset_index[domain.twiddles[i]] = i;
        }
      }
#endif
      domain.initialized = true;
    }

    return CHK_LAST();
  }

  template <typename S>
  cudaError_t release_domain(device_context::DeviceContext& ctx)
  {
    CHK_INIT_IF_RETURN();

    Domain<S>& domain = domains_for_devices<S>[ctx.device_id];

    domain.max_size = 0;
    domain.max_log_size = 0;
    domain.twiddles = nullptr; // allocated via cudaMallocManaged(...) so released without calling cudaFree(...)
    CHK_IF_RETURN(cudaFreeAsync(domain.internal_twiddles, ctx.stream));
    domain.internal_twiddles = nullptr;
    CHK_IF_RETURN(cudaFreeAsync(domain.basic_twiddles, ctx.stream));
    domain.basic_twiddles = nullptr;
    domain.coset_index.clear();

    CHK_IF_RETURN(cudaFreeAsync(domain.fast_external_twiddles, ctx.stream));
    domain.fast_external_twiddles = nullptr;
    CHK_IF_RETURN(cudaFreeAsync(domain.fast_internal_twiddles, ctx.stream));
    domain.fast_internal_twiddles = nullptr;
    CHK_IF_RETURN(cudaFreeAsync(domain.fast_basic_twiddles, ctx.stream));
    domain.fast_basic_twiddles = nullptr;
    CHK_IF_RETURN(cudaFreeAsync(domain.fast_external_twiddles_inv, ctx.stream));
    domain.fast_external_twiddles_inv = nullptr;
    CHK_IF_RETURN(cudaFreeAsync(domain.fast_internal_twiddles_inv, ctx.stream));
    domain.fast_internal_twiddles_inv = nullptr;
    CHK_IF_RETURN(cudaFreeAsync(domain.fast_basic_twiddles_inv, ctx.stream));
    domain.fast_basic_twiddles_inv = nullptr;
    domain.initialized = false;

    return CHK_LAST();
  }

  template <typename S>
  S get_root_of_unity(uint64_t max_size)
  {
    // ceil up
    const auto log_max_size = static_cast<uint32_t>(std::ceil(std::log2(max_size)));
    return S::omega(log_max_size);
  }
  // explicit instantiation to avoid having to include this file
  template scalar_t get_root_of_unity(uint64_t logn);

  template <typename S>
  S get_root_of_unity_from_domain(uint64_t logn, device_context::DeviceContext& ctx)
  {
    Domain<S>& domain = domains_for_devices<S>[ctx.device_id];
    if (logn > domain.max_log_size) {
      std::ostringstream oss;
      oss << "NTT log_size=" << logn
          << " is too large for the domain. Consider generating your domain with a higher order root of unity.\n";
      THROW_ICICLE_ERR(IcicleError_t::InvalidArgument, oss.str().c_str());
    }
    const size_t twiddles_idx = 1ULL << (domain.max_log_size - logn);
    return domain.twiddles[twiddles_idx];
  }
  // explicit instantiation to avoid having to include this file
  template scalar_t get_root_of_unity_from_domain(uint64_t logn, device_context::DeviceContext& ctx);

  template <typename S>
  static bool is_choosing_radix2_algorithm(int logn, int batch_size, const NTTConfig<S>& config)
  {
    const bool is_mixed_radix_alg_supported = (logn > 3 && logn != 7);
    if (!is_mixed_radix_alg_supported && config.columns_batch)
      throw IcicleError(IcicleError_t::InvalidArgument, "columns batch is not supported for given NTT size");
    const bool is_user_selected_radix2_alg = config.ntt_algorithm == NttAlgorithm::Radix2;
    const bool is_force_radix2 = !is_mixed_radix_alg_supported || is_user_selected_radix2_alg;
    if (is_force_radix2) return true;

    const bool is_user_selected_mixed_radix_alg = config.ntt_algorithm == NttAlgorithm::MixedRadix;
    if (is_user_selected_mixed_radix_alg) return false;
    if (config.columns_batch) return false; // radix2 does not currently support columns batch mode.

    // Heuristic to automatically select an algorithm
    // Note that generally the decision depends on {logn, batch, ordering, inverse, coset, in-place, coeff-field} and
    // the specific GPU.
    // the following heuristic is a simplification based on measurements. Users can try both and select the algorithm
    // based on the specific case via the 'NTTConfig.ntt_algorithm' field

    if (logn >= 16) return false; // mixed-radix is typically faster in those cases
    if (logn <= 11) return true;  //  radix-2 is typically faster for batch<=256 in those cases
    const int log_batch = (int)log2(batch_size);
    return (logn + log_batch <= 18); // almost the cutoff point where both are equal
  }

  template <typename S, typename E>
  cudaError_t radix2_ntt(
    const E* d_input,
    E* d_output,
    S* twiddles,
    int ntt_size,
    int max_size,
    int batch_size,
    bool is_inverse,
    Ordering ordering,
    S* arbitrary_coset,
    int coset_gen_index,
    cudaStream_t cuda_stream)
  {
    CHK_INIT_IF_RETURN();

    const int logn = int(log2(ntt_size));

    bool dit = true;
    bool reverse_input = false;
    switch (ordering) {
    case Ordering::kNN:
      reverse_input = true;
      break;
    case Ordering::kNR:
    case Ordering::kNM:
      dit = false;
      break;
    case Ordering::kRR:
      reverse_input = true;
      dit = false;
      break;
    case Ordering::kRN:
    case Ordering::kMN:
      reverse_input = false;
    }

    if (reverse_input) reverse_order_batch(d_input, ntt_size, logn, batch_size, cuda_stream, d_output);

    CHK_IF_RETURN(ntt_inplace_batch_template(
      reverse_input ? d_output : d_input, ntt_size, twiddles, max_size, batch_size, logn, is_inverse, dit,
      arbitrary_coset, coset_gen_index, cuda_stream, d_output));

    return CHK_LAST();
  }

  template <typename S, typename E>
  cudaError_t ntt(const E* input, int size, NTTDir dir, NTTConfig<S>& config, E* output)
  {
    CHK_INIT_IF_RETURN();

    Domain<S>& domain = domains_for_devices<S>[config.ctx.device_id];

    if (size > domain.max_size) {
      std::ostringstream oss;
      oss << "NTT size=" << size
          << " is too large for the domain. Consider generating your domain with a higher order root of unity.\n";
      THROW_ICICLE_ERR(IcicleError_t::InvalidArgument, oss.str().c_str());
    }

    int logn = int(log2(size));
    const bool is_size_power_of_two = size == (1 << logn);
    if (!is_size_power_of_two) {
      std::ostringstream oss;
      oss << "NTT size=" << size << " is not supported since it is not a power of two.\n";
      THROW_ICICLE_ERR(IcicleError_t::InvalidArgument, oss.str().c_str());
    }

    cudaStream_t& stream = config.ctx.stream;
    size_t batch_size = config.batch_size;
    size_t input_size_bytes = (size_t)size * batch_size * sizeof(E);
    bool are_inputs_on_device = config.are_inputs_on_device;
    bool are_outputs_on_device = config.are_outputs_on_device;

    const E* d_input;
    E* d_allocated_input = nullptr;
    if (are_inputs_on_device) {
      d_input = input;
    } else {
      CHK_IF_RETURN(cudaMallocAsync(&d_allocated_input, input_size_bytes, stream));
      CHK_IF_RETURN(cudaMemcpyAsync(d_allocated_input, input, input_size_bytes, cudaMemcpyHostToDevice, stream));
      d_input = d_allocated_input;
    }
    E* d_output;
    E* d_allocated_output = nullptr;
    if (are_outputs_on_device) {
      d_output = output;
    } else {
      CHK_IF_RETURN(cudaMallocAsync(&d_allocated_output, input_size_bytes, stream));
      d_output = d_allocated_output;
    }

    S* coset = nullptr;
    int coset_index = 0;
    try {
      coset_index = domain.coset_index.at(config.coset_gen);
    } catch (...) {
      std::cout << "Computing coset" << std::endl;
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

    const bool is_inverse = dir == NTTDir::kInverse;

    if constexpr (IS_ECNTT) {
      CHK_IF_RETURN(ntt::radix2_ntt(
        d_input, d_output, domain.twiddles, size, domain.max_size, batch_size, is_inverse, config.ordering, coset,
        coset_index, stream));
    } else {
      const bool is_radix2_algorithm = is_choosing_radix2_algorithm(logn, batch_size, config);
      if (is_radix2_algorithm) {
        CHK_IF_RETURN(ntt::radix2_ntt(
          d_input, d_output, domain.twiddles, size, domain.max_size, batch_size, is_inverse, config.ordering, coset,
          coset_index, stream));
      } else {
        const bool is_fast_twiddles_enabled = (domain.fast_external_twiddles != nullptr);
        S* twiddles = is_fast_twiddles_enabled
                        ? (is_inverse ? domain.fast_external_twiddles_inv : domain.fast_external_twiddles)
                        : domain.twiddles;
        S* internal_twiddles = is_fast_twiddles_enabled
                                 ? (is_inverse ? domain.fast_internal_twiddles_inv : domain.fast_internal_twiddles)
                                 : domain.internal_twiddles;
        S* basic_twiddles = is_fast_twiddles_enabled
                              ? (is_inverse ? domain.fast_basic_twiddles_inv : domain.fast_basic_twiddles)
                              : domain.basic_twiddles;
        S* linear_twiddles = domain.twiddles; // twiddles organized as [1,w,w^2,...]
        std::cout << "Calling mixed radix" << std::endl;
        CHK_IF_RETURN(mxntt::mixed_radix_ntt(
          d_input, d_output, twiddles, internal_twiddles, basic_twiddles, linear_twiddles, size, domain.max_log_size,
          batch_size, config.columns_batch, is_inverse, is_fast_twiddles_enabled, config.ordering, coset, coset_index,
          stream));
      }
    }

    if (!are_outputs_on_device)
      CHK_IF_RETURN(cudaMemcpyAsync(output, d_output, input_size_bytes, cudaMemcpyDeviceToHost, stream));

    if (coset) CHK_IF_RETURN(cudaFreeAsync(coset, stream));
    if (d_allocated_input) CHK_IF_RETURN(cudaFreeAsync(d_allocated_input, stream));
    if (d_allocated_output) CHK_IF_RETURN(cudaFreeAsync(d_allocated_output, stream));
    if (!config.is_async) return CHK_STICKY(cudaStreamSynchronize(stream));

    return CHK_LAST();
  }

  template <typename S>
  NTTConfig<S> default_ntt_config(const device_context::DeviceContext& ctx)
  {
    NTTConfig<S> config = {
      ctx,                // ctx
      S::one(),           // coset_gen
      1,                  // batch_size
      false,              // columns_batch
      Ordering::kNN,      // ordering
      false,              // are_inputs_on_device
      false,              // are_outputs_on_device
      false,              // is_async
      NttAlgorithm::Auto, // ntt_algorithm
    };
    return config;
  }
  // explicit instantiation to avoid having to include this file
  template NTTConfig<scalar_t> default_ntt_config(const device_context::DeviceContext& ctx);
} // namespace ntt