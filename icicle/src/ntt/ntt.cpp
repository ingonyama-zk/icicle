

#define FIELD_ID BN254
#include "../../include/fields/field_config.cuh"


using namespace field_config;

#include "../../include/ntt/ntt.cuh"

#include <unordered_map>
#include <vector>
#include <type_traits>

#include "../../include/gpu-utils/sharedmem.cuh"
#include "../../include/utils/utils_kernels.cuh"
#include "../../include/utils/utils.h"
#include "../../include/ntt/ntt_impl.cuh"
#include "../../include/gpu-utils/device_context.cuh"

#include <mutex>

#ifdef CURVE_ID
#include "../../include/curves/curve_config.cuh"
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
    void reverse_order_kernel(const E* arr, E* arr_reversed, uint32_t n, uint32_t logn, uint32_t batch_size)
    {
      return;
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
      return;
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
    void ntt_template_kernel_shared_rev(
      const E* __restrict__ arr_in,
      int n,
      const S* __restrict__ r_twiddles,
      int n_twiddles,
      int max_task,
      int ss,
      int logn,
      E* __restrict__ arr_out)
    {
      return;
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
    void ntt_template_kernel_shared(
      const E* __restrict__ arr_in,
      int n,
      const S* __restrict__ r_twiddles,
      int n_twiddles,
      int max_task,
      int s,
      int logn,
      E* __restrict__ arr_out)
    {
      return;
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
    void
    ntt_template_kernel(const E* arr_in, int n, S* twiddles, int n_twiddles, int max_task, int s, bool rev, E* arr_out)
    {
      return;
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
      return 0;
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
    template <typename U>
    friend cudaError_t init_domain(U primitive_root, device_context::DeviceContext& ctx, bool fast_tw);

    template <typename U>
    friend cudaError_t release_domain(device_context::DeviceContext& ctx);

    template <typename U>
    friend U get_root_of_unity(uint64_t logn, device_context::DeviceContext& ctx);

    template <typename U>
    friend U get_root_of_unity_from_domain(uint64_t logn, device_context::DeviceContext& ctx);

    template <typename U, typename E>
    friend cudaError_t ntt(const E* input, int size, NTTDir dir, NTTConfig<U>& config, E* output);
  };

  template <typename S>
  // static inline Domain<S> domains_for_devices[device_context::MAX_DEVICES] = {};
  static inline Domain<S> domains_for_devices[1] = {};

  template <typename S>
  cudaError_t init_domain(S primitive_root, device_context::DeviceContext& ctx, bool fast_twiddles_mode)
  {
    return 0;
  }

  template <typename S>
  cudaError_t release_domain(device_context::DeviceContext& ctx)
  {
    return 0;
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
    return 0;
  }

  template <typename S, typename E>
  cudaError_t ntt(const E* input, int size, NTTDir dir, NTTConfig<S>& config, E* output)
  {
    return 0;
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