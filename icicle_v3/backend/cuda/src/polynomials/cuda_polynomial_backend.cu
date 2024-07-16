
#include "icicle/polynomials/polynomials.h"

#include "gpu-utils/error_handler.h"
#include "cuda_runtime.h"
#include "icicle/ntt.h"
#include "kernels.cuh"
#include "icicle/runtime.h"
#include "icicle/errors.h"

namespace polynomials {

  // TODO Yuval : CPU polynomials don't need need to be copied to device
  //              actually it's true for any device that shares memory with host

  static uint64_t ceil_to_power_of_two(uint64_t x) { return 1ULL << uint64_t(ceil(log2(x))); }
  /*============================== Polynomial Default-context ==============================*/

  // checking whether a pointer is on host or device and asserts device matches the polynmoial device
  static bool is_host_ptr(const void* ptr)
  {
    if (eIcicleError::SUCCESS == icicle_is_host_memory(ptr)) return true;

    ICICLE_ASSERT(eIcicleError::SUCCESS == icicle_is_active_device_memory(ptr));
    return false;
  }

  template <typename C = scalar_t, typename D = C, typename I = C>
  class DefaultPolynomialContext : public IPolynomialContext<C, D, I>
  {
    typedef IPolynomialContext<C, D, I> PolyContext;
    using typename IPolynomialContext<C, D, I>::State;
    using IPolynomialContext<C, D, I>::ElementSize;

  protected:
    State m_state = State::Invalid; // Current state of the polynomial context.
    uint64_t m_nof_elements = 0;    // Number of elements managed by the context.

  public:
    cudaStream_t m_stream;
    // icicleStreamHandle m_stream;

    DefaultPolynomialContext(const icicleStreamHandle stream) : m_stream{reinterpret_cast<cudaStream_t>(stream)}
    {
      m_integrity_counter = std::make_shared<int>(0);
    }
    ~DefaultPolynomialContext() { release(); }

    void allocate(uint64_t nof_elements, State init_state, bool is_memset_zeros) override
    {
      const bool is_already_allocated = this->m_nof_elements >= nof_elements;
      this->set_state(init_state);

      if (is_already_allocated) {
        // zero the extra elements, if exist
        memset_zeros(this->m_storage, nof_elements, this->m_nof_elements);
        return;
      }

      release(); // in case allocated mem is too small and need to reallocate
      this->m_nof_elements = allocate_mem(nof_elements, &this->m_storage, is_memset_zeros);
      modified();
    }

    void memset_zeros(void* storage, uint64_t element_start_idx, uint64_t element_end_idx)
    {
      const uint64_t size = (element_end_idx - element_start_idx) * ElementSize;
      if (0 == size) { return; }

      modified();

      const auto offset = (void*)((uint64_t)storage + element_start_idx * ElementSize);
      ICICLE_CHECK(icicle_memset_async(offset, 0, size, m_stream));
    }

    uint64_t allocate_mem(uint64_t nof_elements, void** storage /*OUT*/, bool is_memset_zeros)
    {
      const uint64_t nof_elements_nearset_power_of_two = ceil_to_power_of_two(nof_elements);
      const uint64_t mem_size = nof_elements_nearset_power_of_two * ElementSize;

      ICICLE_CHECK(icicle_malloc_async((void**)storage, mem_size, m_stream));

      if (is_memset_zeros) {
        memset_zeros(*storage, 0, nof_elements_nearset_power_of_two);
      } else {
        // if allocating more memory than requested, memset only the pad area to avoid higher invalid coefficients
        memset_zeros(*storage, nof_elements, nof_elements_nearset_power_of_two);
      }

      return nof_elements_nearset_power_of_two;
    }

    void set_storage(void* storage, uint64_t nof_elements)
    {
      release();
      m_storage = storage;
      this->m_nof_elements = nof_elements;

      modified();
    }

    // Note: this is protected and only backend can call
    void* get_storage_mutable() override
    {
      // since giving access to internal memory, cannot know if modified or not
      // backend should not take it mutable if not mutating
      modified();
      return m_storage;
    }
    const void* get_storage_immutable() override { return m_storage; }

    void extend_mem_and_pad(uint64_t nof_elements)
    {
      void* new_storage = nullptr;
      const uint64_t new_nof_elements = allocate_mem(nof_elements, &new_storage, true /*=memset zeros*/);
      const uint64_t old_mem_size = this->m_nof_elements * ElementSize;

      ICICLE_CHECK(icicle_copy_async(new_storage, m_storage, old_mem_size, m_stream));

      set_storage(new_storage, new_nof_elements);
    }

    void release() override
    {
      if (m_storage != nullptr) { ICICLE_CHECK(icicle_free_async(m_storage, m_stream)); }

      m_storage = nullptr;
      this->m_nof_elements = 0;

      modified();
    }

    State get_state() const override { return m_state; }
    void set_state(State state) { m_state = state; }
    uint64_t get_nof_elements() const override { return m_nof_elements; }

    void from_coefficients(uint64_t nof_coefficients, const C* coefficients) override
    {
      const bool is_memset_zeros = coefficients == nullptr;
      allocate(nof_coefficients, State::Coefficients, is_memset_zeros);
      if (coefficients) {
        ICICLE_CHECK(icicle_copy_async(m_storage, coefficients, nof_coefficients * sizeof(C), m_stream));
        ICICLE_CHECK(icicle_stream_synchronize(m_stream)); // protect against coefficients being released too soon
      }
    }

    void from_rou_evaluations(uint64_t nof_evaluations, const I* evaluations) override
    {
      const bool is_memset_zeros = evaluations == nullptr;
      allocate(nof_evaluations, State::EvaluationsOnRou_Natural, is_memset_zeros);
      if (evaluations) {
        ICICLE_CHECK(icicle_copy_async(m_storage, evaluations, nof_evaluations * sizeof(C), m_stream));
        ICICLE_CHECK(icicle_stream_synchronize(m_stream)); // protect against evaluations being released too soon
      }
    }

    void clone(IPolynomialContext<C, D, I>& from) override
    {
      switch (from.get_state()) {
      case State::Coefficients: {
        auto [coeffs, N_coeffs] = from.get_coefficients();
        from_coefficients(N_coeffs, coeffs);
      } break;
      case State::EvaluationsOnRou_Natural: {
        auto [evals, N_evals] = from.get_rou_evaluations();
        from_rou_evaluations(N_evals, evals);
      } break;
      default:
        THROW_ICICLE_ERR(IcicleError_t::InvalidArgument, "clone() from non implemented state");
      }

      this->set_state(from.get_state()); // to handle both reversed evaluations case
    }

    std::pair<const C*, uint64_t> get_coefficients() override
    {
      transform_to_coefficients();
      return std::make_pair(static_cast<const C*>(m_storage), this->m_nof_elements);
    }

    std::tuple<IntegrityPointer<C>, uint64_t> get_coefficients_view() override
    {
      auto [coeffs, N] = get_coefficients();
      // when reading the pointer, if the counter was modified, the pointer is invalid
      IntegrityPointer<C> integrity_pointer(coeffs, m_integrity_counter, *m_integrity_counter);
      ICICLE_CHECK(icicle_stream_synchronize(m_stream));
      return {std::move(integrity_pointer), N};
    }

    std::pair<const I*, uint64_t> get_rou_evaluations() override
    {
      const bool is_reversed = this->m_state == State::EvaluationsOnRou_Reversed;
      transform_to_evaluations(0, is_reversed);
      return std::make_pair(static_cast<const I*>(m_storage), this->m_nof_elements);
    }

    void transform_to_coefficients(uint64_t nof_coefficients = 0) override
    {
      // cannot really get more coefficients but sometimes want to pad for NTT. In that case
      // nof_coefficients>m_nof_elements
      nof_coefficients = (nof_coefficients == 0) ? this->m_nof_elements : ceil_to_power_of_two(nof_coefficients);
      const bool is_same_nof_coefficients = this->m_nof_elements == nof_coefficients;
      const bool is_already_in_state = this->m_state == State::Coefficients && is_same_nof_coefficients;
      if (is_already_in_state) { return; }

      if (nof_coefficients < this->m_nof_elements) {
        THROW_ICICLE_ERR(
          IcicleError_t::InvalidArgument, "polynomial shrinking not supported. Probably encountered a bug");
      }

      modified();

      const bool is_already_in_coeffs = this->m_state == State::Coefficients;
      // case 1: already in coefficients. Need to allocate larger memory and zero pad
      if (is_already_in_coeffs) {
        extend_mem_and_pad(nof_coefficients);
        return;
      }

      // case 2: transform from evaluations. May need to allocate larger memory
      I* evals = static_cast<I*>(m_storage);
      C* coeffs = static_cast<C*>(m_storage);
      const bool is_allocate_new_mem = nof_coefficients > this->m_nof_elements;
      if (is_allocate_new_mem) {
        void* new_mem = nullptr;
        nof_coefficients = allocate_mem(nof_coefficients, &new_mem, true /*=memset zeros*/);
        coeffs = static_cast<C*>(new_mem);
      }

      // transform from evaluations to coefficients
      auto ntt_config = default_ntt_config<C>();
      ntt_config.are_inputs_on_device = true;
      ntt_config.are_outputs_on_device = true;
      ntt_config.is_async = true;

      ntt_config.ordering = (this->m_state == State::EvaluationsOnRou_Natural) ? Ordering::kNN : Ordering::kRN;
      // Note: it is important to do the NTT with old size because padding in evaluations form is computing another
      // (higher order) polynomial
      ICICLE_CHECK(ntt(evals, this->m_nof_elements, NTTDir::kInverse, ntt_config, coeffs));
      this->set_state(State::Coefficients);

      if (is_allocate_new_mem) { set_storage(coeffs, nof_coefficients); } // release old memory and use new
    }

    void transform_to_evaluations(uint64_t nof_evaluations = 0, bool is_reversed = false) override
    {
      // TODO Yuval: can maybe optimize this
      nof_evaluations = (nof_evaluations == 0) ? this->m_nof_elements : ceil_to_power_of_two(nof_evaluations);
      const bool is_same_nof_evaluations = nof_evaluations == this->m_nof_elements;
      const bool is_same_order = is_reversed && this->m_state == State::EvaluationsOnRou_Reversed ||
                                 (!is_reversed && this->m_state == State::EvaluationsOnRou_Natural);
      const bool is_already_in_state = is_same_nof_evaluations && is_same_order;
      if (is_already_in_state) { return; }

      if (nof_evaluations < this->m_nof_elements) {
        THROW_ICICLE_ERR(
          IcicleError_t::InvalidArgument, "polynomial shrinking not supported. Probably encountered a bug");
      }

      modified();

      // TODO Yuval: evaluations->evaluations with different ordering can be implemented via inplace reorder more
      // efficiently than it is now

      // There are 3 cases:
      // (1) coefficients to evaluations
      //    (1a) same size -> NTT (NR or NN)
      //    (1b) different_size -> alloc new mem, copy coeffs and NTT inplace
      // (2) evaluations to evaluations (interpolation)
      //     transform to coefficients, extend memory, then NTT back to evals (NR or NN)

      const bool is_eval_to_eval = this->m_state != State::Coefficients;
      // interpolating more points requires going back to coefficients first. Note that it muse be done with the
      // original size. INTT after padding computes a higher degree polynomial
      if (is_eval_to_eval) { transform_to_coefficients(); }

      // reaching this point means polynomial is in coefficient form
      const bool is_allocate_new_mem = nof_evaluations > this->m_nof_elements;
      // allocate more memory and copy+pad
      if (is_allocate_new_mem) { extend_mem_and_pad(nof_evaluations); }

      C* coeffs = static_cast<C*>(m_storage);
      I* evals = static_cast<I*>(m_storage);
      auto ntt_config = default_ntt_config<C>();
      ntt_config.are_inputs_on_device = true;
      ntt_config.are_outputs_on_device = true;
      ntt_config.is_async = true;
      // already copied the coefficients with padding. Now computing evaluations.
      ntt_config.ordering = is_reversed ? Ordering::kNR : Ordering::kNN;
      ICICLE_CHECK(ntt(coeffs, nof_evaluations, NTTDir::kForward, ntt_config, evals));

      this->set_state(is_reversed ? State::EvaluationsOnRou_Reversed : State::EvaluationsOnRou_Natural);
    }

    void print(std::ostream& os) override
    {
      if (this->get_state() == State::Coefficients) {
        print_coeffs(os);
      } else {
        print_evals(os);
      }
    }

    void print_coeffs(std::ostream& os)
    {
      transform_to_coefficients();
      auto host_coeffs = std::make_unique<C[]>(this->m_nof_elements);
      // using stream since previous ops may still be in progress. Sync stream before reading CPU mem
      ICICLE_CHECK(icicle_copy_async(host_coeffs.get(), m_storage, this->m_nof_elements * sizeof(C), m_stream));
      ICICLE_CHECK(icicle_stream_synchronize(m_stream));

      os << "(id=" << PolyContext::m_id << ")[";
      for (size_t i = 0; i < this->m_nof_elements; ++i) {
        os << host_coeffs[i];
        if (i < this->m_nof_elements - 1) { os << ", "; }
      }
      os << "] (state=coefficients)" << std::endl;
    }

    void print_evals(std::ostream& os)
    {
      transform_to_evaluations();
      auto host_evals = std::make_unique<I[]>(this->m_nof_elements);
      // using stream since previous ops may still be in progress. Sync stream before reading CPU mem
      ICICLE_CHECK(icicle_copy_async(host_evals.get(), m_storage, this->m_nof_elements * sizeof(I), m_stream));
      ICICLE_CHECK(icicle_stream_synchronize(m_stream));

      os << "(id=" << PolyContext::m_id << ")[";
      for (size_t i = 0; i < this->m_nof_elements; ++i) {
        os << host_evals[i];
        if (i < this->m_nof_elements - 1) { os << ", "; }
      }

      if (this->get_state() == State::EvaluationsOnRou_Reversed) {
        os << "] (state=rou evaluations Reversed)" << std::endl;
      } else {
        os << "] (state=rou evaluations )" << std::endl;
      }
    }

  private:
    // Members
    void* m_storage = nullptr;
    std::shared_ptr<int> m_integrity_counter; // used to implement integrity of coefficients pointer

    void modified() { (*m_integrity_counter)++; }
  };

  /*============================== Polynomial Default-backend ==============================*/

  template <typename C = scalar_t, typename D = C, typename I = C>
  class DefaultPolynomialBackend : public IPolynomialBackend<C, D, I>
  {
    typedef std::shared_ptr<IPolynomialContext<C, D, I>> PolyContext;
    typedef typename IPolynomialContext<C, D, I>::State State;

    int64_t* d_degree = nullptr; // used to avoid alloc/release every time

  public:
    cudaStream_t m_stream; // TODO Yuval make it icicleStreamHandle
    DefaultPolynomialBackend(const icicleStreamHandle stream) : m_stream{reinterpret_cast<cudaStream_t>(stream)}
    {
      ICICLE_CHECK(icicle_malloc_async((void**)&d_degree, sizeof(int64_t), m_stream));
    }
    ~DefaultPolynomialBackend() noexcept { icicle_free_async(d_degree, m_stream); }

    void from_coefficients(PolyContext p, uint64_t nof_coefficients, const C* coefficients) override
    {
      p->from_coefficients(nof_coefficients, coefficients);
    }

    void from_rou_evaluations(PolyContext p, uint64_t nof_evaluations, const I* evaluations) override
    {
      p->from_rou_evaluations(nof_evaluations, evaluations);
    }

    void clone(PolyContext out, PolyContext in) override { out->clone(*in); }

    template <typename T = C>
    T* get_context_storage_mutable(PolyContext p)
    {
      return static_cast<T*>(IPolynomialBackend<C, D, I>::get_context_storage_mutable(p));
    }

    template <typename T = C>
    const T* get_context_storage_immutable(PolyContext p) const
    {
      return static_cast<const T*>(IPolynomialBackend<C, D, I>::get_context_storage_immutable(p));
    }

    void slice(PolyContext out, PolyContext in, uint64_t offset, uint64_t stride, uint64_t size) override
    {
      assert_device_compatability({in});
      auto [in_coeffs, in_size] = in->get_coefficients();
      // size=0 means take as much as elements as there are to take
      uint64_t out_size = (size > 0) ? size : (1 + (in_size - 1 - offset) / stride);

      out->allocate(out_size, State::Coefficients, false /*=memset zeros*/);
      auto out_coeffs = get_context_storage_mutable(out);

      const int NOF_THREADS = 128;
      const int NOF_BLOCKS = (out_size + NOF_THREADS - 1) / NOF_THREADS;
      slice_kernel<<<NOF_BLOCKS, NOF_THREADS, 0, m_stream>>>(in_coeffs, out_coeffs, offset, stride, out_size);

      CHK_LAST();
    }

    void add_sub(PolyContext& res, PolyContext a, PolyContext b, bool add1_sub0)
    {
      assert_device_compatability({a, b});

      // add/sub can be done in both coefficients or evaluations, but operands must be in the same state.
      // For evaluations, same state also means same number of evaluations (and on same domain).
      // If not same state, compute in coefficients since computing in evaluations may require to interpolate a large
      // size. Consider a+b where a is degree 128 and b degree 4. In coefficients b has 4 elements but in evaluations
      // need 128.
      const bool is_same_size = a->get_nof_elements() == b->get_nof_elements();
      const bool is_same_state = a->get_state() == b->get_state();
      const auto output_state = (is_same_size && is_same_state) ? a->get_state() : State::Coefficients;
      const auto output_size = max(a->get_nof_elements(), b->get_nof_elements());

      if (State::Coefficients == output_state) {
        a->transform_to_coefficients();
        b->transform_to_coefficients();
      }
      const auto a_mem_p = get_context_storage_immutable(a);
      const auto b_mem_p = get_context_storage_immutable(b);

      res->allocate(output_size, output_state);
      auto res_mem_p = get_context_storage_mutable(res);

      const int NOF_THREADS = 128;
      const int NOF_BLOCKS = (output_size + NOF_THREADS - 1) / NOF_THREADS;
      add_sub_kernel<<<NOF_BLOCKS, NOF_THREADS, 0, m_stream>>>(
        a_mem_p, b_mem_p, a->get_nof_elements(), b->get_nof_elements(), add1_sub0, res_mem_p);

      CHK_LAST();
    }

    void add(PolyContext& res, PolyContext a, PolyContext b) override { add_sub(res, a, b, true /*=add*/); }
    void subtract(PolyContext res, PolyContext a, PolyContext b) override { add_sub(res, a, b, false /*=sub*/); }

    void multiply(PolyContext c, PolyContext a, PolyContext b) override
    {
      assert_device_compatability({a, b});

      const bool is_a_scalar = a->get_nof_elements() == 1;
      const bool is_b_scalar = b->get_nof_elements() == 1;

      // TODO: can add kernel that takes the scalar as device memory
      if (is_a_scalar) {
        return multiply(c, b, get_coeff(a, 0));
      } else if (is_b_scalar) {
        return multiply(c, a, get_coeff(b, 0));
      }

      const bool is_multiply_with_cosets = true; // TODO  Yuval: check when faster to do so.
      if (is_multiply_with_cosets) { return multiply_with_cosets(c, a, b); }
      return multiply_with_padding(c, a, b);
    }

    void multiply(PolyContext out, PolyContext p, D scalar) override
    {
      assert_device_compatability({p});

      // element wise multiplication is similar both in coefficients and evaluations (regardless of order too)
      const auto state = p->get_state();
      const auto N = p->get_nof_elements();

      auto p_elements_p =
        state == State::Coefficients ? get_context_storage_immutable<C>(p) : get_context_storage_immutable<I>(p);

      out->allocate(N, state, false /*=memset zeros*/);
      auto out_evals_p =
        state == State::Coefficients ? get_context_storage_mutable<C>(out) : get_context_storage_mutable<I>(out);

      const int NOF_THREADS = 128;
      const int NOF_BLOCKS = (N + NOF_THREADS - 1) / NOF_THREADS;
      mul_scalar_kernel<<<NOF_BLOCKS, NOF_THREADS, 0, m_stream>>>(p_elements_p, scalar, N, out_evals_p);

      CHK_LAST();
    }

    void multiply_with_padding(PolyContext c, PolyContext a, PolyContext b)
    {
      // TODO Yuval: by using the degree I can optimize the memory size and avoid redundant computations too
      const uint64_t a_N_orig = a->get_nof_elements();
      const uint64_t b_N_orig = b->get_nof_elements();
      const uint64_t N = max(a_N_orig, b_N_orig);
      const uint64_t c_N = 2 * N;

      // (1) transform a,b to 2N evaluations
      a->transform_to_evaluations(c_N, true /*=reversed*/);
      b->transform_to_evaluations(c_N, true /*=reversed*/);
      auto [a_evals_p, a_N] = a->get_rou_evaluations();
      auto [b_evals_p, b_N] = b->get_rou_evaluations();

      // (2) allocate c (c=a*b) and compute element-wise multiplication on evaluations
      c->allocate(c_N, State::EvaluationsOnRou_Reversed, false /*=memset zeros*/);
      auto c_evals_p = get_context_storage_mutable<I>(c);

      const int NOF_THREADS = 128;
      const int NOF_BLOCKS = (c_N + NOF_THREADS - 1) / NOF_THREADS;
      mul_kernel<<<NOF_BLOCKS, NOF_THREADS, 0, m_stream>>>(a_evals_p, b_evals_p, c_N, c_evals_p);

      CHK_LAST();
    }

    void multiply_with_cosets(PolyContext c, PolyContext a, PolyContext b)
    {
      const uint64_t a_N = a->get_nof_elements();
      const uint64_t b_N = b->get_nof_elements();
      const uint64_t N = max(a_N, b_N);

      // (1) transform a,b to coefficients such that both have N coefficients
      a->transform_to_coefficients(N);
      b->transform_to_coefficients(N);
      auto [a_coeff_p, _] = a->get_coefficients();
      auto [b_coeff_p, __] = b->get_coefficients();
      // (2) allocate c (c=a*b)
      const uint64_t c_N = 2 * N;
      c->allocate(c_N, State::EvaluationsOnRou_Reversed, false /*=memset zeros*/);
      auto c_evals_low_p = get_context_storage_mutable<I>(c);
      I* c_evals_high_p = c_evals_low_p + N;

      // (3) compute NTT of a,b on coset and write to c
      auto ntt_config = default_ntt_config<C>();
      ntt_config.are_inputs_on_device = true;
      ntt_config.are_outputs_on_device = true;
      ntt_config.is_async = true;
      ntt_config.ordering = Ordering::kNR;
      ICICLE_CHECK(get_root_of_unity_from_domain<C>((uint64_t)log2(c_N), ntt_config.coset_gen));

      ICICLE_CHECK(ntt(a_coeff_p, N, NTTDir::kForward, ntt_config, c_evals_low_p));  // a_H1
      ICICLE_CHECK(ntt(b_coeff_p, N, NTTDir::kForward, ntt_config, c_evals_high_p)); // b_H1

      // (4) compute a_H1 * b_H1 inplace
      const int NOF_THREADS = 128;
      const int NOF_BLOCKS = (N + NOF_THREADS - 1) / NOF_THREADS;
      mul_kernel<<<NOF_BLOCKS, NOF_THREADS, 0, m_stream>>>(c_evals_low_p, c_evals_high_p, N, c_evals_high_p);
      // (5) transform a,b to evaluations
      a->transform_to_evaluations(N, true /*=reversed*/);
      b->transform_to_evaluations(N, true /*=reversed*/);
      auto [a_evals_p, a_nof_evals] = a->get_rou_evaluations();
      auto [b_evals_p, b_nof_evals] = b->get_rou_evaluations();

      // (6) compute a_H0 * b_H0
      mul_kernel<<<NOF_BLOCKS, NOF_THREADS, 0, m_stream>>>(a_evals_p, b_evals_p, N, c_evals_low_p);

      CHK_LAST();
    }

    void divide(PolyContext Q /*OUT*/, PolyContext R /*OUT*/, PolyContext a, PolyContext b) override
    {
      assert_device_compatability({a, b});

      auto [a_coeffs, a_N] = a->get_coefficients();
      auto [b_coeffs, b_N] = b->get_coefficients();

      const int64_t deg_a = degree(a);
      const int64_t deg_b = degree(b);
      ICICLE_ASSERT(deg_b >= 0) << "Polynomial division:  divide by zero polynomial";

      // init: Q=0, R=a
      Q->allocate(deg_a - deg_b + 1, State::Coefficients, true /*=memset zeros*/);
      auto Q_coeffs = get_context_storage_mutable(Q);

      //    TODO Yuval: Can do better in terms of memory allocation? deg(R) <= deg(b) by definition but it starts as
      R->allocate(a_N, State::Coefficients, false /*=memset_zeros*/);
      auto R_coeffs = get_context_storage_mutable(R);
      ICICLE_CHECK(icicle_copy_async(R_coeffs, a_coeffs, a_N * sizeof(C), m_stream));

      const C& lc_b_inv = C::inverse(get_coeff(b, deg_b)); // largest coeff of b

      int64_t deg_r = deg_a;
      while (deg_r >= deg_b) {
        // each iteration is removing the largest monomial in r until deg(r)<deg(b)
        const int NOF_THREADS = 128;
        const int NOF_BLOCKS = ((deg_r + 1) + NOF_THREADS - 1) / NOF_THREADS; // 'deg_r+1' is number of elements in R
        school_book_division_step<<<NOF_BLOCKS, NOF_THREADS, 0, m_stream>>>(
          R_coeffs, Q_coeffs, b_coeffs, deg_r, deg_b, lc_b_inv);

        // faster than degree(R) based on the fact that degree is decreasing
        deg_r = degree_internal(R, deg_r + 1 /*size of R*/);
      }

      CHK_LAST();
    }

    void quotient(PolyContext Q, PolyContext op_a, PolyContext op_b) override
    {
      // TODO: can implement more efficiently?
      auto R = std::make_shared<DefaultPolynomialContext<C, D, I>>(m_stream);
      divide(Q, R, op_a, op_b);
    }

    void remainder(PolyContext R, PolyContext op_a, PolyContext op_b) override
    {
      // TODO: can implement more efficiently?
      auto Q = std::make_shared<DefaultPolynomialContext<C, D, I>>(m_stream);
      divide(Q, R, op_a, op_b);
    }

    void divide_by_vanishing_polynomial(PolyContext out, PolyContext numerator, uint64_t vanishing_poly_degree) override
    {
      assert_device_compatability({numerator});

      // vanishing polynomial of degree N is the polynomial V(x) such that V(r)=0 for r Nth root-of-unity.
      // For example for N=4 it vanishes on the group [1,W,W^2,W^3] where W is the 4th root of unity. In that
      // case V(x)=(x-1)(x-w)(x-w^2)(x-w^3). It can be easily shown that V(x)=x^N-1. This holds since x^N=1 on this
      // domain (since x is the Nth root of unity).

      // Note that we always represent polynomials with N elements for N a power of two. This is required for NTTs.
      // In addition we consider deg(P) to be this number of elements N even though the real degree may be lower. for
      // example 1+x-2x^2 is degree 2 but we store 4 elements and consider it degree 3.

      // when dividing a polynomial  P(x)/V(x) (The vanishing polynomial) the output is of degree deg(P)-deg(V). There
      // are three cases where V(x) divides P(x) (this is assumed since otherwise the output polynomial does not
      // exist!):
      // (1) deg(P)=2*deg(V): in that case deg(P/V)=deg(V)=N. This is an efficient case since on a domain of size N, the
      // vanishing polynomial evaluates to a constant value.
      // (2) deg(P)=deg(V)=N: in that case the output is a degree 0 polynomial.
      // polynomial (i.e. scalar). (3) general case: deg(P)>2*deg(V): in that case deg(P) is a least 4*deg(V) since N is
      // a power of two. This means that deg(P/V)=deg(P). For example deg(P)=16, deg(V)=4 --> deg(P/V)=12 ceiled to 16.

      // When computing we want to divide P(x)'s evals by V(x)'s evals. Since V(x)=0 on this domain we have to compute
      // on a coset.
      // for case (3) we must evaluate V(x) on deg(P) domain size and compute elementwise division on a coset.
      // case (1) is more efficient because we need N evaluations of V(x) on a coset. Note that V(x)=constant on a coset
      // of rou. This is because V(wu)=(wu)^N-1=W^N*u^N-1 = 1*u^N-1 (as w^N=1 for w Nth root of unity). case (2) can be
      // computed like case (1).

      const bool is_case_2N = numerator->get_nof_elements() == 2 * vanishing_poly_degree;
      const bool is_case_N = numerator->get_nof_elements() == vanishing_poly_degree;
      if (is_case_2N) {
        divide_by_vanishing_case_2N(out, numerator, vanishing_poly_degree);
      } else if (is_case_N) {
        divide_by_vanishing_case_N(out, numerator, vanishing_poly_degree);
      } else {
        divide_by_vanishing_general_case(out, numerator, vanishing_poly_degree);
      }
    }

    void divide_by_vanishing_general_case(PolyContext out, PolyContext numerator, uint64_t vanishing_poly_degree)
    {
      // General case: P(x)/V(x) where v is of degree N and p of any degree>N

      // (1) allocate vanishing polynomial in coefficients form
      // TODO Yuval: maybe instead of taking numerator memory and modiyfing it diretcly add a state for evaluations
      // on coset of rou. In that case I can remain in this state and also won't need to access input memory
      // directly
      numerator->transform_to_coefficients();
      auto numerator_coeffs = get_context_storage_mutable(numerator);
      const auto N = numerator->get_nof_elements();
      if (vanishing_poly_degree > N) {
        THROW_ICICLE_ERR(IcicleError_t::InvalidArgument, "divide_by_vanishing_polynomial(): degree is too large");
      }
      out->allocate(N, State::Coefficients, true /*=set zeros*/);
      add_monomial_inplace(out, C::zero() - C::one(), 0);         //-1
      add_monomial_inplace(out, C::one(), vanishing_poly_degree); //+x^n

      // (2) NTT on coset. Note that NTT on ROU evaluates to zeros for vanihsing polynomials by definition.
      // Therefore evaluation on coset is required to compute non-zero evaluations, which make element-wise division
      // possible
      auto out_coeffs = get_context_storage_mutable(out);
      auto ntt_config = default_ntt_config<C>();
      ntt_config.are_inputs_on_device = true;
      ntt_config.are_outputs_on_device = true;
      ntt_config.is_async = true;
      ntt_config.ordering = Ordering::kNM;
      ICICLE_CHECK(get_root_of_unity_from_domain<C>((uint64_t)log2(2 * N), ntt_config.coset_gen));

      ICICLE_CHECK(ntt(out_coeffs, N, NTTDir::kForward, ntt_config, out_coeffs));
      ICICLE_CHECK(ntt(numerator_coeffs, N, NTTDir::kForward, ntt_config, numerator_coeffs));

      // (3) element wise division
      const int NOF_THREADS = 128;
      const int NOF_BLOCKS = (N + NOF_THREADS - 1) / NOF_THREADS;
      div_element_wise_kernel<<<NOF_BLOCKS, NOF_THREADS, 0, m_stream>>>(numerator_coeffs, out_coeffs, N, out_coeffs);

      // (4) INTT back both numerator and out
      ntt_config.ordering = Ordering::kMN;
      ICICLE_CHECK(ntt(out_coeffs, N, NTTDir::kInverse, ntt_config, out_coeffs));
      ICICLE_CHECK(ntt(numerator_coeffs, N, NTTDir::kInverse, ntt_config, numerator_coeffs));
    }

    void divide_by_vanishing_case_2N(PolyContext out, PolyContext numerator, uint64_t vanishing_poly_degree)
    {
      // in that special case the numertaor has 2N elements and output will be N elements
      if (numerator->get_nof_elements() != 2 * vanishing_poly_degree) {
        THROW_ICICLE_ERR(IcicleError_t::UndefinedError, "invalid input size. Expecting numerator to be of size 2N");
      }

      // In the case where deg(P)=2N, I can transform numerator to Reversed-evals -> The second half is
      // a reversed-coset of size N with coset-gen the 2N-th root of unity.
      const int N = vanishing_poly_degree;
      numerator->transform_to_evaluations(2 * N, true /*=reversed*/);
      // allocate output in coeffs because it will be calculated on a coset but I don't have such a state so will have
      // to INTT back to coeffs
      auto numerator_evals_reversed_p = get_context_storage_immutable<I>(numerator);
      out->allocate(N, State::Coefficients, false /*=set zeros*/);
      auto out_evals_reversed_p = get_context_storage_mutable<I>(out);

      auto ntt_config = default_ntt_config<C>();
      ICICLE_CHECK(get_root_of_unity_from_domain<C>((uint64_t)log2(2 * N), ntt_config.coset_gen));
      // compute inv(u^N-1);
      D v_coset_eval = D::inverse(D::pow(ntt_config.coset_gen, N) - D::one());

      const int NOF_THREADS = 128;
      const int NOF_BLOCKS = (N + NOF_THREADS - 1) / NOF_THREADS;
      mul_scalar_kernel<<<NOF_BLOCKS, NOF_THREADS, 0, m_stream>>>(
        numerator_evals_reversed_p + N /*second half is the reversed coset*/, v_coset_eval, N, out_evals_reversed_p);

      // INTT back from reversed evals on coset to coeffs
      ntt_config.are_inputs_on_device = true;
      ntt_config.are_outputs_on_device = true;
      ntt_config.is_async = true;
      ntt_config.ordering = Ordering::kRN;
      ntt(out_evals_reversed_p, N, NTTDir::kInverse, ntt_config, out_evals_reversed_p);

      CHK_LAST();
    }

    void divide_by_vanishing_case_N(PolyContext out, PolyContext numerator, uint64_t vanishing_poly_degree)
    {
      // in that special case the numertaor has N elements and output will be N elements
      if (numerator->get_nof_elements() != vanishing_poly_degree) {
        THROW_ICICLE_ERR(IcicleError_t::UndefinedError, "invalid input size. Expecting numerator to be of size N");
      }

      const int N = vanishing_poly_degree;
      numerator->transform_to_coefficients(N);
      auto numerator_evals_reversed_p = get_context_storage_immutable<I>(numerator);
      out->allocate(N, State::Coefficients, false /*=set zeros*/);
      auto out_evals_reversed_p = get_context_storage_mutable<I>(out);

      // (1) NTT numerator to coset evals (directly to out)
      auto ntt_config = default_ntt_config<C>();
      ICICLE_CHECK(get_root_of_unity_from_domain<C>((uint64_t)log2(2 * N), ntt_config.coset_gen));
      ntt_config.are_inputs_on_device = true;
      ntt_config.are_outputs_on_device = true;
      ntt_config.is_async = true;
      ntt_config.ordering = Ordering::kNM;
      ntt(numerator_evals_reversed_p, N, NTTDir::kForward, ntt_config, out_evals_reversed_p);

      // (2) divide by constant value (that V(x) evaluates to on the coset)
      D v_coset_eval = D::inverse(D::pow(ntt_config.coset_gen, N) - D::one());

      const int NOF_THREADS = 128;
      const int NOF_BLOCKS = (N + NOF_THREADS - 1) / NOF_THREADS;
      mul_scalar_kernel<<<NOF_BLOCKS, NOF_THREADS, 0, m_stream>>>(
        out_evals_reversed_p, v_coset_eval, N, out_evals_reversed_p);

      // (3) INTT back from coset to coeffs
      ntt_config.are_inputs_on_device = true;
      ntt_config.are_outputs_on_device = true;
      ntt_config.is_async = true;
      ntt_config.ordering = Ordering::kMN;
      ntt(out_evals_reversed_p, N, NTTDir::kInverse, ntt_config, out_evals_reversed_p);

      CHK_LAST();
    }

    // arithmetic with monomials
    void add_monomial_inplace(PolyContext& poly, C monomial_coeff, uint64_t monomial) override
    {
      const uint64_t new_nof_elements = max(poly->get_nof_elements(), monomial + 1);
      poly->transform_to_coefficients(new_nof_elements);
      auto coeffs = get_context_storage_mutable(poly);
      add_single_element_inplace<<<1, 1, 0, m_stream>>>(coeffs + monomial, monomial_coeff);

      CHK_LAST();
    }

    void sub_monomial_inplace(PolyContext& poly, C monomial_coeff, uint64_t monomial) override
    {
      add_monomial_inplace(poly, C::zero() - monomial_coeff, monomial);
    }

    int64_t degree(PolyContext p) override { return degree_internal(p, p->get_nof_elements()); }

    // search degree starting from len, searching down (towards coeff0)
    int64_t degree_internal(PolyContext p, uint64_t len)
    {
      // TODO: parallelize kernel? Note that typically the largest coefficient is expected in the higher half since
      // memory is allocate based on #coefficients

      auto [coeff, _] = p->get_coefficients();

      int64_t h_degree;
      highest_non_zero_idx<<<1, 1, 0, m_stream>>>(coeff, len, d_degree);
      ICICLE_CHECK(icicle_copy_async(&h_degree, d_degree, sizeof(int64_t), m_stream));
      ICICLE_CHECK(icicle_stream_synchronize(m_stream)); // sync to make sure return value is copied to host

      return h_degree;
    }

  public:
    void evaluate(PolyContext p, const D* x, I* eval) override
    {
      // TODO Yuval: maybe use Horner's rule and just evaluate each domain point per thread. Alternatively Need to
      // reduce in parallel.

      auto [coeff, nof_coeff] = p->get_coefficients();

      const bool is_x_on_host = is_host_ptr(x);
      const bool is_eval_on_host = is_host_ptr(eval);

      const D* d_x = x;
      D* allocated_x = nullptr;
      if (is_x_on_host) {
        ICICLE_CHECK(icicle_malloc_async((void**)&allocated_x, sizeof(I), m_stream));
        ICICLE_CHECK(icicle_copy_async(allocated_x, x, sizeof(I), m_stream));
        d_x = allocated_x;
      }
      I* d_eval = eval;
      if (is_eval_on_host) { ICICLE_CHECK(icicle_malloc_async((void**)&d_eval, sizeof(I), m_stream)); }

      // TODO Yuval: other methods can avoid this allocation. Also for eval_on_domain() no need to reallocate every time
      I* d_tmp = nullptr;
      ICICLE_CHECK(icicle_malloc_async((void**)&d_tmp, sizeof(I) * nof_coeff, m_stream));
      const int NOF_THREADS = 32;
      const int NOF_BLOCKS = (nof_coeff + NOF_THREADS - 1) / NOF_THREADS;
      evaluate_polynomial_without_reduction<<<NOF_BLOCKS, NOF_THREADS, 0, m_stream>>>(
        d_x, coeff, nof_coeff, d_tmp); // TODO Yuval: parallelize kernel
      dummy_reduce<<<1, 1, 0, m_stream>>>(d_tmp, nof_coeff, d_eval);

      if (is_eval_on_host) {
        ICICLE_CHECK(icicle_copy_async(eval, d_eval, sizeof(I), m_stream));
        ICICLE_CHECK(icicle_stream_synchronize(m_stream)); // sync to make sure return value is copied to host
        ICICLE_CHECK(icicle_free_async(d_eval, m_stream));
      }
      if (allocated_x) { ICICLE_CHECK(icicle_free_async(allocated_x, m_stream)); }
      ICICLE_CHECK(icicle_free_async(d_tmp, m_stream));
    }

    void evaluate_on_domain(PolyContext p, const D* domain, uint64_t size, I* evaluations /*OUT*/) override
    {
      // TODO Yuval: implement more efficiently ??
      for (uint64_t i = 0; i < size; ++i) {
        evaluate(p, &domain[i], &evaluations[i]);
      }
    }

    void evaluate_on_rou_domain(PolyContext p, uint64_t domain_log_size, I* evals /*OUT*/) override
    {
      const uint64_t poly_size = p->get_nof_elements();
      const uint64_t domain_size = 1 << domain_log_size;
      const bool is_evals_on_host = is_host_ptr(evals);

      I* d_evals = evals;
      // if evals on host, allocate memory
      if (is_evals_on_host) { ICICLE_CHECK(icicle_malloc_async((void**)&d_evals, domain_size * sizeof(I), m_stream)); }

      // If domain size is smaller the polynomial size -> transform to evals and copy the evals with stride.
      // Else, if in coeffs copy coeffs to evals mem and NTT inplace to compute the evals, else INTT to d_evals and back
      // inplace to larger domain
      const bool is_domain_size_smaller_than_poly_size = domain_size <= poly_size;
      if (is_domain_size_smaller_than_poly_size) {
        // TODO Yuval: in reversed evals, can reverse the first 'domain_size' elements to d_evals instead of
        // transforming back to evals.
        p->transform_to_evaluations();
        const auto stride = poly_size / domain_size;
        const int NOF_THREADS = 128;
        const int NOF_BLOCKS = (domain_size + NOF_THREADS - 1) / NOF_THREADS;
        slice_kernel<<<NOF_BLOCKS, NOF_THREADS, 0, m_stream>>>(
          get_context_storage_immutable<I>(p), d_evals, 0 /*offset*/, stride, domain_size);
      } else {
        ICICLE_CHECK(icicle_memset(d_evals, 0, domain_size * sizeof(I)));
        auto ntt_config = default_ntt_config<D>();
        ntt_config.are_inputs_on_device = true;
        ntt_config.are_outputs_on_device = true;
        ntt_config.is_async = true;
        // TODO Yuval: in evals I can NTT directly to d_evals without changing my state
        switch (p->get_state()) {
        case State::Coefficients: {
          // copy to evals memory and inplace NTT of domain size
          ICICLE_CHECK(icicle_copy(d_evals, get_context_storage_immutable<I>(p), poly_size * sizeof(I)));
          ntt_config.ordering = Ordering::kNN;
          ntt(d_evals, domain_size, NTTDir::kForward, ntt_config, d_evals);
        } break;
        case State::EvaluationsOnRou_Natural:
        case State::EvaluationsOnRou_Reversed: {
          const bool is_from_natrual = p->get_state() == State::EvaluationsOnRou_Natural;
          // INTT to coeffs and back to evals
          ntt_config.ordering = is_from_natrual ? Ordering::kNM : Ordering::kRN;
          ntt(get_context_storage_immutable<I>(p), poly_size, NTTDir::kInverse, ntt_config, d_evals);
          ntt_config.ordering = is_from_natrual ? Ordering::kMN : Ordering::kNN;
          ntt(d_evals, poly_size, NTTDir::kForward, ntt_config, d_evals);
        } break;
        default:
          THROW_ICICLE_ERR(IcicleError_t::UndefinedError, "Invalid state to compute evaluations");
          break;
        }
      }

      // release memory if allocated
      if (is_evals_on_host) {
        ICICLE_CHECK(icicle_copy_async(evals, d_evals, domain_size * sizeof(I), m_stream));
        ICICLE_CHECK(icicle_free_async(d_evals, m_stream));
      }

      // sync since user cannot reuse this stream so need to make sure evals are computed
      ICICLE_CHECK(icicle_stream_synchronize(m_stream)); // sync to make sure return value is copied to host

      CHK_LAST();
    }

    uint64_t copy_coeffs(PolyContext op, C* out_coeffs, uint64_t start_idx, uint64_t end_idx) override
    {
      const uint64_t nof_coeffs = op->get_nof_elements();
      if (nullptr == out_coeffs) { return nof_coeffs; } // no allocated memory

      const bool is_valid_start_idx = start_idx < nof_coeffs;
      const bool is_valid_end_idx = end_idx < nof_coeffs && end_idx >= start_idx;
      const bool is_valid_indices = is_valid_start_idx && is_valid_end_idx;
      if (!is_valid_indices) {
        // return -1 instead? I could but 'get_coeff()' cannot with its current declaration
        THROW_ICICLE_ERR(IcicleError_t::InvalidArgument, "copy_coeffs() invalid indices");
      }

      op->transform_to_coefficients();
      auto [device_coeffs, _] = op->get_coefficients();
      const size_t nof_coeffs_to_copy = end_idx - start_idx + 1;
      ICICLE_CHECK(icicle_copy_async(out_coeffs, device_coeffs + start_idx, nof_coeffs_to_copy * sizeof(C), m_stream));
      ICICLE_CHECK(icicle_stream_synchronize(m_stream)); // sync to make sure return value is copied

      return nof_coeffs_to_copy;
    }

    // read coefficients to host
    C get_coeff(PolyContext op, uint64_t coeff_idx) override
    {
      C host_coeff;
      copy_coeffs(op, &host_coeff, coeff_idx, coeff_idx);
      return host_coeff;
    }

    std::tuple<IntegrityPointer<C>, uint64_t /*size*/> get_coefficients_view(PolyContext p) override
    {
      return p->get_coefficients_view();
    }

    inline void assert_device_compatability(const std::list<PolyContext>& polys) const
    {
      // TODO Yuval : move to context class
      for (const PolyContext& p : polys) {
        ICICLE_CHECK(icicle_is_active_device_memory(get_context_storage_immutable(p)));
      }
    }
  };

  /*============================== Polynomial CUDA-factory ==============================*/

#include "icicle/fields/field_config.h"

  template <typename C = scalar_t, typename D = C, typename I = C>
  class CUDAPolynomialFactory : public AbstractPolynomialFactory<C, D, I>
  {
    std::vector<cudaStream_t> m_device_streams; // device-id --> device stream

  public:
    CUDAPolynomialFactory();
    ~CUDAPolynomialFactory();
    std::shared_ptr<IPolynomialContext<C, D, I>> create_context() override;
    std::shared_ptr<IPolynomialBackend<C, D, I>> create_backend() override;
  };

  template <typename C, typename D, typename I>
  CUDAPolynomialFactory<C, D, I>::CUDAPolynomialFactory()
  {
    int nof_cuda_devices = -1;
    CHK_STICKY(cudaGetDeviceCount(&nof_cuda_devices));
    int orig_device = -1;

    CHK_STICKY(cudaGetDevice(&orig_device));
    m_device_streams.resize(nof_cuda_devices, nullptr);

    for (int dev_id = 0; dev_id < nof_cuda_devices; ++dev_id) {
      CHK_STICKY(cudaSetDevice(dev_id));
      CHK_STICKY(cudaStreamCreate(&m_device_streams[dev_id]));
    }
    CHK_STICKY(cudaSetDevice(orig_device)); // setting back original device
  }

  template <typename C, typename D, typename I>
  CUDAPolynomialFactory<C, D, I>::~CUDAPolynomialFactory()
  {
    for (auto stream_it : m_device_streams) {
      CHK_STICKY(cudaStreamDestroy(stream_it));
    }
  }

  template <typename C, typename D, typename I>
  std::shared_ptr<IPolynomialContext<C, D, I>> CUDAPolynomialFactory<C, D, I>::create_context()
  {
    int cuda_device_id = -1;
    CHK_STICKY(cudaGetDevice(&cuda_device_id));
    return std::make_shared<DefaultPolynomialContext<C, D, I>>(m_device_streams[cuda_device_id]);
  }

  template <typename C, typename D, typename I>
  std::shared_ptr<IPolynomialBackend<C, D, I>> CUDAPolynomialFactory<C, D, I>::create_backend()
  {
    int cuda_device_id = -1;
    CHK_STICKY(cudaGetDevice(&cuda_device_id));
    return std::make_shared<DefaultPolynomialBackend<C, D, I>>(m_device_streams[cuda_device_id]);
  }

  /************************************** BACKEND REGISTRATION **************************************/

  REGISTER_SCALAR_POLYNOMIAL_FACTORY_BACKEND("CUDA", CUDAPolynomialFactory<scalar_t>)

} // namespace polynomials