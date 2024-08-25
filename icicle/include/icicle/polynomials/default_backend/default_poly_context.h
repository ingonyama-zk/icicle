

#include "icicle/runtime.h"
#include "icicle/errors.h"
#include "icicle/ntt.h"
#include "icicle/polynomials/polynomial_context.h"

namespace icicle {

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
    icicleStreamHandle m_stream;

    DefaultPolynomialContext(const icicleStreamHandle stream) : m_stream{stream}
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
        ICICLE_ASSERT(false) << "clone() from non implemented polynomial state";
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

      ICICLE_ASSERT(nof_coefficients >= this->m_nof_elements)
        << "polynomial shrinking not supported. Probably encountered a bug";

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

      ICICLE_ASSERT(nof_evaluations >= this->m_nof_elements)
        << "polynomial shrinking not supported. Probably encountered a bug";

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

} // namespace icicle