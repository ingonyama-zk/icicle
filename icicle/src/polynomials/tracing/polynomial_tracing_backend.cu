
#include "polynomials/polynomial_context.h"
#include "polynomials/polynomial_backend.h"
#include "polynomials/tracing/polynomial_tracing_backend.cuh"
#include "polynomials/tracing/interpreter.h"

namespace polynomials {

  template <typename C = scalar_t, typename D = C, typename I = C>
  class TracingPolynomialBackend : public IPolynomialBackend<C, D, I>
  {
    typedef std::shared_ptr<IPolynomialContext<C, D, I>> PolyContext;
    typedef typename IPolynomialContext<C, D, I>::State State;

    std::shared_ptr<IPolynomialBackend<C, D, I>> m_compute_backend; // delegating compute to compute backend

  public:
    TracingPolynomialBackend(std::shared_ptr<IPolynomialBackend<C, D, I>> compute_backend)
        : m_compute_backend(compute_backend)
    {
    }
    ~TracingPolynomialBackend() = default;

    static inline std::shared_ptr<TracingPolynomialContext<C, D, I>> as_tracing_context(PolyContext context)
    {
      // Note: tracing backend is coupled to tracing context by design.
      //       Therefore we safely assume the cast is valid and avoid a dynamic cast
      return std::static_pointer_cast<TracingPolynomialContext<C, D, I>>(context);
    }

    void from_coefficients(PolyContext p, uint64_t nof_coefficients, const C* coefficients) override
    {
      auto trace_ctxt = as_tracing_context(p);
      trace_ctxt->m_opcode = eOpcode::FROM_COEFFS;
      trace_ctxt->from_coefficients(nof_coefficients, coefficients);
    }

    void from_rou_evaluations(PolyContext p, uint64_t nof_evaluations, const I* evaluations) override
    {
      auto trace_ctxt = as_tracing_context(p);
      trace_ctxt->m_opcode = eOpcode::FROM_ROU_EVALS;
      trace_ctxt->from_rou_evaluations(nof_evaluations, evaluations);
    }

    void clone(PolyContext out, PolyContext in) override
    {
      auto trace_ctxt = as_tracing_context(out);
      trace_ctxt->m_opcode = eOpcode::CLONE;
      trace_ctxt->set_operands({as_tracing_context(in)});
    }

    void slice(PolyContext out, PolyContext in, uint64_t offset, uint64_t stride, uint64_t size) override
    {
      auto trace_ctxt = as_tracing_context(out);
      trace_ctxt->m_opcode = eOpcode::SLICE;
      trace_ctxt->set_operands({as_tracing_context(in)});
      trace_ctxt->m_attrs.setAttribute(OP_ATTR_OFFSET, offset);
      trace_ctxt->m_attrs.setAttribute(OP_ATTR_STRIDE, stride);
      trace_ctxt->m_attrs.setAttribute(OP_ATTR_SIZE, size);
    }

    void add(PolyContext& res, PolyContext a, PolyContext b) override
    {
      const bool is_inplace = res.get() == a.get();
      auto trace_ctxt = as_tracing_context(res);
      if (is_inplace) {
        // create new trace node and share the memory context
        auto new_trace_ctxt = TracingPolynomialContext<C, D, I>::create(trace_ctxt->m_memory_context);
        trace_ctxt = new_trace_ctxt;
        res = trace_ctxt;
      }
      trace_ctxt->m_opcode = eOpcode::ADD;
      trace_ctxt->set_operands({as_tracing_context(a), as_tracing_context(b)});
    }

    void subtract(PolyContext res, PolyContext a, PolyContext b) override
    {
      auto trace_ctxt = as_tracing_context(res);
      trace_ctxt->m_opcode = eOpcode::SUB;
      trace_ctxt->set_operands({as_tracing_context(a), as_tracing_context(b)});
    }

    void multiply(PolyContext res, PolyContext a, PolyContext b) override
    {
      auto trace_ctxt = as_tracing_context(res);
      trace_ctxt->m_opcode = eOpcode::MUL;
      trace_ctxt->set_operands({as_tracing_context(a), as_tracing_context(b)});
    }

    // scalar multiplication
    void multiply(PolyContext res, PolyContext a, D scalar) override
    {
      auto trace_ctxt = as_tracing_context(res);
      trace_ctxt->m_opcode = eOpcode::SCALAR_MUL;
      trace_ctxt->set_operands({as_tracing_context(a)});
      trace_ctxt->m_attrs.setAttribute(OP_ATTR_SCALAR, scalar);
    }

    void quotient(PolyContext Q, PolyContext a, PolyContext b) override
    {
      auto trace_ctxt = as_tracing_context(Q);
      trace_ctxt->m_opcode = eOpcode::QUOTIENT;
      trace_ctxt->set_operands({as_tracing_context(a), as_tracing_context(b)});
    }

    void remainder(PolyContext R, PolyContext a, PolyContext b) override
    {
      auto trace_ctxt = as_tracing_context(R);
      trace_ctxt->m_opcode = eOpcode::REMAINDER;
      trace_ctxt->set_operands({as_tracing_context(a), as_tracing_context(b)});
    }

    void divide_by_vanishing_polynomial(PolyContext out, PolyContext numerator, uint64_t vanishing_poly_degree) override
    {
      auto trace_ctxt = as_tracing_context(out);
      trace_ctxt->m_opcode = eOpcode::DIV_BY_VANISHING;
      trace_ctxt->m_attrs.setAttribute(OP_ATTR_DEGREE, vanishing_poly_degree);
      trace_ctxt->set_operands({as_tracing_context(numerator)});
    }

    // arithmetic with monomials
    void add_monomial_inplace(PolyContext& p, C monomial_coeff, uint64_t monomial) override
    {
      auto inplace_modified_context =
        TracingPolynomialContext<C, D, I>::create(as_tracing_context(p)->m_memory_context);
      inplace_modified_context->m_opcode = eOpcode::ADD_MONOMIAL_INPLACE;
      inplace_modified_context->m_attrs.setAttribute("monomial_coeff", monomial_coeff);
      inplace_modified_context->m_attrs.setAttribute("monomial", monomial);
      inplace_modified_context->set_operands({as_tracing_context(p)});
      p = inplace_modified_context;
    }

    void sub_monomial_inplace(PolyContext& p, C monomial_coeff, uint64_t monomial) override
    {
      auto inplace_modified_context =
        TracingPolynomialContext<C, D, I>::create(as_tracing_context(p)->m_memory_context);
      inplace_modified_context->m_opcode = eOpcode::SUB_MONOMIAL_INPLACE;
      inplace_modified_context->m_attrs.setAttribute("monomial_coeff", monomial_coeff);
      inplace_modified_context->m_attrs.setAttribute("monomial", monomial);
      inplace_modified_context->set_operands({as_tracing_context(p)});
      p = inplace_modified_context;
    }

    // Following ops invoke trace evaluation:

    void divide(PolyContext Q /*OUT*/, PolyContext R /*OUT*/, PolyContext a, PolyContext b) override
    {
      // constructing Q seperately. Later they may be evaluated together.
      quotient(Q, a, b);
      remainder(R, a, b);
    }

    int64_t degree(PolyContext p) override
    {
      evaluate_expression(p);
      return m_compute_backend->degree(p);
    }

    I evaluate(PolyContext p, const D& domain_x) override
    {
      evaluate_expression(p);
      return m_compute_backend->evaluate(p, domain_x);
    }

    void evaluate_on_domain(PolyContext p, const D* domain, uint64_t size, I* evaluations /*OUT*/) override
    {
      evaluate_expression(p);
      return m_compute_backend->evaluate_on_domain(p, domain, size, evaluations);
    }

    int64_t copy_coefficients_to_host(PolyContext p, C* host_coeffs, int64_t start_idx, int64_t end_idx) override
    {
      evaluate_expression(p);
      return m_compute_backend->copy_coefficients_to_host(p, host_coeffs, start_idx, end_idx);
    }

    // read coefficients to host
    C copy_coefficient_to_host(PolyContext p, uint64_t coeff_idx) override
    {
      evaluate_expression(p);
      return m_compute_backend->copy_coefficient_to_host(p, coeff_idx);
    }

    std::tuple<IntegrityPointer<C>, uint64_t /*size*/, uint64_t /*device_id*/>
    get_coefficients_view(PolyContext p) override
    {
      evaluate_expression(p);
      return m_compute_backend->get_coefficients_view(p);
    }

    std::tuple<IntegrityPointer<I>, uint64_t /*size*/, uint64_t /*device_id*/>
    get_rou_evaluations_view(PolyContext p, uint64_t nof_evaluations, bool is_reversed) override
    {
      evaluate_expression(p);
      return m_compute_backend->get_rou_evaluations_view(p, nof_evaluations, is_reversed);
    }

    void evaluate_expression(PolyContext p)
    {
      Interpreter interpreter{m_compute_backend};
      interpreter.run(as_tracing_context(p));
    }
  };

  /*============================== Polynomial Tracing-context ==============================*/
  template <typename C, typename D, typename I>
  void TracingPolynomialContext<C, D, I>::bind()
  {
    m_bound = true;
  }
  template <typename C, typename D, typename I>
  void TracingPolynomialContext<C, D, I>::unbind()
  {
    m_bound = false;
  }
  template <typename C, typename D, typename I>
  bool TracingPolynomialContext<C, D, I>::is_bound() const
  {
    return m_bound;
  }

  template <typename C, typename D, typename I>
  void TracingPolynomialContext<C, D, I>::set_memory_context(std::shared_ptr<IPolynomialContext<C, D, I>>)
  {
  }

  template <typename C, typename D, typename I>
  void TracingPolynomialContext<C, D, I>::set_operands(std::vector<SharedTracingContext>&& operands)
  {
    m_operands = std::move(operands);
    WeakTracingContext weak_this = TracingPolynomialContext<C, D, I>::shared_from_this();
    for (auto& op : m_operands) {
      op->m_dependents.insert(weak_this);
    }
  }

  template <typename C, typename D, typename I>
  void TracingPolynomialContext<C, D, I>::clear_operands()
  {
    WeakTracingContext weak_this = TracingPolynomialContext<C, D, I>::shared_from_this();
    for (auto& op : m_operands) {
      op->m_dependents.erase(weak_this);
    }
    m_operands.clear();
  }

  template <typename C, typename D, typename I>
  const std::set<
    std::weak_ptr<TracingPolynomialContext<C, D, I>>,
    std::owner_less<std::weak_ptr<TracingPolynomialContext<C, D, I>>>>&
  TracingPolynomialContext<C, D, I>::get_dependents() const
  {
    return m_dependents;
  }

  template <typename C, typename D, typename I>
  std::shared_ptr<TracingPolynomialContext<C, D, I>> TracingPolynomialContext<C, D, I>::get_operand(unsigned idx)
  {
    if (idx >= m_operands.size()) {
      THROW_ICICLE_ERR(IcicleError_t::InvalidArgument, "TracingPolynomialContext::get_operand() invalid operand idx");
    }
    return m_operands[idx];
  }

  template <typename C, typename D, typename I>
  std::shared_ptr<IPolynomialContext<C, D, I>> TracingPolynomialContext<C, D, I>::get_op_mem_ctxt(unsigned idx)
  {
    return get_operand(idx)->m_memory_context;
  }

  template <typename C, typename D, typename I>
  const std::vector<std::shared_ptr<TracingPolynomialContext<C, D, I>>>&
  TracingPolynomialContext<C, D, I>::get_operands()
  {
    return m_operands;
  }

  template <typename C, typename D, typename I>
  bool TracingPolynomialContext<C, D, I>::is_evaluated() const
  {
    return m_operands.empty(); // no operands signals the node is evaluated
  }

  template <typename C, typename D, typename I>
  void TracingPolynomialContext<C, D, I>::from_coefficients(uint64_t nof_coefficients, const C* coefficients)
  {
    m_memory_context->from_coefficients(nof_coefficients, coefficients);
  }

  template <typename C, typename D, typename I>
  void TracingPolynomialContext<C, D, I>::from_rou_evaluations(uint64_t nof_evaluations, const I* evaluations)
  {
    m_memory_context->from_rou_evaluations(nof_evaluations, evaluations);
  }

  template <typename C, typename D, typename I>
  void TracingPolynomialContext<C, D, I>::clone(IPolynomialContext<C, D, I>& from)
  {
    m_memory_context->clone(from);
  }

  template <typename C, typename D, typename I>
  void TracingPolynomialContext<C, D, I>::allocate(uint64_t nof_elements, State init_state, bool memset_zeros)
  {
    m_memory_context->allocate(nof_elements, init_state, memset_zeros);
  }

  template <typename C, typename D, typename I>
  void TracingPolynomialContext<C, D, I>::release()
  {
    m_memory_context->release();
  }

  template <typename C, typename D, typename I>
  void TracingPolynomialContext<C, D, I>::transform_to_coefficients(uint64_t nof_coefficients)
  {
    m_memory_context->transform_to_coefficients(nof_coefficients);
  }

  template <typename C, typename D, typename I>
  void TracingPolynomialContext<C, D, I>::transform_to_evaluations(uint64_t nof_evaluations, bool is_reversed)
  {
    return m_memory_context->transform_to_evaluations(nof_evaluations, is_reversed);
  }

  template <typename C, typename D, typename I>
  TracingPolynomialContext<C, D, I>::State TracingPolynomialContext<C, D, I>::get_state() const
  {
    return m_memory_context->get_state();
  }

  template <typename C, typename D, typename I>
  uint64_t TracingPolynomialContext<C, D, I>::get_nof_elements() const
  {
    return m_memory_context->get_nof_elements();
  }

  template <typename C, typename D, typename I>
  std::pair<const C*, uint64_t> TracingPolynomialContext<C, D, I>::get_coefficients()
  {
    return m_memory_context->get_coefficients();
  }

  template <typename C, typename D, typename I>
  std::pair<const I*, uint64_t> TracingPolynomialContext<C, D, I>::get_rou_evaluations()
  {
    return m_memory_context->get_rou_evaluations();
  }

  template <typename C, typename D, typename I>
  std::tuple<IntegrityPointer<C>, uint64_t /*size*/, uint64_t /*device_id*/>
  TracingPolynomialContext<C, D, I>::get_coefficients_view()
  {
    return m_memory_context->get_coefficients_view();
  }

  template <typename C, typename D, typename I>
  std::tuple<IntegrityPointer<I>, uint64_t /*size*/, uint64_t /*device_id*/>
  TracingPolynomialContext<C, D, I>::get_rou_evaluations_view(uint64_t nof_evaluations, bool is_reversed)
  {
    return m_memory_context->get_rou_evaluations_view(nof_evaluations, is_reversed);
  }

  template <typename C, typename D, typename I>
  void TracingPolynomialContext<C, D, I>::print(std::ostream& os)
  {
    m_memory_context->print(os);
  }

  template <typename C, typename D, typename I>
  void* TracingPolynomialContext<C, D, I>::get_storage_mutable()
  {
    throw std::runtime_error("Not expected");
  }

  template <typename C, typename D, typename I>
  const void* TracingPolynomialContext<C, D, I>::get_storage_immutable()
  {
    throw std::runtime_error("Not expected");
  }

  /*============================== Polynomial Tracing-factory ==============================*/
  template <typename C, typename D, typename I>
  TracingPolynomialFactory<C, D, I>::TracingPolynomialFactory(
    std::shared_ptr<AbstractPolynomialFactory<C, D, I>> base_factory)
      : m_base_factory(base_factory)
  {
  }

  template <typename C, typename D, typename I>
  std::shared_ptr<IPolynomialContext<C, D, I>> TracingPolynomialFactory<C, D, I>::create_context()
  {
    auto tracing_context = TracingPolynomialContext<C, D, I>::create(m_base_factory->create_context());
    return tracing_context;
  }

  template <typename C, typename D, typename I>
  std::shared_ptr<IPolynomialBackend<C, D, I>> TracingPolynomialFactory<C, D, I>::create_backend()
  {
    auto tracing_backend = std::make_shared<TracingPolynomialBackend<C, D, I>>(m_base_factory->create_backend());
    return tracing_backend;
  }

  template class TracingPolynomialContext<>;
  template class TracingPolynomialBackend<>;
  template class TracingPolynomialFactory<>;

} // namespace polynomials