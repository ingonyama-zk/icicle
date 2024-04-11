#pragma once

#include <set>

#include "fields/field_config.cuh"
#include "polynomials/polynomials.h"
#include "polynomials/tracing/polynomial_ops.h"

using field_config::scalar_t;

namespace polynomials {

  // TracingPolynomialContext: A class template that adds tracing capabilities to polynomial operations.
  // It decorates an existing IPolynomialContext with additional functionalities like operation tracking.
  template <typename C = scalar_t, typename D = C, typename I = C>
  class TracingPolynomialContext : public IPolynomialContext<C, D, I>,
                                   public std::enable_shared_from_this<TracingPolynomialContext<C, D, I>>
  {
    using typename IPolynomialContext<C, D, I>::State;
    typedef std::shared_ptr<TracingPolynomialContext> SharedTracingContext;
    typedef std::weak_ptr<TracingPolynomialContext> WeakTracingContext;
    typedef std::shared_ptr<IPolynomialContext<C, D, I>> MemoryContext;

  public:
    static SharedTracingContext create(MemoryContext memory_context)
    {
      return std::shared_ptr<TracingPolynomialContext>(new TracingPolynomialContext(memory_context));
    }
    SharedTracingContext getptr() { return TracingPolynomialContext<C, D, I>::shared_from_this(); }
    ~TracingPolynomialContext() = default;

  private:
    // constructor accessed via create() to manage shared/weak ptrs correctly
    TracingPolynomialContext(MemoryContext memory_context) : m_memory_context(memory_context), m_bound(false) {}

  public:
    // Operation code indicating the polynomial operation being traced.
    eOpcode m_opcode = eOpcode::INVALID;
    // The underlying memory context, typically representing a device-context like CUDA or ZPU
    MemoryContext m_memory_context;
    // Additional attributes providing more information about the operation, enhancing traceability.
    Attributes m_attrs;

  private: // members that should be modified carefully
    // A vector of arguments to the operation, represented as shared pointers to other TracingPolynomialContexts,
    // facilitating the construction of a computational graph.
    std::vector<SharedTracingContext> m_operands;

    // Set of nodes that have this node as operand
    std::set<WeakTracingContext, std::owner_less<WeakTracingContext>> m_dependents;

    // Bound to a polynomial. Temporary values are not bound and can be optimized in terms of memory
    bool m_bound;

  public:
    void bind() override;
    void unbind() override;
    bool is_bound() const;

    void set_memory_context(MemoryContext);
    void set_operands(std::vector<SharedTracingContext>&& operands);
    void clear_operands();
    SharedTracingContext get_operand(unsigned idx);
    MemoryContext get_op_mem_ctxt(unsigned idx);
    const std::vector<SharedTracingContext>& get_operands();
    const std::set<WeakTracingContext, std::owner_less<WeakTracingContext>>& get_dependents() const;
    bool is_evaluated() const;

    // Decorator methods overriding IPolynomialContext methods, potentially incorporating tracing logic.
    void from_coefficients(uint64_t nof_coefficients, const C* coefficients = nullptr) override;
    void from_rou_evaluations(uint64_t nof_evaluations, const I* evaluations = nullptr) override;
    void clone(IPolynomialContext<C, D, I>& from) override;
    void allocate(uint64_t nof_elements, State init_state = State::Coefficients, bool memset_zeros = true);
    void release() override;
    void transform_to_coefficients(uint64_t nof_coefficients = 0);
    void transform_to_evaluations(uint64_t nof_evaluations = 0, bool is_reversed = false) override;
    State get_state() const override;
    uint64_t get_nof_elements() const override;
    std::pair<const C*, uint64_t> get_coefficients() override;
    std::pair<const I*, uint64_t> get_rou_evaluations() override;
    std::tuple<IntegrityPointer<C>, uint64_t /*size*/, uint64_t /*device_id*/> get_coefficients_view() override;
    std::tuple<IntegrityPointer<I>, uint64_t /*size*/, uint64_t /*device_id*/>
    get_rou_evaluations_view(uint64_t nof_evaluations = 0, bool is_reversed = false) override;
    void print(std::ostream& os) override;

  protected:
    void* get_storage_mutable() override;
    const void* get_storage_immutable() override;
  };

  // TracingPolynomialFactory: A factory class for creating TracingPolynomialContexts and backends,
  // wrapping an existing AbstractPolynomialFactory to incorporate tracing functionalities.
  template <typename C = scalar_t, typename D = C, typename I = C>
  class TracingPolynomialFactory : public AbstractPolynomialFactory<C, D, I>
  {
  private:
    // The base factory used for creating non-tracing polynomial contexts and backends.
    std::shared_ptr<AbstractPolynomialFactory<C, D, I>> m_base_factory;

  public:
    TracingPolynomialFactory(std::shared_ptr<AbstractPolynomialFactory<C, D, I>> base_factory);
    ~TracingPolynomialFactory() = default;

    // Creates and returns a shared pointer to a TracingPolynomialContext/IPolynomialBackend, enhancing the base
    // factory's context/backend with tracing.
    std::shared_ptr<IPolynomialContext<C, D, I>> create_context() override;
    std::shared_ptr<IPolynomialBackend<C, D, I>> create_backend() override;
  };
} // namespace polynomials
