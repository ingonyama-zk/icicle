
#include "polynomials/polynomial_context.h"
#include "polynomials/polynomial_backend.h"
#include "polynomials/tracing/polynomial_tracing_backend.cuh"

namespace polynomials {

  /*============================== Polynomial Tracing-backend ==============================*/
  template <typename C = scalar_t, typename D = C, typename I = C>
  class TracingPolynomialBackend : public IPolynomialBackend<C, D, I>
  {
    typedef std::shared_ptr<IPolynomialContext<C, D, I>> PolyContext;
    typedef typename IPolynomialContext<C, D, I>::State State;

    std::shared_ptr<IPolynomialBackend<C, D, I>> m_compute_backend; // delegating compute to compute backend

  public:
    TracingPolynomialBackend(std::shared_ptr<IPolynomialBackend<C, D, I>> compute_backend)
        : m_compute_backend{compute_backend}
    {
    }
    ~TracingPolynomialBackend() = default;

    void from_coefficients(PolyContext p, uint64_t nof_coefficients, const C* coefficients) override
    {
      p->m_op.opcode = eOpcode::FROM_COEFFS;
      p->from_coefficients(nof_coefficients, coefficients);
    }

    void from_rou_evaluations(PolyContext p, uint64_t nof_evaluations, const I* evaluations) override
    {
      p->m_op.opcode = eOpcode::FROM_ROU_EVALS;
      p->from_rou_evaluations(nof_evaluations, evaluations);
    }

    void clone(PolyContext out, PolyContext in) override
    { // TODO Yuval: evaluate poly
      out->clone(*in);
    }

    void slice(PolyContext out, PolyContext in, uint64_t offset, uint64_t stride, uint64_t size) override {}

    void add(PolyContext res, PolyContext a, PolyContext b) override
    {
      res->m_op.opcode = eOpcode::ADD;
      res->m_args = {a, b};
    }
    void subtract(PolyContext res, PolyContext a, PolyContext b) override
    {
      res->m_op.opcode = eOpcode::SUB;
      res->m_args = {a, b};
    }

    void multiply(PolyContext res, PolyContext a, PolyContext b) override
    {
      res->m_op.opcode = eOpcode::MUL;
      res->m_args = {a, b};
    }

    void quotient(PolyContext Q, PolyContext a, PolyContext b) override
    {
      Q->m_op.opcode = eOpcode::QUOTIENT;
      Q->m_args = {a, b};
    }

    void remainder(PolyContext R, PolyContext a, PolyContext b) override
    {
      R->m_op.opcode = eOpcode::REMAINDER;
      R->m_args = {a, b};
    }

    void divide_by_vanishing_polynomial(PolyContext out, PolyContext numerator, uint64_t vanishing_poly_degree) override
    {
      out->m_op.opcode = eOpcode::DIV_BY_VANISHING;
      out->m_op.attributes.setAttribute(OP_ATTR_DEGREE, (int)vanishing_poly_degree);
      out->m_args = {numerator};
    }

    // arithmetic with monomials
    void add_monomial_inplace(PolyContext poly, C monomial_coeff, uint64_t monomial) override {}

    void sub_monomial_inplace(PolyContext poly, C monomial_coeff, uint64_t monomial) override {}

    // Following ops invoke trace evaluation:

    void divide(PolyContext Q /*OUT*/, PolyContext R /*OUT*/, PolyContext a, PolyContext b) override {}

    int64_t degree(PolyContext p) override
    {
      // TODO Yuval: evaluate poly
      return -2;
    }

    I evaluate(PolyContext p, const D& domain_x) override
    {
      // TODO Yuval: evaluate poly
      return m_compute_backend->evaluate(p, domain_x);
    }

    void evaluate_on_domain(PolyContext p, const D* domain, uint64_t size, I* evaluations /*OUT*/) override
    {
      // TODO Yuval: evaluate poly
      return m_compute_backend->evaluate_on_domain(p, domain, size, evaluations);
    }

    int64_t
    copy_coefficients_to_host(PolyContext op, C* host_coeffs, int64_t start_idx = 0, int64_t end_idx = -1) override
    {
      // TODO Yuval: evaluate poly
      return -1;
    }

    // read coefficients to host
    C copy_coefficient_to_host(PolyContext op, uint64_t coeff_idx) override
    {
      C host_coeff;
      copy_coefficients_to_host(op, &host_coeff, coeff_idx, coeff_idx);
      return host_coeff;
    }

    std::tuple<IntegrityPointer<C>, uint64_t /*size*/, uint64_t /*device_id*/>
    get_coefficients_view(PolyContext p) override
    {
      // TODO Yuval: evaluate poly
      return p->get_coefficients_view();
    }

    std::tuple<IntegrityPointer<I>, uint64_t /*size*/, uint64_t /*device_id*/>
    get_rou_evaluations_view(PolyContext p, uint64_t nof_evaluations, bool is_reversed) override
    {
      // TODO Yuval: evaluate poly
      return p->get_rou_evaluations_view(nof_evaluations, is_reversed);
    }
  };

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
    return m_base_factory->create_context();
  }

  template <typename C, typename D, typename I>
  std::shared_ptr<IPolynomialBackend<C, D, I>> TracingPolynomialFactory<C, D, I>::create_backend()
  {
    auto tracing_backend = std::make_shared<TracingPolynomialBackend<C, D, I>>(m_base_factory->create_backend());
    return tracing_backend;
  }

  template class TracingPolynomialBackend<>;
  template class TracingPolynomialFactory<>;

} // namespace polynomials