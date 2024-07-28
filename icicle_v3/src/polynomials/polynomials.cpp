#include "icicle/polynomials/polynomials.h"
#include "icicle/fields/field_config.h"

using namespace field_config;


namespace icicle {

  std::shared_ptr<AbstractPolynomialFactory<scalar_t>> get_polynomial_abstract_factory();

  template <typename C, typename D, typename I>
  Polynomial<C, D, I>::Polynomial()
  {
    auto backend_factory = get_polynomial_abstract_factory();
    m_context = backend_factory->create_context();
    m_backend = backend_factory->create_backend();
  }

  template <typename C, typename D, typename I>
  Polynomial<C, D, I> Polynomial<C, D, I>::from_coefficients(const C* coefficients, uint64_t nof_coefficients)
  {
    Polynomial<C, D, I> P = {};
    P.m_backend->from_coefficients(P.m_context, nof_coefficients, coefficients);
    return P;
  }

  template <typename C, typename D, typename I>
  Polynomial<C, D, I> Polynomial<C, D, I>::from_rou_evaluations(const I* evaluations, uint64_t nof_evaluations)
  {
    Polynomial<C, D, I> P = {};
    P.m_backend->from_rou_evaluations(P.m_context, nof_evaluations, evaluations);
    return P;
  }

  template <typename C, typename D, typename I>
  Polynomial<C, D, I> Polynomial<C, D, I>::clone() const
  {
    Polynomial<C, D, I> P = {};
    m_backend->clone(P.m_context, m_context);
    return P;
  }

  template <typename C, typename D, typename I>
  Polynomial<C, D, I> Polynomial<C, D, I>::slice(uint64_t offset, uint64_t stride, uint64_t size)
  {
    Polynomial res = {};
    m_backend->slice(res.m_context, this->m_context, offset, stride, size);
    return res;
  }
  template <typename C, typename D, typename I>
  Polynomial<C, D, I> Polynomial<C, D, I>::even()
  {
    return slice(0, 2, 0 /*all elements*/);
  }
  template <typename C, typename D, typename I>
  Polynomial<C, D, I> Polynomial<C, D, I>::odd()
  {
    return slice(1, 2, 0 /*all elements*/);
  }

  template <typename C, typename D, typename I>
  Polynomial<C, D, I> Polynomial<C, D, I>::operator+(const Polynomial<C, D, I>& rhs) const
  {
    Polynomial<C, D, I> res = {};
    m_backend->add(res.m_context, m_context, rhs.m_context);
    return res;
  }

  template <typename C, typename D, typename I>
  Polynomial<C, D, I> Polynomial<C, D, I>::operator-(const Polynomial<C, D, I>& rhs) const
  {
    Polynomial<C, D, I> res = {};
    m_backend->subtract(res.m_context, m_context, rhs.m_context);
    return res;
  }

  template <typename C, typename D, typename I>
  Polynomial<C, D, I> Polynomial<C, D, I>::operator*(const Polynomial& rhs) const
  {
    Polynomial<C, D, I> res = {};
    m_backend->multiply(res.m_context, m_context, rhs.m_context);
    return res;
  }

  template <typename C, typename D, typename I>
  Polynomial<C, D, I> Polynomial<C, D, I>::operator*(const D& scalar) const
  {
    Polynomial<C, D, I> res = {};
    m_backend->multiply(res.m_context, m_context, scalar);
    return res;
  }

  template <typename C, typename D, typename I>
  Polynomial<C, D, I> operator*(const D& scalar, const Polynomial<C, D, I>& rhs)
  {
    return rhs * scalar;
  }

  template <typename C, typename D, typename I>
  std::pair<Polynomial<C, D, I>, Polynomial<C, D, I>> Polynomial<C, D, I>::divide(const Polynomial<C, D, I>& rhs) const
  {
    Polynomial<C, D, I> Q = {}, R = {};
    m_backend->divide(Q.m_context, R.m_context, m_context, rhs.m_context);
    return std::make_pair(std::move(Q), std::move(R));
  }

  template <typename C, typename D, typename I>
  Polynomial<C, D, I> Polynomial<C, D, I>::operator/(const Polynomial& rhs) const
  {
    Polynomial<C, D, I> res = {};
    m_backend->quotient(res.m_context, m_context, rhs.m_context);
    return res;
  }

  template <typename C, typename D, typename I>
  Polynomial<C, D, I> Polynomial<C, D, I>::operator%(const Polynomial& rhs) const
  {
    Polynomial<C, D, I> res = {};
    m_backend->remainder(res.m_context, m_context, rhs.m_context);
    return res;
  }

  template <typename C, typename D, typename I>
  Polynomial<C, D, I> Polynomial<C, D, I>::divide_by_vanishing_polynomial(uint64_t vanishing_polynomial_degree) const
  {
    Polynomial<C, D, I> res = {};
    m_backend->divide_by_vanishing_polynomial(res.m_context, m_context, vanishing_polynomial_degree);
    return res;
  }

  template <typename C, typename D, typename I>
  Polynomial<C, D, I>& Polynomial<C, D, I>::operator+=(const Polynomial& rhs)
  {
    m_backend->add(m_context, m_context, rhs.m_context);
    return *this;
  }

  template <typename C, typename D, typename I>
  Polynomial<C, D, I>& Polynomial<C, D, I>::add_monomial_inplace(C monomial_coeff, uint64_t monomial)
  {
    m_backend->add_monomial_inplace(m_context, monomial_coeff, monomial);
    return *this;
  }

  template <typename C, typename D, typename I>
  Polynomial<C, D, I>& Polynomial<C, D, I>::sub_monomial_inplace(C monomial_coeff, uint64_t monomial)
  {
    m_backend->sub_monomial_inplace(m_context, monomial_coeff, monomial);
    return *this;
  }

  template <typename C, typename D, typename I>
  I Polynomial<C, D, I>::operator()(const D& x) const
  {
    I eval = {};
    evaluate(&x, &eval);
    return eval;
  }

  template <typename C, typename D, typename I>
  void Polynomial<C, D, I>::evaluate(const D* x, I* eval) const
  {
    m_backend->evaluate(m_context, x, eval);
  }

  template <typename C, typename D, typename I>
  void Polynomial<C, D, I>::evaluate_on_domain(D* domain, uint64_t size, I* evals /*OUT*/) const
  {
    return m_backend->evaluate_on_domain(m_context, domain, size, evals);
  }

  template <typename C, typename D, typename I>
  void Polynomial<C, D, I>::evaluate_on_rou_domain(uint64_t domain_log_size, I* evals /*OUT*/) const
  {
    return m_backend->evaluate_on_rou_domain(m_context, domain_log_size, evals);
  }

  template <typename C, typename D, typename I>
  int64_t Polynomial<C, D, I>::degree()
  {
    return m_backend->degree(m_context);
  }

  template <typename C, typename D, typename I>
  C Polynomial<C, D, I>::get_coeff(uint64_t idx) const
  {
    return m_backend->get_coeff(m_context, idx);
  }

  template <typename C, typename D, typename I>
  uint64_t Polynomial<C, D, I>::copy_coeffs(C* host_coeffs, uint64_t start_idx, uint64_t end_idx) const
  {
    return m_backend->copy_coeffs(m_context, host_coeffs, start_idx, end_idx);
  }

  template <typename C, typename D, typename I>
  std::tuple<IntegrityPointer<C>, uint64_t /*size*/> Polynomial<C, D, I>::get_coefficients_view()
  {
    return m_backend->get_coefficients_view(m_context);
  }

  // explicit instantiation for default type (scalar field)
  template class Polynomial<scalar_t>;
  template Polynomial<scalar_t> operator*(const scalar_t& c, const Polynomial<scalar_t>& rhs);

} // namespace icicle