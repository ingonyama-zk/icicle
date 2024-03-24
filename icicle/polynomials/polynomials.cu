#include "polynomials.h"

namespace polynomials {

  template <typename C, typename D, typename I>
  Polynomial<C, D, I>::Polynomial()
  {
    if (nullptr == s_factory) {
      throw std::runtime_error("Polynomial factory not initialized. Must call Polynomial::initialize(factory)");
    }
    m_context = s_factory->create_context();
    m_backend = s_factory->create_backend();
  }

  template <typename C, typename D, typename I>
  Polynomial<C, D, I> Polynomial<C, D, I>::from_coefficients(const C* coefficients, uint64_t nof_coefficients)
  {
    Polynomial P = {};
    P.m_context->init_from_coefficients(nof_coefficients, coefficients);
    return P;
  }

  template <typename C, typename D, typename I>
  Polynomial<C, D, I> Polynomial<C, D, I>::from_rou_evaluations(const I* evaluations, uint64_t nof_evaluations)
  {
    Polynomial P = {};
    P.m_context->init_from_rou_evaluations(nof_evaluations, evaluations);
    return P;
  }

  template <typename C, typename D, typename I>
  Polynomial<C, D, I> Polynomial<C, D, I>::clone() const
  {
    Polynomial P = {};
    P.m_context = m_context->clone();
    return P;
  }

  template <typename C, typename D, typename I>
  Polynomial<C, D, I> Polynomial<C, D, I>::operator+(const Polynomial<C, D, I>& rhs) const
  {
    Polynomial<C, D, I> res = {};
    m_backend->add(*res.m_context.get(), *m_context, *rhs.m_context);
    return res;
  }

  template <typename C, typename D, typename I>
  Polynomial<C, D, I> Polynomial<C, D, I>::operator-(const Polynomial<C, D, I>& rhs) const
  {
    Polynomial<C, D, I> res = {};
    m_backend->subtract(*res.m_context.get(), *m_context, *rhs.m_context);
    return res;
  }

  template <typename C, typename D, typename I>
  Polynomial<C, D, I> Polynomial<C, D, I>::operator*(const Polynomial& rhs) const
  {
    Polynomial<C, D, I> res = {};
    m_backend->multiply(*res.m_context.get(), *m_context, *rhs.m_context);
    return res;
  }

  template <typename C, typename D, typename I>
  Polynomial<C, D, I> Polynomial<C, D, I>::operator*(const C& c) const
  {
    Polynomial rhs = {};
    rhs.m_context->init_from_coefficients(1 /*nof_coefficients*/, &c);
    return *this * rhs;
  }

  template <typename C, typename D, typename I>
  Polynomial<C, D, I> operator*(const C& v, const Polynomial<C, D, I>& rhs)
  {
    return rhs * v;
  }

  template <typename C, typename D, typename I>
  std::pair<Polynomial<C, D, I>, Polynomial<C, D, I>> Polynomial<C, D, I>::divide(const Polynomial<C, D, I>& rhs) const
  {
    Polynomial<C, D, I> Q = {}, R = {};
    m_backend->divide(*Q.m_context.get(), *R.m_context.get(), *m_context, *rhs.m_context.get());
    return std::make_pair(std::move(Q), std::move(R));
  }

  template <typename C, typename D, typename I>
  Polynomial<C, D, I> Polynomial<C, D, I>::divide_by_vanishing_polynomial(uint64_t vanishing_polynomial_degree) const
  {
    Polynomial<C, D, I> res = {};
    m_backend->divide_by_vanishing_polynomial(*res.m_context.get(), *m_context, vanishing_polynomial_degree);
    return res;
  }

  template <typename C, typename D, typename I>
  Polynomial<C, D, I>& Polynomial<C, D, I>::operator+=(const Polynomial& rhs)
  {
    m_backend->add(*m_context, *m_context, *rhs.m_context);
    return *this;
  }

  template <typename C, typename D, typename I>
  Polynomial<C, D, I>& Polynomial<C, D, I>::add_monomial_inplace(C monomial_coeff, uint64_t monomial)
  {
    m_backend->add_monomial_inplace(*m_context, monomial_coeff, monomial);
    return *this;
  }

  template <typename C, typename D, typename I>
  Polynomial<C, D, I>& Polynomial<C, D, I>::sub_monomial_inplace(C monomial_coeff, uint64_t monomial)
  {
    m_backend->sub_monomial_inplace(*m_context, monomial_coeff, monomial);
    return *this;
  }

  template <typename C, typename D, typename I>
  I Polynomial<C, D, I>::operator()(const D& x) const
  {
    return evaluate(x);
  }

  template <typename C, typename D, typename I>
  I Polynomial<C, D, I>::evaluate(const D& x) const
  {
    return m_backend->evaluate(*m_context, x);
  }

  template <typename C, typename D, typename I>
  int64_t Polynomial<C, D, I>::degree()
  {
    return m_backend->degree(*m_context);
  }

  template <typename C, typename D, typename I>
  C Polynomial<C, D, I>::copy_coefficient_to_host(uint64_t idx) const
  {
    return m_backend->copy_coefficient_to_host(*m_context, idx);
  }

  template <typename C, typename D, typename I>
  int64_t Polynomial<C, D, I>::copy_coefficients_to_host(C* host_coeffs, int64_t start_idx, int64_t end_idx) const
  {
    return m_backend->copy_coefficients_to_host(*m_context, host_coeffs, start_idx, end_idx);
  }

  template <typename C, typename D, typename I>
  std::tuple<IntegrityPointer<C>, uint64_t /*size*/, uint64_t /*device_id*/>
  Polynomial<C, D, I>::get_coefficients_view()
  {
    return m_context->get_coefficients_view();
  }

  template <typename C, typename D, typename I>
  std::tuple<IntegrityPointer<I>, uint64_t /*size*/, uint64_t /*device_id*/>
  Polynomial<C, D, I>::get_rou_evaluations_view(uint64_t nof_evaluations, bool is_reversed)
  {
    return m_context->get_rou_evaluations_view(nof_evaluations, is_reversed);
  }

  // explicit instantiation for default type (scalar field)
  template class Polynomial<>;
  template Polynomial<> operator*(const scalar_t& c, const Polynomial<>& rhs);

} // namespace polynomials