#pragma once // TODO Yuval remove this

#include "polynomials.h"

namespace polynomials {

  template <typename C, typename D, typename I>
  Polynomial<C, D, I>::Polynomial()
  {
    // TODO Yuval: how to choose backend
    m_backend = std::make_unique<GPUPolynomialBackend<C, D, I>>();
  }

  template <typename C, typename D, typename I>
  Polynomial<C, D, I> Polynomial<C, D, I>::from_coefficients(const C* coefficients, uint32_t nof_coefficients)
  {
    Polynomial P = {};
    P.m_backend->init_from_coefficients(coefficients, nof_coefficients);
    return P;
  }

  template <typename C, typename D, typename I>
  Polynomial<C, D, I> Polynomial<C, D, I>::from_rou_evaluations(const I* evaluations, uint32_t nof_evaluations)
  {
    Polynomial P = {};
    P.m_backend->init_from_rou_evaluations(evaluations, nof_evaluations);
    return P;
  }

  template <typename C, typename D, typename I>
  Polynomial<C, D, I> Polynomial<C, D, I>::operator+(const Polynomial<C, D, I>& rhs) const
  {
    Polynomial<C, D, I> res = {};
    m_backend->add(res, *this, rhs);
    return res;
  }

  template <typename C, typename D, typename I>
  Polynomial<C, D, I> Polynomial<C, D, I>::operator-(const Polynomial<C, D, I>& rhs) const
  {
    Polynomial<C, D, I> res = {};
    m_backend->subtract(res, *this, rhs);
    return res;
  }

  template <typename C, typename D, typename I>
  std::pair<Polynomial<C, D, I>, Polynomial<C, D, I>> Polynomial<C, D, I>::divide(const Polynomial<C, D, I>& rhs) const
  {
    Polynomial<C, D, I> Q = {}, R = {};
    m_backend->divide(Q, R, *this, rhs);
    return std::make_pair(Q, R);
  }

  template <typename C, typename D, typename I>
  Polynomial<C, D, I>& Polynomial<C, D, I>::add_monomial_inplace(C monomial_coeff, uint32_t monomial) const
  {
    m_backend->add_monomial_in_place(*this, monomial_coeff, monomial);
    return *this;
  }

  template <typename C, typename D, typename I>
  I Polynomial<C, D, I>::operator()(const D& x) const
  {
    return m_backend->evaluate(*this, x);
  }

  template <typename C, typename D, typename I>
  int32_t Polynomial<C, D, I>::degree()
  {
    return m_backend->degree(*this);
  }

  template <typename C, typename D, typename I>
  C Polynomial<C, D, I>::get_coefficient(uint32_t idx) const
  {
    return m_backend->get_coefficient(*this, idx);
  }

  template <typename C, typename D, typename I>
  void Polynomial<C, D, I>::get_coefficients(C* coeff, uint32_t& nof_coeff) const
  {
    m_backend->get_coefficients(*this, coeff, nof_coeff);
  }

} // namespace polynomials