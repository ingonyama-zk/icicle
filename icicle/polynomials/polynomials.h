#pragma once

#include <iostream>
#include <memory>

namespace polynomials {
  template <typename CoefficientType, typename DomainType = CoefficientType, typename ImageType = CoefficientType>
  class IPolynomialBackend;
  template <typename CoefficientType, typename DomainType = CoefficientType, typename ImageType = CoefficientType>
  class Polynomial
  {
  public:
    // initialization
    static Polynomial from_coefficients(const CoefficientType* coefficients, uint32_t nof_coefficients);
    static Polynomial from_rou_evaluations(const ImageType* evaluations, uint32_t nof_evaluations);
    // static Polynomial from_evaluations(const DomainType* domain, const ImageType* evaluations, uint32_t size);

    // destruct polynomial (to be called from bindings given handle to polynomial)
    static void destruct(Polynomial& poly) { delete &poly; }

    // arithmetic ops (two polynomials)
    Polynomial operator+(const Polynomial& rhs) const;
    Polynomial operator-(const Polynomial& rhs) const;
    // Polynomial operator*(const Polynomial& rhs) const;
    // Polynomial operator/(const Polynomial& rhs) const; // returns Quotient Q(x) for A(x) = Q(x)B(x) + R(x)
    // Polynomial operator%(const Polynomial& rhs) const; // returns Remainder R(x) for A(x) = Q(x)B(x) + R(x)
    std::pair<Polynomial, Polynomial> divide(const Polynomial& rhs) const; //  returns (Q(x), R(x))
    // Polynomial divide_by_vanishing_polynomial(uint32_t vanishing_polynomial_degree) const;

    // dot-product with coefficients (e.g. MSM when computing P(tau)G1)
    template <typename R>
    R dot_product_with_coefficients(R* points, uint32_t nof_points);

    // // arithmetic ops with monomial
    Polynomial& add_monomial_inplace(CoefficientType monomial_coeff, uint32_t monomial = 0) const;
    // Polynomial& sub_monomial_inplace(CoefficientType monomial_coeff, uint32_t monomial = 0);

    // Polynomial reciprocal() const;

    // // evaluation (caller is allocating output memory, for evalute(...))
    ImageType operator()(const DomainType& x) const;
    // void evaluate(const DomainType& x, ImageType& eval /*OUT*/) const;
    // void evaluate(DomainType* x, uint32_t nof_points, ImageType* evals /*OUT*/) const;

    // // highest non-zero coefficient degree
    int32_t degree();

    CoefficientType get_coefficient(uint32_t idx) const;
    // caller is allocating output memory. If coeff==nullptr, returning nof_coeff only
    void get_coefficients(CoefficientType* coeff, uint32_t& nof_coeff) const;

    friend std::ostream& operator<<(std::ostream& os, Polynomial& poly)
    {
      poly.m_backend->print(os);
      return os;
    }

    std::unique_ptr<IPolynomialBackend<CoefficientType, DomainType, ImageType>> m_backend = nullptr;
    // TODO Yuval: split storage from backend too? may be useful for executing a trace, reusing the compute

  private:
    Polynomial();

  public:
    ~Polynomial() = default;
    // make sure polynomials can be moved but not copied
    Polynomial(Polynomial&&) = default;
    Polynomial& operator=(Polynomial&&) = default;
    Polynomial(const Polynomial&) = delete;
    Polynomial& operator=(const Polynomial&) = delete;
  }; // namespace polynomials

  // TODO Yuval: backend expects inputs although is a polynomial backend itself. Is that a bad thing? Maybe I should
  // make the backend one instance (rather than per polynomial) and then each polynomial would have a storage member
  // instead? what will I gain by doing it?
  template <typename C, typename D, typename I>
  class IPolynomialBackend
  {
  public:
    IPolynomialBackend() : m_id(s_id++) {}
    virtual ~IPolynomialBackend() {}

    // Interface for operations
    virtual void init_from_coefficients(const C* coefficients, uint32_t nof_coefficients) = 0;
    virtual void init_from_rou_evaluations(const I* evaluations, uint32_t nof_evaluations) = 0;

    virtual void print(std::ostream& oss) = 0;

    virtual void add(Polynomial<C, D, I>& result, const Polynomial<C, D, I>& a, const Polynomial<C, D, I>& b) = 0;
    virtual void subtract(Polynomial<C, D, I>& result, const Polynomial<C, D, I>& a, const Polynomial<C, D, I>& b) = 0;
    virtual void divide(
      Polynomial<C, D, I>& Quotient,
      Polynomial<C, D, I>& Remainder,
      const Polynomial<C, D, I>& a,
      const Polynomial<C, D, I>& b) = 0;
    virtual void add_monomial_inplace(Polynomial<C, D, I>& poly, C monomial_coeff, uint32_t monomial) = 0;

    virtual int32_t degree(Polynomial<C, D, I>& poly) = 0;

    virtual I evaluate(Polynomial<C, D, I>& poly, const D& domain_x) = 0;
    virtual void
    evaluate(Polynomial<C, D, I>& poly, const D* domain_x, uint32_t nof_domain_points, I* evaluations /*OUT*/) = 0;

    // TODO Yuval: some backends would not be able to implement this (??)
    virtual C get_coefficient(Polynomial<C, D, I>& poly, uint32_t coeff_idx) = 0;
    // if coefficients==nullptr, fills nof_coeff only
    virtual void get_coefficients(Polynomial<C, D, I>& poly, C* coefficients, uint32_t& nof_coeff) = 0;

  protected:
    // for debug. remove?
    static inline uint32_t s_id = 0;
    const uint32_t m_id;
  };

} // namespace polynomials

#include "gpu_backend/polynomial_gpu_backend.cuh"
#include "polynomials.cpp" // TODO Yuval: avoid include with explicit instantiation?
#include "polynomials_c_api.h"
