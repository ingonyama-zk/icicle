#pragma once

#include <iostream>
#include <memory>

namespace polynomials {
  template <typename CoefficientType, typename DomainType, typename ImageType, typename ECpoint>
  class IPolynomialBackend;

  template <typename CoefficientType, typename DomainType, typename ImageType>
  class IPolynomialContext;

  /*============================== Polynomial API ==============================*/
  template <
    typename CoefficientType = curve_config::scalar_t,
    typename DomainType = CoefficientType,
    typename ImageType = CoefficientType,
    typename ECpoint = curve_config::affine_t>
  class Polynomial
  {
  public:
    // initialization
    static Polynomial from_coefficients(const CoefficientType* coefficients, uint32_t nof_coefficients);
    static Polynomial from_rou_evaluations(const ImageType* evaluations, uint32_t nof_evaluations);
    // static Polynomial from_evaluations(const DomainType* domain, const ImageType* evaluations, uint32_t size);

    // arithmetic ops (two polynomials)
    Polynomial operator+(const Polynomial& rhs) const;
    Polynomial operator-(const Polynomial& rhs) const;
    Polynomial operator*(const Polynomial& rhs) const;
    Polynomial operator/(const Polynomial& rhs) const; // returns Quotient Q(x) for A(x) = Q(x)B(x) + R(x)
    Polynomial operator%(const Polynomial& rhs) const; // returns Remainder R(x) for A(x) = Q(x)B(x) + R(x)
    std::pair<Polynomial, Polynomial> divide(const Polynomial& rhs) const; //  returns (Q(x), R(x))
    Polynomial divide_by_vanishing_polynomial(uint32_t vanishing_polynomial_degree) const;

    // dot-product with coefficients (e.g. MSM when computing P(tau)G1)
    ECpoint dot_product_with_coefficients(ECpoint* points, uint32_t nof_points);

    // arithmetic ops with monomial
    Polynomial& add_monomial_inplace(CoefficientType monomial_coeff, uint32_t monomial = 0) const;
    Polynomial& sub_monomial_inplace(CoefficientType monomial_coeff, uint32_t monomial = 0);

    // evaluation (caller is allocating output memory, for evalute(...))
    ImageType operator()(const DomainType& x) const;
    ImageType evaluate(const DomainType& x) const;
    void evaluate(DomainType* x, uint32_t nof_points, ImageType* evals /*OUT*/) const;

    // highest non-zero coefficient degree
    int32_t degree();

    CoefficientType get_coefficient(uint32_t idx) const;
    // caller is allocating output memory. If coeff==nullptr, returning nof_coeff only
    uint32_t get_coefficients(CoefficientType* coeff) const;

    friend std::ostream& operator<<(std::ostream& os, Polynomial& poly)
    {
      poly.m_context->print(os);
      return os;
    }

  private:
    // context is a wrapper for the polynomial state, including allocated memory, device context etc.
    std::unique_ptr<IPolynomialContext<CoefficientType, DomainType, ImageType>> m_context = nullptr;
    // backend is the actual API implementation
    std::unique_ptr<IPolynomialBackend<CoefficientType, DomainType, ImageType, ECpoint>> m_backend = nullptr;

    Polynomial();

  public:
    ~Polynomial() = default;
    // make sure polynomials can be moved but not copied
    Polynomial(Polynomial&&) = default;
    Polynomial& operator=(Polynomial&&) = default;
    Polynomial(const Polynomial&) = delete;
    Polynomial& operator=(const Polynomial&) = delete;
  };

  /*============================== Polynomial Context ==============================*/
  // Interface for the polynomial state, including memory, device context etc.
  template <typename C, typename D, typename I>
  class IPolynomialContext
  {
  public:
    enum State { Coefficients, EvaluationsOnRou_Natural, EvaluationsOnRou_Reversed };
    static constexpr size_t ElementSize = std::max(sizeof(C), sizeof(I));

    IPolynomialContext() : m_id{s_id++}, m_nof_elements{0} {}
    virtual ~IPolynomialContext() = default;

    virtual void allocate(uint64_t nof_elements, State init_state = State::Coefficients) = 0;
    virtual void release() = 0;
    virtual void set_state(State state) = 0;

    virtual C* init_from_coefficients(uint64_t nof_coefficients, const C* host_coefficients = nullptr) = 0;
    virtual I* init_from_rou_evaluations(uint64_t nof_evalutions, const I* host_evaluations = nullptr) = 0;

    virtual std::pair<C*, uint64_t> get_coefficients() = 0;
    virtual std::pair<I*, uint64_t> get_rou_evaluations() = 0;

    virtual void transform_to_coefficients() = 0;
    virtual void transform_to_evaluations(uint64_t nof_evalutions = 0, bool is_reversed = 0) = 0;

    State get_state() const { return m_state; }
    uint64_t get_nof_elements() const { return m_nof_elements; }

    virtual void print(std::ostream& os) = 0;

  protected:
    State m_state;
    uint64_t m_nof_elements = 0;

    // id for debug
    static inline uint64_t s_id = 0;
    const uint64_t m_id;
  };

  /*============================== Polynomial Backend ==============================*/
  template <typename C, typename D, typename I, typename ECpoint>
  class IPolynomialBackend
  {
  public:
    IPolynomialBackend() = default;
    virtual ~IPolynomialBackend() {}

    typedef IPolynomialContext<C, D, I> PolyContext;

    // arithmetics
    virtual void add(PolyContext& out, PolyContext& op_a, PolyContext& op_b) = 0;
    virtual void subtract(PolyContext& out, PolyContext& op_a, PolyContext& op_b) = 0;
    virtual void multiply(PolyContext& out, PolyContext& op_a, PolyContext& op_b) = 0;
    virtual void
    divide(PolyContext& Quotient_out, PolyContext& Remainder_out, PolyContext& op_a, PolyContext& op_b) = 0;
    virtual void quotient(PolyContext& out, PolyContext& op_a, PolyContext& op_b) = 0;
    virtual void remainder(PolyContext& out, PolyContext& op_a, PolyContext& op_b) = 0;
    virtual void
    divide_by_vanishing_polynomial(PolyContext& out, PolyContext& op_a, uint32_t vanishing_poly_degree) = 0;

    // arithmetic with monomials
    virtual void add_monomial_inplace(PolyContext& poly, C monomial_coeff, uint32_t monomial) = 0;
    virtual void sub_monomial_inplace(PolyContext& poly, C monomial_coeff, uint32_t monomial) = 0;

    // dot product with coefficients
    virtual ECpoint dot_product_with_coefficients(PolyContext& op, ECpoint* points, uint32_t nof_points) = 0;

    virtual int32_t degree(PolyContext& op) = 0;

    virtual I evaluate(PolyContext& op, const D& domain_x) = 0;
    virtual void evaluate(PolyContext& op, const D* domain_x, uint32_t nof_domain_points, I* evaluations /*OUT*/) = 0;

    // TODO Yuval: should backend or context implement get_coefficients()? Is it something any backend/context should
    // implement
    virtual C get_coefficient(PolyContext& op, uint32_t coeff_idx) = 0;
    // if coefficients==nullptr, return nof_coefficients, without writing
    virtual uint32_t get_coefficients(PolyContext& op, C* coefficients) = 0;
  };

} // namespace polynomials

#include "gpu_backend/polynomial_gpu_backend.cuh"
#include "polynomials.cpp" // TODO Yuval: avoid include with explicit instantiation?
#include "polynomials_c_api.h"
