#pragma once

#include <iostream>
#include <memory>

namespace polynomials {
  template <typename CoeffType, typename DomainType, typename ImageType>
  class IPolynomialBackend;

  template <typename CoeffType, typename DomainType, typename ImageType>
  class IPolynomialContext;

  template <typename C, typename D, typename I>
  class AbstractPolynomialFactory;

  /*============================== Polynomial API ==============================*/
  template <
    typename CoeffType = curve_config::scalar_t,
    typename DomainType = CoeffType,
    typename ImageType = CoeffType>
  class Polynomial
  {
  public:
    // initialization
    static Polynomial from_coefficients(const CoeffType* coefficients, uint64_t nof_coefficients);
    static Polynomial from_rou_evaluations(const ImageType* evaluations, uint64_t nof_evaluations);

    // arithmetic ops (two polynomials)
    Polynomial operator+(const Polynomial& rhs) const;
    Polynomial operator-(const Polynomial& rhs) const;
    Polynomial operator*(const Polynomial& rhs) const;
    Polynomial operator/(const Polynomial& rhs) const; // returns Quotient Q(x) for A(x) = Q(x)B(x) + R(x)
    Polynomial operator%(const Polynomial& rhs) const; // returns Remainder R(x) for A(x) = Q(x)B(x) + R(x)
    std::pair<Polynomial, Polynomial> divide(const Polynomial& rhs) const; //  returns (Q(x), R(x))
    Polynomial divide_by_vanishing_polynomial(uint64_t vanishing_polynomial_degree) const;

    // arithmetic ops with monomial
    Polynomial& add_monomial_inplace(CoeffType monomial_coeff, uint64_t monomial = 0);
    Polynomial& sub_monomial_inplace(CoeffType monomial_coeff, uint64_t monomial = 0);

    // evaluation
    ImageType operator()(const DomainType& x) const;
    ImageType evaluate(const DomainType& x) const;
    void evaluate(DomainType* x, uint64_t nof_points, ImageType* evals /*OUT*/) const; // caller allocates memory

    // highest non-zero coefficient degree
    int32_t degree();

    // Read coefficients. TODO Yuval: flag for Host/Device? For Device can return const reference to coeffs but cannot
    // guarantee that polynomial remains in coeffs and even that it is not released. Copy?
    CoeffType get_coefficient_on_host(uint64_t idx) const;
    // caller is allocating output memory. If coeff==nullptr, returning nof_coeff only
    int64_t
    get_coefficients_on_host(CoeffType* host_coeffs = nullptr, int64_t start_idx = 0, int64_t end_idx = -1) const;

    friend std::ostream& operator<<(std::ostream& os, Polynomial& poly)
    {
      poly.m_context->print(os);
      return os;
    }

    static void initialize(std::unique_ptr<AbstractPolynomialFactory<CoeffType, DomainType, ImageType>> factory)
    {
      s_factory = std::move(factory);
    }

  private:
    // context is a wrapper for the polynomial state, including allocated memory, device context etc.
    std::shared_ptr<IPolynomialContext<CoeffType, DomainType, ImageType>> m_context = nullptr;
    // backend is the actual API implementation
    std::shared_ptr<IPolynomialBackend<CoeffType, DomainType, ImageType>> m_backend = nullptr;

    // factory is building the context and backend for polynomial objects
    static inline std::unique_ptr<AbstractPolynomialFactory<CoeffType, DomainType, ImageType>> s_factory = nullptr;

    Polynomial();

  public:
    ~Polynomial() = default;
    // make sure polynomials can be moved but not copied
    Polynomial(Polynomial&&) = default;
    Polynomial& operator=(Polynomial&&) = default;
    Polynomial(const Polynomial&) = delete;
    Polynomial& operator=(const Polynomial&) = delete;
  };
  /*============================== Polynomial API END==============================*/

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

    virtual C* init_from_coefficients(uint64_t nof_coefficients, const C* host_coefficients = nullptr) = 0;
    virtual I* init_from_rou_evaluations(uint64_t nof_evaluations, const I* host_evaluations = nullptr) = 0;

    virtual std::pair<C*, uint64_t> get_coefficients() = 0;
    virtual std::pair<I*, uint64_t> get_rou_evaluations() = 0;

    virtual void transform_to_coefficients(uint64_t nof_coefficients = 0) = 0;
    virtual void transform_to_evaluations(uint64_t nof_evaluations = 0, bool is_reversed = 0) = 0;

    virtual void allocate(uint64_t nof_elements, State init_state = State::Coefficients, bool memset_zeros = true) = 0;
    virtual void release() = 0;

    State get_state() const { return m_state; }
    uint64_t get_nof_elements() const { return m_nof_elements; }

    virtual void print(std::ostream& os) = 0;

  protected:
    void set_state(State state) { m_state = state; }

    State m_state;
    uint64_t m_nof_elements = 0;

    // id for debug
    static inline uint64_t s_id = 0;
    const uint64_t m_id;
  };

  /*============================== Polynomial Backend ==============================*/
  template <typename C, typename D, typename I>
  class IPolynomialBackend
  {
  public:
    IPolynomialBackend() = default;
    virtual ~IPolynomialBackend() {}

    typedef IPolynomialContext<C, D, I> PolyContext;

    // arithmetic
    virtual void add(PolyContext& out, PolyContext& op_a, PolyContext& op_b) = 0;
    virtual void subtract(PolyContext& out, PolyContext& op_a, PolyContext& op_b) = 0;
    virtual void multiply(PolyContext& out, PolyContext& op_a, PolyContext& op_b) = 0;
    virtual void
    divide(PolyContext& Quotient_out, PolyContext& Remainder_out, PolyContext& op_a, PolyContext& op_b) = 0;
    virtual void quotient(PolyContext& out, PolyContext& op_a, PolyContext& op_b) = 0;
    virtual void remainder(PolyContext& out, PolyContext& op_a, PolyContext& op_b) = 0;
    virtual void
    divide_by_vanishing_polynomial(PolyContext& out, PolyContext& op_a, uint64_t vanishing_poly_degree) = 0;

    // arithmetic with monomials
    virtual void add_monomial_inplace(PolyContext& poly, C monomial_coeff, uint64_t monomial) = 0;
    virtual void sub_monomial_inplace(PolyContext& poly, C monomial_coeff, uint64_t monomial) = 0;

    virtual int32_t degree(PolyContext& op) = 0;

    virtual I evaluate(PolyContext& op, const D& domain_x) = 0;
    virtual void evaluate(PolyContext& op, const D* domain_x, uint64_t nof_domain_points, I* evaluations /*OUT*/) = 0;

    virtual C get_coefficient_on_host(PolyContext& op, uint64_t coeff_idx) = 0;
    // if coefficients==nullptr, return nof_coefficients, without writing
    virtual int64_t
    get_coefficients_on_host(PolyContext& op, C* host_coeffs, int64_t start_idx = 0, int64_t end_idx = -1) = 0;
  };

  /*============================== Polynomial Absract Factory ==============================*/
  template <typename C, typename D, typename I>
  class AbstractPolynomialFactory
  {
  public:
    virtual std::shared_ptr<IPolynomialContext<C, D, I>> create_context() = 0;
    virtual std::shared_ptr<IPolynomialBackend<C, D, I>> create_backend() = 0;
    virtual ~AbstractPolynomialFactory() = default;
  };

} // namespace polynomials

#include "cuda_backend/polynomial_cuda_backend.cuh"
#include "polynomials.cpp" // TODO Yuval: avoid include with explicit instantiation?
#include "polynomials_c_api.h"
