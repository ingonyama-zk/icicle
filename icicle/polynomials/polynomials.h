#pragma once

#include <iostream>
#include <memory>
#include "utils/integrity_pointer.h"
#include "curves/curve_config.cuh"

namespace polynomials {
  template <typename Coeff, typename Domain, typename Image>
  class IPolynomialBackend;

  template <typename Coeff, typename Domain, typename Image>
  class IPolynomialContext;

  template <typename C, typename D, typename I>
  class AbstractPolynomialFactory;

  using curve_config::scalar_t;

  /*============================== Polynomial API ==============================*/
  template <typename Coeff = scalar_t, typename Domain = Coeff, typename Image = Coeff>
  class Polynomial
  {
  public:
    // initialization (coefficients/evaluations can reside on host or device)
    static Polynomial from_coefficients(const Coeff* coefficients, uint64_t nof_coefficients);
    static Polynomial from_rou_evaluations(const Image* evaluations, uint64_t nof_evaluations);
    Polynomial clone() const;

    // arithmetic ops (two polynomials)
    Polynomial operator+(const Polynomial& rhs) const;
    Polynomial& operator+=(const Polynomial& rhs);

    Polynomial operator-(const Polynomial& rhs) const;

    Polynomial operator*(const Polynomial& rhs) const;
    Polynomial operator*(const Coeff& c) const; // syntax sugar for polynomial of degree 1 with coefficient c
    template <typename C, typename D, typename I>
    friend Polynomial<C, D, I> operator*(const C& c, const Polynomial<C, D, I>& rhs);

    std::pair<Polynomial, Polynomial> divide(const Polynomial& rhs) const; //  returns (Q(x), R(x))
    Polynomial operator/(const Polynomial& rhs) const; // returns Quotient Q(x) for A(x) = Q(x)B(x) + R(x)
    Polynomial operator%(const Polynomial& rhs) const; // returns Remainder R(x) for A(x) = Q(x)B(x) + R(x)
    Polynomial divide_by_vanishing_polynomial(uint64_t degree) const;

    // arithmetic ops with monomial
    Polynomial& add_monomial_inplace(Coeff monomial_coeff, uint64_t monomial = 0);
    Polynomial& sub_monomial_inplace(Coeff monomial_coeff, uint64_t monomial = 0);

    Polynomial slice(uint64_t offset, uint64_t stride, uint64_t size);
    Polynomial even();
    Polynomial odd();
    // Following ops cannot be traced. Calling them requires to compute

    // evaluation
    Image operator()(const Domain& x) const;
    Image evaluate(const Domain& x) const;
    void evaluate_on_domain(Domain* domain, uint64_t size, Image* evals /*OUT*/) const; // caller allocates memory

    // highest non-zero coefficient degree
    int64_t degree();

    // Copy coefficients to host memory
    Coeff copy_coefficient_to_host(uint64_t idx) const; // single coefficient
    // caller is allocating output memory. If coeff==nullptr, returning nof_coeff only
    int64_t copy_coefficients_to_host(Coeff* host_coeffs = nullptr, int64_t start_idx = 0, int64_t end_idx = -1) const;

    // Returns a self-invalidating view of coefficients/rou-evaluations. Once the polynomial is modified, the view
    // identifies and invalidates itself.
    std::tuple<IntegrityPointer<Coeff>, uint64_t /*size*/, uint64_t /*device_id*/> get_coefficients_view();
    std::tuple<IntegrityPointer<Image>, uint64_t /*size*/, uint64_t /*device_id*/>
    get_rou_evaluations_view(uint64_t nof_evaluations = 0, bool is_reversed = false);

    friend std::ostream& operator<<(std::ostream& os, Polynomial& poly)
    {
      poly.m_context->print(os);
      return os;
    }

    static void initialize(std::unique_ptr<AbstractPolynomialFactory<Coeff, Domain, Image>> factory)
    {
      s_factory = std::move(factory);
    }

  private:
    // context is a wrapper for the polynomial state, including allocated memory, device context etc.
    std::shared_ptr<IPolynomialContext<Coeff, Domain, Image>> m_context = nullptr;
    // backend is the actual API implementation
    std::shared_ptr<IPolynomialBackend<Coeff, Domain, Image>> m_backend = nullptr;

    // factory is building the context and backend for polynomial objects
    static inline std::unique_ptr<AbstractPolynomialFactory<Coeff, Domain, Image>> s_factory = nullptr;

  public:
    Polynomial();
    ~Polynomial() = default;
    // make sure polynomials can be moved but not copied
    Polynomial(Polynomial&&) = default;
    Polynomial& operator=(Polynomial&&) = default;
    Polynomial(const Polynomial&) = delete;
    Polynomial& operator=(const Polynomial&) = delete;
  };

  // allows multiplication c*Poly in addition to Poly*c
  template <typename C = scalar_t, typename D = C, typename I = C>
  Polynomial<C, D, I> operator*(const C& c, const Polynomial<C, D, I>& rhs);
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

    // coefficients/evaluations can reside on host or device
    virtual C* init_from_coefficients(uint64_t nof_coefficients, const C* coefficients = nullptr) = 0;
    virtual I* init_from_rou_evaluations(uint64_t nof_evaluations, const I* evaluations = nullptr) = 0;
    virtual std::shared_ptr<IPolynomialContext> clone() const = 0;

    virtual void allocate(uint64_t nof_elements, State init_state = State::Coefficients, bool memset_zeros = true) = 0;
    virtual void release() = 0;

    virtual void transform_to_coefficients(uint64_t nof_coefficients = 0) = 0;
    virtual void transform_to_evaluations(uint64_t nof_evaluations = 0, bool is_reversed = false) = 0;

    State get_state() const { return m_state; }
    uint64_t get_nof_elements() const { return m_nof_elements; }

    virtual std::pair<C*, uint64_t> get_coefficients() = 0;
    virtual std::pair<I*, uint64_t> get_rou_evaluations() = 0;

    virtual std::tuple<IntegrityPointer<C>, uint64_t /*size*/, uint64_t /*device_id*/> get_coefficients_view() = 0;
    virtual std::tuple<IntegrityPointer<I>, uint64_t /*size*/, uint64_t /*device_id*/>
    get_rou_evaluations_view(uint64_t nof_evaluations = 0, bool is_reversed = false) = 0;

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

    virtual void slice(PolyContext& out, PolyContext& in, uint64_t offset, uint64_t stride, uint64_t size) = 0;

    virtual int64_t degree(PolyContext& op) = 0;

    virtual I evaluate(PolyContext& op, const D& domain_x) = 0;
    virtual void evaluate_on_domain(PolyContext& op, const D* domain, uint64_t size, I* evaluations /*OUT*/) = 0;

    virtual C copy_coefficient_to_host(PolyContext& op, uint64_t coeff_idx) = 0;
    // if coefficients==nullptr, return nof_coefficients, without writing
    virtual int64_t
    copy_coefficients_to_host(PolyContext& op, C* host_coeffs, int64_t start_idx = 0, int64_t end_idx = -1) = 0;
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

  extern template class Polynomial<>;

} // namespace polynomials
