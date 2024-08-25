#pragma once

#include <iostream>
#include <memory>
#include "icicle/utils/integrity_pointer.h"

#include "polynomial_context.h"
#include "polynomial_backend.h"
#include "polynomial_abstract_factory.h"

namespace icicle {

  /**
   * @brief Represents a polynomial and provides operations for polynomial arithmetic, evaluation, and manipulation.
   *
   * This class models a polynomial with coefficients of type `Coeff`, defined over a domain `Domain` and producing
   * outputs of type `Image`. It supports a range of operations including basic arithmetic (addition, subtraction,
   * multiplication, division), evaluation at points or over domains, and manipulation (slicing, adding monomials).
   * The implementation abstracts over the specifics of computation and storage through the use of an abstract factory,
   * contexts, and backends, allowing for efficient execution across various computational environments.
   *
   * @tparam Coeff Type of the coefficients of the polynomial.
   * @tparam Domain Type representing the input space of the polynomial (defaults to `Coeff`).
   * @tparam Image Type representing the output space of the polynomial (defaults to `Coeff`).
   */
  template <typename Coeff, typename Domain = Coeff, typename Image = Coeff>
  class Polynomial
  {
  public:
    // Initialization (coefficients/evaluations can reside on host or device)
    static Polynomial from_coefficients(const Coeff* coefficients, uint64_t nof_coefficients);
    static Polynomial from_rou_evaluations(const Image* evaluations, uint64_t nof_evaluations);

    // Clone the polynomial
    Polynomial clone() const;

    // Arithmetic ops
    Polynomial operator+(const Polynomial& rhs) const;
    Polynomial& operator+=(const Polynomial& rhs);

    Polynomial operator-(const Polynomial& rhs) const;

    Polynomial operator*(const Polynomial& rhs) const;
    Polynomial operator*(const Domain& scalar) const; // scalar multiplication
    template <typename C, typename D, typename I>
    friend Polynomial<C, D, I> operator*(const D& scalar, const Polynomial<C, D, I>& rhs);

    std::pair<Polynomial, Polynomial> divide(const Polynomial& rhs) const; //  returns (Q(x), R(x))
    Polynomial operator/(const Polynomial& rhs) const; // returns Quotient Q(x) for A(x) = Q(x)B(x) + R(x)
    Polynomial operator%(const Polynomial& rhs) const; // returns Remainder R(x) for A(x) = Q(x)B(x) + R(x)
    Polynomial divide_by_vanishing_polynomial(uint64_t degree) const;

    // arithmetic ops with monomial
    Polynomial& add_monomial_inplace(Coeff monomial_coeff, uint64_t monomial = 0);
    Polynomial& sub_monomial_inplace(Coeff monomial_coeff, uint64_t monomial = 0);

    // Slicing and selecting even or odd components.
    Polynomial slice(uint64_t offset, uint64_t stride, uint64_t size = 0 /*0 means take all elements*/);
    Polynomial even();
    Polynomial odd();

    // Note: Following ops cannot be traced. Calling them invokes polynomial evaluation

    // Evaluation methods
    Image operator()(const Domain& x) const;
    void evaluate(const Domain* x, Image* eval /*OUT*/) const;
    void evaluate_on_domain(Domain* domain, uint64_t size, Image* evals /*OUT*/) const; // caller allocates memory
    void evaluate_on_rou_domain(uint64_t domain_log_size, Image* evals /*OUT*/) const;  // caller allocate memory

    // Method to obtain the degree of the polynomial
    int64_t degree();

    // Methods for copying coefficients to host memory.
    Coeff get_coeff(uint64_t idx) const; // single coefficient
    // caller is allocating output memory. If coeff==nullptr, returning nof_coeff only
    uint64_t copy_coeffs(Coeff* host_coeffs, uint64_t start_idx, uint64_t end_idx) const;

    // Methods for obtaining a view of the coefficients
    std::tuple<IntegrityPointer<Coeff>, uint64_t /*size*/> get_coefficients_view();

    // Overload stream insertion operator for printing.
    friend std::ostream& operator<<(std::ostream& os, Polynomial& poly)
    {
      poly.m_context->print(os);
      return os;
    }

  private:
    // The context of the polynomial, encapsulating its state.
    std::shared_ptr<IPolynomialContext<Coeff, Domain, Image>> m_context = nullptr;
    // The computational backend for the polynomial operations.
    std::shared_ptr<IPolynomialBackend<Coeff, Domain, Image>> m_backend = nullptr;

  public:
    Polynomial();
    ~Polynomial() = default;

    // Ensures polynomials can be moved but not copied, to manage resources efficiently.
    Polynomial(Polynomial&&) = default;
    Polynomial& operator=(Polynomial&&) = default;
    Polynomial(const Polynomial&) = delete;
    Polynomial& operator=(const Polynomial&) = delete;

    std::shared_ptr<IPolynomialContext<Coeff, Domain, Image>> get_context() { return m_context; }
  };

} // namespace icicle
