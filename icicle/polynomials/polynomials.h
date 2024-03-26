#pragma once

#include <iostream>
#include <memory>
#include "utils/integrity_pointer.h"
#include "curves/curve_config.cuh"

#include "polynomial_context.h"
#include "polynomial_backend.h"
#include "polynomial_abstract_factory.h"

namespace polynomials {

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
    Polynomial operator*(const Coeff& c) const; // syntax sugar for polynomial of degree 0 with coefficient c
    template <typename C, typename D, typename I>
    friend Polynomial<C, D, I> operator*(const C& c, const Polynomial<C, D, I>& rhs);

    std::pair<Polynomial, Polynomial> divide(const Polynomial& rhs) const; //  returns (Q(x), R(x))
    Polynomial operator/(const Polynomial& rhs) const; // returns Quotient Q(x) for A(x) = Q(x)B(x) + R(x)
    Polynomial operator%(const Polynomial& rhs) const; // returns Remainder R(x) for A(x) = Q(x)B(x) + R(x)
    Polynomial divide_by_vanishing_polynomial(uint64_t degree) const;

    // arithmetic ops with monomial
    Polynomial& add_monomial_inplace(Coeff monomial_coeff, uint64_t monomial = 0);
    Polynomial& sub_monomial_inplace(Coeff monomial_coeff, uint64_t monomial = 0);

    // Slicing and selecting even or odd components.
    Polynomial slice(uint64_t offset, uint64_t stride, uint64_t size);
    Polynomial even();
    Polynomial odd();

    // Note: Following ops cannot be traced. Calling them invokes polynomial evaluation

    // Evaluation methods
    Image operator()(const Domain& x) const;
    Image evaluate(const Domain& x) const;
    void evaluate_on_domain(Domain* domain, uint64_t size, Image* evals /*OUT*/) const; // caller allocates memory

    // Method to obtain the degree of the polynomial
    int64_t degree();

    // Methods for copying coefficients to host memory.
    Coeff copy_coefficient_to_host(uint64_t idx) const; // single coefficient
    // caller is allocating output memory. If coeff==nullptr, returning nof_coeff only
    int64_t copy_coefficients_to_host(Coeff* host_coeffs = nullptr, int64_t start_idx = 0, int64_t end_idx = -1) const;

    // Methods for obtaining a view of the coefficients or evaluations
    std::tuple<IntegrityPointer<Coeff>, uint64_t /*size*/, uint64_t /*device_id*/> get_coefficients_view();
    std::tuple<IntegrityPointer<Image>, uint64_t /*size*/, uint64_t /*device_id*/>
    get_rou_evaluations_view(uint64_t nof_evaluations = 0, bool is_reversed = false);

    // Overload stream insertion operator for printing.
    friend std::ostream& operator<<(std::ostream& os, Polynomial& poly)
    {
      poly.m_context->print(os);
      return os;
    }

    // Static method to initialize the polynomial class with a factory for context and backend creation.
    static void initialize(std::unique_ptr<AbstractPolynomialFactory<Coeff, Domain, Image>> factory)
    {
      std::atexit(cleanup);
      s_factory = std::move(factory);
    }

    // Cleanup method for releasing factory resources.
    static void cleanup() { s_factory = nullptr; }

  private:
    // The context of the polynomial, encapsulating its state.
    std::shared_ptr<IPolynomialContext<Coeff, Domain, Image>> m_context = nullptr;
    // The computational backend for the polynomial operations.
    std::shared_ptr<IPolynomialBackend<Coeff, Domain, Image>> m_backend = nullptr;

    // Factory for constructing the context and backend instances.
    static inline std::unique_ptr<AbstractPolynomialFactory<Coeff, Domain, Image>> s_factory = nullptr;

  public:
    Polynomial();
    ~Polynomial() = default;

    // Ensures polynomials can be moved but not copied, to manage resources efficiently.
    Polynomial(Polynomial&&) = default;
    Polynomial& operator=(Polynomial&&) = default;
    Polynomial(const Polynomial&) = delete;
    Polynomial& operator=(const Polynomial&) = delete;
  };

  // explicit instantiation
  using curve_config::scalar_t;

  // Friend operator to allow multiplication with a scalar from the left-hand side
  template <typename C = scalar_t, typename D = C, typename I = C>
  Polynomial<C, D, I> operator*(const C& c, const Polynomial<C, D, I>& rhs);

  // External template instantiation to ensure the template is compiled for specific types.
  extern template class Polynomial<scalar_t>;

} // namespace polynomials
