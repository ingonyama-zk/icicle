#pragma once

#include <cstdint> // for uint64_t, int64_t

namespace polynomials {

  /**
   * @brief Interface for the polynomial computational backend.
   *
   * The `IPolynomialBackend` interface defines the set of operations for polynomial arithmetic
   * and manipulation that can be performed on a given computational device or platform (e.g., GPU, ZPU).
   * This interface abstracts the computational logic, allowing for implementation-specific optimizations
   * and hardware utilization. It interacts closely with `IPolynomialContext` to manage polynomial data
   * states and perform computations.
   *
   * @tparam C Type of the coefficients.
   * @tparam D Domain type, representing the input space of the polynomial.
   * @tparam I Image type, representing the output space of the polynomial.
   */
  template <typename C, typename D, typename I>
  class IPolynomialBackend
  {
  public:
    IPolynomialBackend() = default;
    virtual ~IPolynomialBackend() = default;

    typedef IPolynomialContext<C, D, I> PolyContext;

    // Initialization methods
    virtual void from_coefficients(PolyContext& p, uint64_t nof_coefficients, const C* coefficients = nullptr) = 0;
    virtual void from_rou_evaluations(PolyContext& p, uint64_t nof_evaluations, const I* evaluations = nullptr) = 0;
    virtual void clone(PolyContext& out, PolyContext& in) = 0;

    // Arithmetic operations
    virtual void add(PolyContext& out, PolyContext& op_a, PolyContext& op_b) = 0;
    virtual void subtract(PolyContext& out, PolyContext& op_a, PolyContext& op_b) = 0;
    virtual void multiply(PolyContext& out, PolyContext& op_a, PolyContext& op_b) = 0;
    virtual void
    divide(PolyContext& Quotient_out, PolyContext& Remainder_out, PolyContext& op_a, PolyContext& op_b) = 0;
    virtual void quotient(PolyContext& out, PolyContext& op_a, PolyContext& op_b) = 0;
    virtual void remainder(PolyContext& out, PolyContext& op_a, PolyContext& op_b) = 0;
    virtual void
    divide_by_vanishing_polynomial(PolyContext& out, PolyContext& op_a, uint64_t vanishing_poly_degree) = 0;

    // Operations specific to monomials
    virtual void add_monomial_inplace(PolyContext& poly, C monomial_coeff, uint64_t monomial) = 0;
    virtual void sub_monomial_inplace(PolyContext& poly, C monomial_coeff, uint64_t monomial) = 0;

    // Utility methods
    virtual void slice(PolyContext& out, PolyContext& in, uint64_t offset, uint64_t stride, uint64_t size) = 0;
    virtual int64_t degree(PolyContext& op) = 0;

    // Method to access mutable storage within the context
    void* get_context_storage_mutable(PolyContext& ctxt) { return ctxt.get_storage_mutable(); }

    // Evaluation methods
    virtual I evaluate(PolyContext& op, const D& domain_x) = 0;
    virtual void evaluate_on_domain(PolyContext& op, const D* domain, uint64_t size, I* evaluations /*OUT*/) = 0;

    // Methods to copy coefficients to host memory
    virtual C copy_coefficient_to_host(PolyContext& op, uint64_t coeff_idx) = 0;
    virtual int64_t
    copy_coefficients_to_host(PolyContext& op, C* host_coeffs, int64_t start_idx = 0, int64_t end_idx = -1) = 0;

    // Methods to get views of coefficients and evaluations, including device id
    virtual std::tuple<IntegrityPointer<C>, uint64_t /*size*/, uint64_t /*device_id*/>
    get_coefficients_view(PolyContext& p) = 0;
    virtual std::tuple<IntegrityPointer<I>, uint64_t /*size*/, uint64_t /*device_id*/>
    get_rou_evaluations_view(PolyContext& p, uint64_t nof_evaluations = 0, bool is_reversed = false) = 0;
  };

} // namespace polynomials
