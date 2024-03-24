#pragma once

#include "polynomials.h"
#include "curves/curve_config.cuh"
#include "utils/utils.h"
#include "utils/integrity_pointer.h"

namespace polynomials {
  extern "C" {

  // Defines a polynomial instance based on the scalar type from the curve configuration.
  typedef Polynomial<curve_config::scalar_t> PolynomialInst;

  // Constructs a polynomial from a set of coefficients.
  // coeffs: Array of coefficients.
  // size: Number of coefficients in the array.
  // Returns a pointer to the newly created polynomial instance.
  PolynomialInst* CONCAT_EXPAND(CURVE, polynomial_create_from_coefficients)(curve_config::scalar_t* coeffs, size_t size)
  {
    auto result = new PolynomialInst(PolynomialInst::from_coefficients(coeffs, size));
    return result;
  }

  // Constructs a polynomial from evaluations at the roots of unity.
  // evals: Array of evaluations.
  // size: Number of evaluations in the array.
  // Returns a pointer to the newly created polynomial instance.
  PolynomialInst*
  CONCAT_EXPAND(CURVE, polynomial_create_from_rou_evaluations)(curve_config::scalar_t* evals, size_t size)
  {
    auto result = new PolynomialInst(PolynomialInst::from_rou_evaluations(evals, size));
    return result;
  }

  // Clones an existing polynomial instance.
  // p: Pointer to the polynomial instance to clone.
  // Returns a pointer to the cloned polynomial instance.
  PolynomialInst* CONCAT_EXPAND(CURVE, polynomial_clone)(const PolynomialInst* p)
  {
    auto result = new PolynomialInst(p->clone());
    return result;
  }

  // Deletes a polynomial instance, freeing its memory.
  // instance: Pointer to the polynomial instance to delete.
  void CONCAT_EXPAND(CURVE, polynomial_delete)(PolynomialInst* instance) { delete instance; }

  // Adds two polynomials.
  // a, b: Pointers to the polynomial instances to add.
  // Returns a pointer to the resulting polynomial instance.
  PolynomialInst* CONCAT_EXPAND(CURVE, polynomial_add)(const PolynomialInst* a, const PolynomialInst* b)
  {
    auto result = new PolynomialInst(std::move(*a + *b));
    return result;
  }

  // Adds a polynomial to another in place.
  // a: Pointer to the polynomial to add to.
  // b: Pointer to the polynomial to add.
  void CONCAT_EXPAND(CURVE, polynomial_add_inplace)(PolynomialInst* a, const PolynomialInst* b) { *a += *b; }

  // Subtracts one polynomial from another.
  // a, b: Pointers to the polynomial instances (minuend and subtrahend, respectively).
  // Returns a pointer to the resulting polynomial instance.

  PolynomialInst* CONCAT_EXPAND(CURVE, polynomial_subtract)(const PolynomialInst* a, const PolynomialInst* b)
  {
    auto result = new PolynomialInst(std::move(*a - *b));
    return result;
  }

  // Multiplies two polynomials.
  // a, b: Pointers to the polynomial instances to multiply.
  // Returns a pointer to the resulting polynomial instance.
  PolynomialInst* CONCAT_EXPAND(CURVE, polynomial_multiply)(const PolynomialInst* a, const PolynomialInst* b)
  {
    auto result = new PolynomialInst(std::move(*a * *b));
    return result;
  }

  // Multiplies a polynomial by a scalar coefficient.
  // a: Pointer to the polynomial instance.
  // coeff: Scalar coefficient to multiply by.
  // Returns a pointer to the resulting polynomial instance.
  PolynomialInst* CONCAT_EXPAND(CURVE, polynomial_multiply_by_coeff)(const PolynomialInst* a, const scalar_t* coeff)
  {
    auto result = new PolynomialInst(std::move(*a * *coeff));
    return result;
  }

  // Divides one polynomial by another, returning both quotient and remainder.
  // a, b: Pointers to the polynomial instances (dividend and divisor, respectively).
  // q: Output parameter for the quotient.
  // r: Output parameter for the remainder.
  void CONCAT_EXPAND(CURVE, polynomial_division)(
    const PolynomialInst* a, const PolynomialInst* b, PolynomialInst** q /*OUT*/, PolynomialInst** r /*OUT*/)
  {
    auto [_q, _r] = a->divide(*b);
    *q = new PolynomialInst(std::move(_q));
    *r = new PolynomialInst(std::move(_r));
  }

  // Calculates the quotient of dividing one polynomial by another.
  // a, b: Pointers to the polynomial instances (dividend and divisor, respectively).
  // Returns a pointer to the resulting quotient polynomial instance.
  PolynomialInst* CONCAT_EXPAND(CURVE, polynomial_quotient)(const PolynomialInst* a, const PolynomialInst* b)
  {
    auto result = new PolynomialInst(std::move(*a / *b));
    return result;
  }

  // Calculates the remainder of dividing one polynomial by another.
  // a, b: Pointers to the polynomial instances (dividend and divisor, respectively).
  // Returns a pointer to the resulting remainder polynomial instance.

  PolynomialInst* CONCAT_EXPAND(CURVE, polynomial_remainder)(const PolynomialInst* a, const PolynomialInst* b)
  {
    auto result = new PolynomialInst(std::move(*a % *b));
    return result;
  }

  // Divides a polynomial by a vanishing polynomial of a given degree, over rou domain.
  // p: Pointer to the polynomial instance.
  // vanishing_poly_degree: Degree of the vanishing polynomial.
  // Returns a pointer to the resulting polynomial instance.
  PolynomialInst*
  CONCAT_EXPAND(CURVE, polynomial_divide_by_vanishing)(const PolynomialInst* p, uint64_t vanishing_poly_degree)
  {
    auto result = new PolynomialInst(std::move(p->divide_by_vanishing_polynomial(vanishing_poly_degree)));
    return result;
  }

  // Adds a monomial to a polynomial in place.
  // p: Pointer to the polynomial instance.
  // monomial_coeff: Coefficient of the monomial to add.
  // monomial: Degree of the monomial to add.
  void CONCAT_EXPAND(CURVE, polynomial_add_monomial_inplace)(
    PolynomialInst* p, const scalar_t* monomial_coeff, uint64_t monomial)
  {
    p->add_monomial_inplace(*monomial_coeff, monomial);
  }

  // Subtracts a monomial from a polynomial in place.
  // p: Pointer to the polynomial instance.
  // monomial_coeff: Coefficient of the monomial to subtract.
  // monomial: Degree of the monomial to subtract.
  void CONCAT_EXPAND(CURVE, polynomial_sub_monomial_inplace)(
    PolynomialInst* p, const scalar_t* monomial_coeff, uint64_t monomial)
  {
    p->sub_monomial_inplace(*monomial_coeff, monomial);
  }

  // Evaluates a polynomial at a given point.
  // p: Pointer to the polynomial instance.
  // x: Point at which to evaluate the polynomial.
  // Returns the evaluation result.
  scalar_t CONCAT_EXPAND(CURVE, polynomial_evaluate)(const PolynomialInst* p, const scalar_t& x)
  {
    return p->evaluate(x);
  }

  // Evaluates a polynomial on a domain of points.
  // p: Pointer to the polynomial instance.
  // domain: Array of points constituting the domain.
  // domain_size: Number of points in the domain.
  // evals: Output array for the evaluations.
  void CONCAT_EXPAND(CURVE, polynomial_evaluate_on_domain)(
    const PolynomialInst* p, scalar_t* domain, uint64_t domain_size, scalar_t* evals /*OUT*/)
  {
    return p->evaluate_on_domain(domain, domain_size, evals);
  }

  // Returns the degree of a polynomial.
  // p: Pointer to the polynomial instance.
  // Returns the degree of the polynomial.
  int64_t CONCAT_EXPAND(CURVE, polynomial_degree)(PolynomialInst* p) { return p->degree(); }

  // Copies a single coefficient of a polynomial to host memory.
  // p: Pointer to the polynomial instance.
  // idx: Index of the coefficient to copy.
  // Returns the coefficient value.
  scalar_t CONCAT_EXPAND(CURVE, polynomial_copy_single_coeff_to_host)(PolynomialInst* p, uint64_t idx)
  {
    return p->copy_coefficient_to_host(idx);
  }

  // Copies a range of polynomial coefficients to host memory.
  // p: Pointer to the polynomial instance.
  // host_memory: Array to copy the coefficients into. If NULL, not copying.
  // start_idx: Start index of the range to copy.
  // end_idx: End index of the range to copy.
  // Returns the number of coefficients copied. if host_memory is NULL, returns number of coefficients.
  int64_t CONCAT_EXPAND(CURVE, polynomial_coeffs_to_host)(
    PolynomialInst* p, scalar_t* host_memory, uint64_t start_idx, uint64_t end_idx)
  {
    return p->copy_coefficients_to_host(host_memory, start_idx, end_idx);
  }

  // Retrieves a device-memory view of the polynomial coefficients.
  // p: Pointer to the polynomial instance.
  // size: Output parameter for the size of the view.
  // device_id: Output parameter for the device ID.
  // Returns a pointer to an integrity pointer encapsulating the coefficients view.
  IntegrityPointer<scalar_t>* CONCAT_EXPAND(CURVE, polynomial_get_coeff_view)(
    PolynomialInst* p, uint64_t* size /*OUT*/, uint64_t* device_id /*OUT*/)
  {
    auto [coeffs, _size, _device_id] = p->get_coefficients_view();
    *size = _size;
    *device_id = _device_id;
    return new IntegrityPointer<scalar_t>(std::move(coeffs));
  }

  // Retrieves a device-memory view of the polynomial's evaluations on the roots of unity.
  // p: Pointer to the polynomial instance.
  // nof_evals: Number of evaluations.
  // is_reversed: Whether the evaluations are in reversed order.
  // size: Output parameter for the size of the view.
  // device_id: Output parameter for the device ID.
  // Returns a pointer to an integrity pointer encapsulating the evaluations view.
  IntegrityPointer<scalar_t>* CONCAT_EXPAND(CURVE, polynomial_get_rou_evaluations_view)(
    PolynomialInst* p, uint64_t nof_evals, bool is_reversed, uint64_t* size /*OUT*/, uint64_t* device_id /*OUT*/)
  {
    auto [rou_evals, _size, _device_id] = p->get_rou_evaluations_view(nof_evals, is_reversed);
    *size = _size;
    *device_id = _device_id;
    return new IntegrityPointer<scalar_t>(std::move(rou_evals));
  }

  // Reads the pointer from an integrity pointer.
  // p: Pointer to the integrity pointer.
  // Returns the raw pointer if still valid, otherwise NULL.
  const scalar_t* CONCAT_EXPAND(CURVE, polynomial_intergrity_ptr_get)(IntegrityPointer<scalar_t>* p)
  {
    return p->get();
  }

  // Checks if an integrity pointer is still valid.
  // p: Pointer to the integrity pointer.
  // Returns true if the pointer is valid, false otherwise.
  bool CONCAT_EXPAND(CURVE, polynomial_intergrity_ptr_is_valid)(IntegrityPointer<scalar_t>* p) { return p->isValid(); }

  // Destroys an integrity pointer, freeing its resources.
  // p: Pointer to the integrity pointer to destroy.
  void CONCAT_EXPAND(CURVE, polynomial_intergrity_ptr_destroy)(IntegrityPointer<scalar_t>* p) { delete p; }

  } // extern "C"

} // namespace polynomials
