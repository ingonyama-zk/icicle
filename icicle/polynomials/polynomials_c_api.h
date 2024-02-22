#pragma once

#include "polynomials.h"
#include "curves/curve_config.cuh"

namespace polynomials {
  extern "C" {
  typedef Polynomial<curve_config::scalar_t> PolynomialInst;

  // Create a new Polynomial instance with given coefficients
  PolynomialInst* polynomial_create_from_coefficients(curve_config::scalar_t* coeffs, size_t size)
  {
    auto result = new PolynomialInst(PolynomialInst::from_coefficients(coeffs, size));
    return result;
  }

  // Add two Polynomial instances
  PolynomialInst* polynomial_add(const PolynomialInst* a, const PolynomialInst* b)
  {
    auto result = new PolynomialInst(std::move(*a + *b));
    return result;
  }

  // Delete a Polynomial instance
  void polynomial_delete(PolynomialInst* instance) { delete instance; }
  }

} // namespace polynomials
