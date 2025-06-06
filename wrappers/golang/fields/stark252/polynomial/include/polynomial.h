#include <stdbool.h>

#ifndef _STARK252_POLY_H
#define _STARK252_POLY_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct scalar_t scalar_t;
typedef struct PolynomialInst PolynomialInst;
typedef struct IntegrityPointer IntegrityPointer;

PolynomialInst* stark252_polynomial_create_from_coefficients(scalar_t* coeffs, size_t size);
PolynomialInst* stark252_polynomial_create_from_rou_evaluations(scalar_t* evals, size_t size);
PolynomialInst* stark252_polynomial_clone(const PolynomialInst* p);
void stark252_polynomial_print(PolynomialInst* p);
void stark252_polynomial_delete(PolynomialInst* instance);
PolynomialInst* stark252_polynomial_add(const PolynomialInst* a, const PolynomialInst* b);
void stark252_polynomial_add_inplace(PolynomialInst* a, const PolynomialInst* b);
PolynomialInst* stark252_polynomial_subtract(const PolynomialInst* a, const PolynomialInst* b);
PolynomialInst* stark252_polynomial_multiply(const PolynomialInst* a, const PolynomialInst* b);
PolynomialInst* stark252_polynomial_multiply_by_scalar(const PolynomialInst* a, const scalar_t* scalar);
void stark252_polynomial_division(const PolynomialInst* a, const PolynomialInst* b, PolynomialInst** q /*OUT*/, PolynomialInst** r /*OUT*/);
PolynomialInst* stark252_polynomial_quotient(const PolynomialInst* a, const PolynomialInst* b);
PolynomialInst* stark252_polynomial_remainder(const PolynomialInst* a, const PolynomialInst* b);
PolynomialInst* stark252_polynomial_divide_by_vanishing(const PolynomialInst* p, size_t vanishing_poly_degree);
void stark252_polynomial_add_monomial_inplace(PolynomialInst* p, const scalar_t* monomial_coeff, size_t monomial);
void stark252_polynomial_sub_monomial_inplace(PolynomialInst* p, const scalar_t* monomial_coeff, size_t monomial);
void stark252_polynomial_evaluate_on_domain(const PolynomialInst* p, scalar_t* domain, size_t domain_size, scalar_t* evals /*OUT*/);
size_t stark252_polynomial_degree(PolynomialInst* p);
size_t stark252_polynomial_copy_coeffs_range(PolynomialInst* p, scalar_t* memory, size_t start_idx, size_t end_idx);
PolynomialInst* stark252_polynomial_even(PolynomialInst* p);
PolynomialInst* stark252_polynomial_odd(PolynomialInst* p);

// scalar_t* stark252_polynomial_get_coeffs_raw_ptr(PolynomialInst* p, size_t* size /*OUT*/, size_t* device_id /*OUT*/);
// PolynomialInst* stark252_polynomial_slice(PolynomialInst* p, size_t offset, size_t stride, size_t size);
// IntegrityPointer* stark252_polynomial_get_coeff_view(PolynomialInst* p, size_t* size /*OUT*/, size_t* device_id /*OUT*/);
// IntegrityPointer* stark252_polynomial_get_rou_evaluations_view(PolynomialInst* p, size_t nof_evals, bool is_reversed, size_t* size /*OUT*/, size_t* device_id /*OUT*/);
// const scalar_t* stark252_polynomial_intergrity_ptr_get(IntegrityPointer* p);
// bool stark252_polynomial_intergrity_ptr_is_valid(IntegrityPointer* p);
// void stark252_polynomial_intergrity_ptr_destroy(IntegrityPointer* p);

#ifdef __cplusplus
}
#endif

#endif

