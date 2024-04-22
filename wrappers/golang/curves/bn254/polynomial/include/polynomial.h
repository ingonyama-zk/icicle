#include <cuda_runtime.h>
#include <stdbool.h>

#ifndef _BN254_POLY_H
#define _BN254_POLY_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct scalar_t scalar_t;
typedef struct PolynomialInst PolynomialInst;
typedef struct IntegrityPointer IntegrityPointer;

bool bn254polynomial_init_cuda_backend();
PolynomialInst* bn254polynomial_create_from_coefficients(scalar_t* coeffs, size_t size);
PolynomialInst* bn254polynomial_create_from_rou_evaluations(scalar_t* evals, size_t size);
PolynomialInst* bn254polynomial_clone(const PolynomialInst* p);
void bn254polynomial_print(PolynomialInst* p);
void bn254polynomial_delete(PolynomialInst* instance);
PolynomialInst* bn254polynomial_add(const PolynomialInst* a, const PolynomialInst* b);
void bn254polynomial_add_inplace(PolynomialInst* a, const PolynomialInst* b);
PolynomialInst* bn254polynomial_subtract(const PolynomialInst* a, const PolynomialInst* b);
PolynomialInst* bn254polynomial_multiply(const PolynomialInst* a, const PolynomialInst* b);
PolynomialInst* bn254polynomial_multiply_by_scalar(const PolynomialInst* a, const scalar_t* scalar);
void bn254polynomial_division(const PolynomialInst* a, const PolynomialInst* b, PolynomialInst** q /*OUT*/, PolynomialInst** r /*OUT*/);
PolynomialInst* bn254polynomial_quotient(const PolynomialInst* a, const PolynomialInst* b);
PolynomialInst* bn254polynomial_remainder(const PolynomialInst* a, const PolynomialInst* b);
PolynomialInst* bn254polynomial_divide_by_vanishing(const PolynomialInst* p, size_t vanishing_poly_degree);
void bn254polynomial_add_monomial_inplace(PolynomialInst* p, const scalar_t* monomial_coeff, size_t monomial);
void bn254polynomial_sub_monomial_inplace(PolynomialInst* p, const scalar_t* monomial_coeff, size_t monomial);
void bn254polynomial_evaluate_on_domain(const PolynomialInst* p, scalar_t* domain, size_t domain_size, scalar_t* evals /*OUT*/);
size_t bn254polynomial_degree(PolynomialInst* p);
size_t bn254polynomial_copy_coeffs_range(PolynomialInst* p, scalar_t* memory, size_t start_idx, size_t end_idx);
PolynomialInst* bn254polynomial_even(PolynomialInst* p);
PolynomialInst* bn254polynomial_odd(PolynomialInst* p);

// scalar_t* bn254polynomial_get_coeffs_raw_ptr(PolynomialInst* p, size_t* size /*OUT*/, size_t* device_id /*OUT*/);
// PolynomialInst* bn254polynomial_slice(PolynomialInst* p, size_t offset, size_t stride, size_t size);
// IntegrityPointer* bn254polynomial_get_coeff_view(PolynomialInst* p, size_t* size /*OUT*/, size_t* device_id /*OUT*/);
// IntegrityPointer* bn254polynomial_get_rou_evaluations_view(PolynomialInst* p, size_t nof_evals, bool is_reversed, size_t* size /*OUT*/, size_t* device_id /*OUT*/);
// const scalar_t* bn254polynomial_intergrity_ptr_get(IntegrityPointer* p);
// bool bn254polynomial_intergrity_ptr_is_valid(IntegrityPointer* p);
// void bn254polynomial_intergrity_ptr_destroy(IntegrityPointer* p);

#ifdef __cplusplus
}
#endif

#endif

