#include <cuda_runtime.h>
#include <stdbool.h>

#ifndef _BABYBEAR_POLY_H
#define _BABYBEAR_POLY_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct scalar_t scalar_t;
typedef struct PolynomialInst PolynomialInst;
typedef struct IntegrityPointer IntegrityPointer;

bool babybearpolynomial_init_cuda_backend();
PolynomialInst* babybearpolynomial_create_from_coefficients(scalar_t* coeffs, size_t size);
PolynomialInst* babybearpolynomial_create_from_rou_evaluations(scalar_t* evals, size_t size);
PolynomialInst* babybearpolynomial_clone(const PolynomialInst* p);
void babybearpolynomial_print(PolynomialInst* p);
void babybearpolynomial_delete(PolynomialInst* instance);
PolynomialInst* babybearpolynomial_add(const PolynomialInst* a, const PolynomialInst* b);
void babybearpolynomial_add_inplace(PolynomialInst* a, const PolynomialInst* b);
PolynomialInst* babybearpolynomial_subtract(const PolynomialInst* a, const PolynomialInst* b);
PolynomialInst* babybearpolynomial_multiply(const PolynomialInst* a, const PolynomialInst* b);
PolynomialInst* babybearpolynomial_multiply_by_scalar(const PolynomialInst* a, const scalar_t* scalar);
void babybearpolynomial_division(const PolynomialInst* a, const PolynomialInst* b, PolynomialInst** q /*OUT*/, PolynomialInst** r /*OUT*/);
PolynomialInst* babybearpolynomial_quotient(const PolynomialInst* a, const PolynomialInst* b);
PolynomialInst* babybearpolynomial_remainder(const PolynomialInst* a, const PolynomialInst* b);
PolynomialInst* babybearpolynomial_divide_by_vanishing(const PolynomialInst* p, size_t vanishing_poly_degree);
void babybearpolynomial_add_monomial_inplace(PolynomialInst* p, const scalar_t* monomial_coeff, size_t monomial);
void babybearpolynomial_sub_monomial_inplace(PolynomialInst* p, const scalar_t* monomial_coeff, size_t monomial);
void babybearpolynomial_evaluate_on_domain(const PolynomialInst* p, scalar_t* domain, size_t domain_size, scalar_t* evals /*OUT*/);
size_t babybearpolynomial_degree(PolynomialInst* p);
size_t babybearpolynomial_copy_coeffs_range(PolynomialInst* p, scalar_t* memory, size_t start_idx, size_t end_idx);
PolynomialInst* babybearpolynomial_even(PolynomialInst* p);
PolynomialInst* babybearpolynomial_odd(PolynomialInst* p);

// scalar_t* babybearpolynomial_get_coeffs_raw_ptr(PolynomialInst* p, size_t* size /*OUT*/, size_t* device_id /*OUT*/);
// PolynomialInst* babybearpolynomial_slice(PolynomialInst* p, size_t offset, size_t stride, size_t size);
// IntegrityPointer* babybearpolynomial_get_coeff_view(PolynomialInst* p, size_t* size /*OUT*/, size_t* device_id /*OUT*/);
// IntegrityPointer* babybearpolynomial_get_rou_evaluations_view(PolynomialInst* p, size_t nof_evals, bool is_reversed, size_t* size /*OUT*/, size_t* device_id /*OUT*/);
// const scalar_t* babybearpolynomial_intergrity_ptr_get(IntegrityPointer* p);
// bool babybearpolynomial_intergrity_ptr_is_valid(IntegrityPointer* p);
// void babybearpolynomial_intergrity_ptr_destroy(IntegrityPointer* p);

#ifdef __cplusplus
}
#endif

#endif

