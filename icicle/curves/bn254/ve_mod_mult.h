#include <stdbool.h>
#include <cuda.h>
// ve_mod_mult.h

#ifndef _BN254_VEC_MULT_H
#define _BN254_VEC_MULT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct BN254_projective_t BN254_projective_t;
typedef struct BN254_scalar_t BN254_scalar_t;

int32_t vec_mod_mult_point_bn254(BN254_projective_t *inout, BN254_scalar_t *scalar_vec, size_t n_elments, size_t device_id);
int32_t vec_mod_mult_scalar_bn254(BN254_scalar_t *inout, BN254_scalar_t *scalar_vec, size_t n_elments, size_t device_id);
int32_t matrix_vec_mod_mult_bn254(BN254_scalar_t *matrix_flattened, BN254_scalar_t *input, BN254_scalar_t *output, size_t n_elments, size_t device_id);


#ifdef __cplusplus
}
#endif

#endif /* _BN254_VEC_MULT_H */
