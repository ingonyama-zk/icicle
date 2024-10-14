#include <stdbool.h>

#ifndef _BLS12_381_MSM_H
  #define _BLS12_381_MSM_H

  #ifdef __cplusplus
extern "C" {
  #endif

typedef struct scalar_t scalar_t;
typedef struct projective_t projective_t;
typedef struct affine_t affine_t;
typedef struct MSMConfig MSMConfig;

int bls12_381_msm(const scalar_t* scalars, const affine_t* points, int count, MSMConfig* config, projective_t* out);
int bls12_381_msm_precompute_bases(affine_t* input_bases, int bases_size, MSMConfig* config, affine_t* output_bases);

  #ifdef __cplusplus
}
  #endif

#endif
