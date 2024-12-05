#include <stdbool.h>

#ifndef _BN254_G2MSM_H
#define _BN254_G2MSM_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct scalar_t scalar_t;
typedef struct g2_projective_t g2_projective_t;
typedef struct g2_affine_t g2_affine_t;
typedef struct MSMConfig MSMConfig;

int bn254_g2_msm(const scalar_t* scalars, const g2_affine_t* points, int count, MSMConfig* config, g2_projective_t* out);
int bn254_g2_msm_precompute_bases(g2_affine_t* input_bases, int bases_size, MSMConfig* config, g2_affine_t* output_bases);

#ifdef __cplusplus
}
#endif

#endif
