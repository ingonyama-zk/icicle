#include <stdbool.h>

#ifndef _GRUMPKIN_MSM_H
  #define _GRUMPKIN_MSM_H

  #ifdef __cplusplus
extern "C" {
  #endif

typedef struct scalar_t scalar_t;
typedef struct projective_t projective_t;
typedef struct affine_t affine_t;
typedef struct MSMConfig MSMConfig;

int grumpkin_msm(const scalar_t* scalars, const affine_t* points, int count, MSMConfig* config, projective_t* out);
int grumpkin_msm_precompute_bases(affine_t* input_bases, int bases_size, MSMConfig* config, affine_t* output_bases);

  #ifdef __cplusplus
}
  #endif

#endif
