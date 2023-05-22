#include <stdbool.h>
#include <cuda.h>
// msm.h

#ifndef _BN254_MSM_H
#define _BN254_MSM_H

#ifdef __cplusplus
extern "C" {
#endif

// Incomplete declaration of BN254 projective and affine structs
typedef struct BN254_projective_t BN254_projective_t;
typedef struct BN254_affine_t BN254_affine_t;
typedef struct BN254_scalar_t BN254_scalar_t;

int msm_cuda_bn254(BN254_projective_t* out, BN254_affine_t* points,
                   BN254_scalar_t* scalars, size_t count, size_t device_id);

int msm_batch_cuda_bn254(BN254_projective_t* out, BN254_affine_t* points,
                         BN254_scalar_t* scalars, size_t batch_size,
                         size_t msm_size, size_t device_id);

int commit_cuda_bn254(BN254_projective_t* d_out, BN254_scalar_t* d_scalars,
                      BN254_affine_t* d_points, size_t count, size_t device_id);

int commit_batch_cuda_bn254(BN254_projective_t* d_out, BN254_scalar_t* d_scalars,
                            BN254_affine_t* d_points, size_t count,
                            size_t batch_size, size_t device_id);

#ifdef __cplusplus
}
#endif

#endif /* _BN254_MSM_H */
