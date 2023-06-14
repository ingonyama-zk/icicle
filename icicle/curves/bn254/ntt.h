#include <stdbool.h>
#include <cuda.h>
// ntt.h

#ifndef _BN254_NTT_H
#define _BN254_NTT_H

#ifdef __cplusplus
extern "C" {
#endif

// Incomplete declaration of BN254 projective and affine structs
typedef struct BN254_projective_t BN254_projective_t;
typedef struct BN254_affine_t BN254_affine_t;
typedef struct BN254_scalar_t BN254_scalar_t;

int ntt_cuda_bn254(BN254_scalar_t *arr, uint32_t n, bool inverse, size_t decimation, size_t device_id);
int ntt_batch_cuda_bn254(BN254_scalar_t *arr, uint32_t arr_size, uint32_t batch_size, bool inverse, size_t device_id);

int ecntt_cuda_bn254(BN254_projective_t *arr, uint32_t n, bool inverse, size_t device_id);
int ecntt_batch_cuda_bn254(BN254_projective_t *arr, uint32_t arr_size, uint32_t batch_size, bool inverse, size_t device_id);

#ifdef __cplusplus
}
#endif

#endif /* _BN254_NTT_H */
