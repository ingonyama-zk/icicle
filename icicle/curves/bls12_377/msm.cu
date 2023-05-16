#ifndef _BLS12_377_MSM
#define _BLS12_377_MSM
#include "../../appUtils/msm/msm.cu"
#include <stdexcept>
#include <cuda.h>
#include "curve_config.cuh"


extern "C"
int msm_cuda_bls12_377(BLS12_377::projective_t *out, BLS12_377::affine_t points[],
              BLS12_377::scalar_t scalars[], size_t count, size_t device_id = 0, cudaStream_t stream = 0)
{
    try
    {
        if (count>256){
            large_msm<BLS12_377::scalar_t, BLS12_377::projective_t, BLS12_377::affine_t>(scalars, points, count, out, false, stream);
        }
        else{
            short_msm<BLS12_377::scalar_t, BLS12_377::projective_t, BLS12_377::affine_t>(scalars, points, count, out, false, stream);
        }

        return CUDA_SUCCESS;
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int msm_batch_cuda_bls12_377(BLS12_377::projective_t* out, BLS12_377::affine_t points[],
                              BLS12_377::scalar_t scalars[], size_t batch_size, size_t msm_size, size_t device_id = 0, cudaStream_t stream = 0)
{
  try
  {
    batched_large_msm<BLS12_377::scalar_t, BLS12_377::projective_t, BLS12_377::affine_t>(scalars, points, batch_size, msm_size, out, false, stream);

    return CUDA_SUCCESS;
  }
  catch (const std::runtime_error &ex)
  {
    printf("error %s", ex.what());
    return -1;
  }
}

/**
 * Commit to a polynomial using the MSM.
 * Note: this function just calls the MSM, it doesn't convert between evaluation and coefficient form of scalars or points.
 * @param d_out Ouptut point to write the result to.
 * @param d_scalars Scalars for the MSM. Must be on device.
 * @param d_points Points for the MSM. Must be on device.
 * @param count Length of `d_scalars` and `d_points` arrays (they should have equal length).
 */
 extern "C"
 int commit_cuda_bls12_377(BLS12_377::projective_t* d_out, BLS12_377::scalar_t* d_scalars, BLS12_377::affine_t* d_points, size_t count, size_t device_id = 0, cudaStream_t stream = 0)
 {
     try
     {
         large_msm(d_scalars, d_points, count, d_out, true, stream);
         return 0;
     }
     catch (const std::runtime_error &ex)
     {
         printf("error %s", ex.what());
         return -1;
     }
 }
 
 /**
  * Commit to a batch of polynomials using the MSM.
  * Note: this function just calls the MSM, it doesn't convert between evaluation and coefficient form of scalars or points.
  * @param d_out Ouptut point to write the results to.
  * @param d_scalars Scalars for the MSMs of all polynomials. Must be on device.
  * @param d_points Points for the MSMs. Must be on device. It is assumed that this set of bases is used for each MSM.
  * @param count Length of `d_points` array, `d_scalar` has length `count` * `batch_size`.
  * @param batch_size Size of the batch.
  */
 extern "C"
 int commit_batch_cuda_bls12_377(BLS12_377::projective_t* d_out, BLS12_377::scalar_t* d_scalars, BLS12_377::affine_t* d_points, size_t count, size_t batch_size, size_t device_id = 0, cudaStream_t stream = 0)
 {
     try
     {
         batched_large_msm(d_scalars, d_points, batch_size, count, d_out, true, stream);
         return 0;
     }
     catch (const std::runtime_error &ex)
     {
         printf("error %s", ex.what());
         return -1;
     }
 }

 #endif
