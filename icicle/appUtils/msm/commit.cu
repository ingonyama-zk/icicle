#include <cuda.h>
#include "commit.cuh"
#include "msm.cuh"


/**
 * Commit to a polynomial using the MSM.
 * Note: this function just calls the MSM, it doesn't convert between evaluation and coefficient form of scalars or points.
 * @param d_out Ouptut point to write the result to.
 * @param d_scalars Scalars for the MSM. Must be on device.
 * @param d_points Points for the MSM. Must be on device.
 * @param count Length of `d_scalars` and `d_points` arrays (they should have equal length).
 */
extern "C"
int commit_cuda(projective_t *d_out, scalar_t* d_scalars, affine_t* d_points, size_t count, size_t device_id = 0)
{
    try
    {
        // TODO: set c depending on `count` instead of just 10
        bucket_method_msm(scalar_t::NBITS, 10, d_scalars, d_points, count, d_out, true);
        return CUDA_SUCCESS;
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
int commit_batch_cuda(projective_t *d_out, scalar_t* d_scalars, affine_t* d_points, 
                      size_t count, size_t batch_size, size_t device_id = 0)
{
    try
    {
        // TODO: set c depending on `count` instead of just 10
        batched_bucket_method_msm(scalar_t::NBITS, 10, d_scalars, d_points, batch_size, count, d_out, true);
        return CUDA_SUCCESS;
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}
