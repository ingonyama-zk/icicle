#include <stdexcept>
#include <cuda.h>

#include "../../primitives/projective.cuh"
#include "../../primitives/affine.cuh"
#include "../../curves/curve_config.cuh"

template <typename S, typename P, typename A>
void large_msm(S* scalars, A* points, unsigned size, P* result);

template <typename S, typename P, typename A>
void short_msm(S *h_scalars, A *h_points, unsigned size, P* h_final_result);

extern "C"
int msm_cuda(projective_t *out, affine_t points[],
              scalar_t scalars[], size_t count, size_t device_id = 0)
{
    try
    {
        if (count>256){
            large_msm<scalar_t, projective_t, affine_t>(scalars, points, count, out);
        }
        else{
            short_msm<scalar_t, projective_t, affine_t>(scalars, points, count, out);
        }

        return CUDA_SUCCESS;
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}
