#include <stdexcept>
#include <cuda.h>

#include "../../primitives/projective.cuh"
#include "../../primitives/affine.cuh"
#include "../../curves/curve_config.cuh"

template <typename S, typename P, typename A>
void large_msm(S* scalars, A* points, unsigned size, P* result);

template <typename S, typename P, typename A>
void short_msm(S *h_scalars, A *h_points, unsigned size, P* h_final_result);
// Rust expects
// fn msm_cuda(projective,
//     out: *mut Point,
//     points: *const PointAffineNoInfinity,
//     scalars: *const ScalarField,
//     count: usize, //TODO: is needed?
//     device_id: usize,
// ) -> c_uint;
extern "C"
int msm_cuda(projective_t *out, affine_t points[],
              scalar_t scalars[], size_t count, size_t device_id = 0)
{
    try
    {
        if (count>1024){
            large_msm<scalar_t, projective_t, affine_t>(scalars, points, count, out);
        }
        else{
            short_msm<scalar_t, projective_t, affine_t>(scalars, points, count, out);
        }
        // stub here(uint32_t *)scalar
        // printf("\nx %08X", points[0].x.export_limbs()[0]); // TODO: error code and message
        // printf("\ny %08X", points[0].y.export_limbs()[8]); // TODO: error code and message

        // uint32_t one = 1;

        // out->y.export_limbs()[0] = one; // TODO: just roundtrip stub

        return CUDA_SUCCESS;
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what()); // TODO: error code and message
        // out->z = 0; //TODO: .set_infinity()
        return -1;
    }
}
