#include <cuda.h>
#include "lde.cuh"


extern "C" int build_domain_cuda(scalar_t* d_domain, uint32_t logn, bool inverse, size_t device_id = 0)
{
    try
    {
        int domain_size = 1 << logn;
        if (inverse) {
            d_domain = fill_twiddle_factors_array(domain_size, scalar_t::omega_inv(logn));
        } else {
            d_domain = fill_twiddle_factors_array(domain_size, scalar_t::omega(logn));
        }
        return 0;
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int ntt_cuda(scalar_t *arr, uint32_t n, bool inverse, size_t device_id = 0)
{
    try
    {
        return ntt_end2end(arr, n, inverse); // TODO: pass device_id
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        
        return -1;        
    }
}

extern "C" int ecntt_cuda(projective_t *arr, uint32_t n, bool inverse, size_t device_id = 0)
{
    try
    {
        return ecntt_end2end(arr, n, inverse); // TODO: pass device_id
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int ntt_batch_cuda(scalar_t *arr, uint32_t arr_size, uint32_t batch_size, bool inverse, size_t device_id = 0)
{
    try
    {
        return ntt_end2end_batch(arr, arr_size, batch_size, inverse); // TODO: pass device_id
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int ecntt_batch_cuda(projective_t *arr, uint32_t arr_size, uint32_t batch_size, bool inverse, size_t device_id = 0)
{
    try
    {
        return ecntt_end2end_batch(arr, arr_size, batch_size, inverse); // TODO: pass device_id
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int interpolate_scalars_cuda(scalar_t *res, scalar_t *d_evaluations, scalar_t *d_domain, unsigned n, unsigned device_id = 0)
{
    try
    {
        res = interpolate_scalars(d_evaluations, d_domain, n); // TODO: pass device_id
        return 0;
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        
        return -1;
    }
}

extern "C" int interpolate_scalars_batch_cuda(scalar_t* res, scalar_t* d_evaluations, scalar_t* d_domain, 
                                              unsigned n, unsigned batch_size, size_t device_id = 0)
{
    try
    {
        res = interpolate_scalars_batch(d_evaluations, d_domain, n, batch_size); // TODO: pass device_id
        return 0;
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int interpolate_points_cuda(projective_t *res, projective_t *d_evaluations, scalar_t *d_domain, unsigned n, size_t device_id = 0)
{
    try
    {
        res = interpolate_points(d_evaluations, d_domain, n); // TODO: pass device_id
        return 0;
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        
        return -1;        
    }
}

extern "C" int interpolate_points_batch_cuda(projective_t* res, projective_t* d_evaluations, scalar_t* d_domain,
                                             unsigned n, unsigned batch_size, size_t device_id = 0)
{
    try
    {
        res = interpolate_points_batch(d_evaluations, d_domain, n, batch_size); // TODO: pass device_id
        return 0;
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int evaluate_scalars_cuda(scalar_t *res, scalar_t *d_coefficients, scalar_t *d_domain, 
                                     unsigned domain_size, unsigned n, unsigned device_id = 0)
{
    try
    {
        res = evaluate_scalars(d_coefficients, d_domain, domain_size, n); // TODO: pass device_id
        return 0;
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        
        return -1;        
    }
}

extern "C" int evaluate_scalars_batch_cuda(scalar_t* res, scalar_t* d_coefficients, scalar_t* d_domain, 
                                           unsigned domain_size, unsigned n, unsigned batch_size, size_t device_id = 0)
{
    try
    {
        res = evaluate_scalars_batch(d_coefficients, d_domain, domain_size, n, batch_size); // TODO: pass device_id
        return 0;
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int evaluate_points_cuda(projective_t *res, projective_t *d_coefficients, scalar_t *d_domain, 
                                    unsigned domain_size, unsigned n, size_t device_id = 0)
{
    try
    {
        res = evaluate_points(d_coefficients, d_domain, domain_size, n); // TODO: pass device_id
        return 0;
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        
        return -1;        
    }
}

extern "C" int evaluate_points_batch_cuda(projective_t* res, projective_t* d_coefficients, scalar_t* d_domain,
                                          unsigned domain_size, unsigned n, unsigned batch_size, size_t device_id = 0)
{
    try
    {
        res = evaluate_points_batch(d_coefficients, d_domain, domain_size, n, batch_size); // TODO: pass device_id
        return 0;
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int evaluate_scalars_on_coset_cuda(scalar_t *res, scalar_t *d_coefficients, scalar_t *d_domain, 
                                              unsigned domain_size, unsigned n, scalar_t *coset_powers, unsigned device_id = 0)
{
    try
    {
        res = evaluate_scalars_on_coset(d_coefficients, d_domain, domain_size, n, coset_powers); // TODO: pass device_id
        return 0;
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        
        return -1;        
    }
}

extern "C" int evaluate_scalars_on_coset_batch_cuda(scalar_t* res, scalar_t* d_coefficients, scalar_t* d_domain, unsigned domain_size, 
                                                    unsigned n, unsigned batch_size, scalar_t *coset_powers, size_t device_id = 0)
{
    try
    {
        res = evaluate_scalars_on_coset_batch(d_coefficients, d_domain, domain_size, n, batch_size, coset_powers); // TODO: pass device_id
        return 0;
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int evaluate_points_on_coset_cuda(projective_t *res, projective_t *d_coefficients, scalar_t *d_domain, 
                                             unsigned domain_size, unsigned n, scalar_t *coset_powers, size_t device_id = 0)
{
    try
    {
        res = evaluate_points_on_coset(d_coefficients, d_domain, domain_size, n, coset_powers); // TODO: pass device_id
        return 0;
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        
        return -1;        
    }
}

extern "C" int evaluate_points_on_coset_batch_cuda(projective_t* res, projective_t* d_coefficients, scalar_t* d_domain, unsigned domain_size, 
                                                   unsigned n, unsigned batch_size, scalar_t *coset_powers, size_t device_id = 0)
{
    try
    {
        res = evaluate_points_on_coset_batch(d_coefficients, d_domain, domain_size, n, batch_size, coset_powers); // TODO: pass device_id
        return 0;
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}
