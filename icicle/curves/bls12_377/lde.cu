#ifndef _BLS12_377_LDE
#define _BLS12_377_LDE
#include <cuda.h>
#include "../../appUtils/ntt/lde.cu"
#include "../../appUtils/ntt/ntt.cuh"
#include "../../appUtils/vector_manipulation/ve_mod_mult.cuh"
#include "curve_config.cuh"

extern "C" BLS12_377::scalar_t* build_domain_cuda_bls12_377(uint32_t domain_size, uint32_t logn, bool inverse, size_t device_id = 0)
{
    try
    {
        if (inverse) {
            return fill_twiddle_factors_array(domain_size, BLS12_377::scalar_t::omega_inv(logn));
        } else {
            return fill_twiddle_factors_array(domain_size, BLS12_377::scalar_t::omega(logn));
        }
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return nullptr;
    }
}

extern "C" int ntt_cuda_bls12_377(BLS12_377::scalar_t *arr, uint32_t n, bool inverse, size_t device_id = 0)
{
    try
    {
        return ntt_end2end_template<BLS12_377::scalar_t,BLS12_377::scalar_t>(arr, n, inverse); // TODO: pass device_id
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        
        return -1;        
    }
}

extern "C" int ecntt_cuda_bls12_377(BLS12_377::projective_t *arr, uint32_t n, bool inverse, size_t device_id = 0)
{
    try
    {
        return ntt_end2end_template<BLS12_377::projective_t,BLS12_377::scalar_t>(arr, n, inverse); // TODO: pass device_id
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int ntt_batch_cuda_bls12_377(BLS12_377::scalar_t *arr, uint32_t arr_size, uint32_t batch_size, bool inverse, size_t device_id = 0)
{
    try
    {
        return ntt_end2end_batch_template<BLS12_377::scalar_t,BLS12_377::scalar_t>(arr, arr_size, batch_size, inverse); // TODO: pass device_id
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int ecntt_batch_cuda_bls12_377(BLS12_377::projective_t *arr, uint32_t arr_size, uint32_t batch_size, bool inverse, size_t device_id = 0)
{
    try
    {
        return ntt_end2end_batch_template<BLS12_377::projective_t,BLS12_377::scalar_t>(arr, arr_size, batch_size, inverse); // TODO: pass device_id
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int interpolate_scalars_cuda_bls12_377(BLS12_377::scalar_t* d_out, BLS12_377::scalar_t *d_evaluations, BLS12_377::scalar_t *d_domain, unsigned n, unsigned device_id = 0)
{
    try
    {
        return interpolate(d_out, d_evaluations, d_domain, n);
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int interpolate_scalars_batch_cuda_bls12_377(BLS12_377::scalar_t* d_out, BLS12_377::scalar_t* d_evaluations, BLS12_377::scalar_t* d_domain, unsigned n,
                                              unsigned batch_size, size_t device_id = 0)
{
    try
    {
        return interpolate_batch(d_out, d_evaluations, d_domain, n, batch_size);
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int interpolate_points_cuda_bls12_377(BLS12_377::projective_t* d_out, BLS12_377::projective_t *d_evaluations, BLS12_377::scalar_t *d_domain, unsigned n, size_t device_id = 0)
{
    try
    {
        return interpolate(d_out, d_evaluations, d_domain, n);
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int interpolate_points_batch_cuda_bls12_377(BLS12_377::projective_t* d_out, BLS12_377::projective_t* d_evaluations, BLS12_377::scalar_t* d_domain,
                                             unsigned n, unsigned batch_size, size_t device_id = 0)
{
    try
    {
        return interpolate_batch(d_out, d_evaluations, d_domain, n, batch_size);
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int evaluate_scalars_cuda_bls12_377(BLS12_377::scalar_t* d_out, BLS12_377::scalar_t *d_coefficients, BLS12_377::scalar_t *d_domain, 
                                     unsigned domain_size, unsigned n, unsigned device_id = 0)
{
    try
    {
        BLS12_377::scalar_t* _null = nullptr;
        return evaluate(d_out, d_coefficients, d_domain, domain_size, n, false, _null);
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int evaluate_scalars_batch_cuda_bls12_377(BLS12_377::scalar_t* d_out, BLS12_377::scalar_t* d_coefficients, BLS12_377::scalar_t* d_domain, unsigned domain_size,
                                           unsigned n, unsigned batch_size, size_t device_id = 0)
{
    try
    {
        BLS12_377::scalar_t* _null = nullptr;
        return evaluate_batch(d_out, d_coefficients, d_domain, domain_size, n, batch_size, false, _null);
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int evaluate_points_cuda_bls12_377(BLS12_377::projective_t* d_out, BLS12_377::projective_t *d_coefficients, BLS12_377::scalar_t *d_domain, 
                                    unsigned domain_size, unsigned n, size_t device_id = 0)
{
    try
    {
        BLS12_377::scalar_t* _null = nullptr;
        return evaluate(d_out, d_coefficients, d_domain, domain_size, n, false, _null);
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int evaluate_points_batch_cuda_bls12_377(BLS12_377::projective_t* d_out, BLS12_377::projective_t* d_coefficients, BLS12_377::scalar_t* d_domain, unsigned domain_size,
                                          unsigned n, unsigned batch_size, size_t device_id = 0)
{
    try
    {
        BLS12_377::scalar_t* _null = nullptr;
        return evaluate_batch(d_out, d_coefficients, d_domain, domain_size, n, batch_size, false, _null);
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int evaluate_scalars_on_coset_cuda_bls12_377(BLS12_377::scalar_t* d_out, BLS12_377::scalar_t *d_coefficients, BLS12_377::scalar_t *d_domain, unsigned domain_size,
                                              unsigned n, BLS12_377::scalar_t *coset_powers, unsigned device_id = 0)
{
    try
    {
        return evaluate(d_out, d_coefficients, d_domain, domain_size, n, true, coset_powers);
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int evaluate_scalars_on_coset_batch_cuda_bls12_377(BLS12_377::scalar_t* d_out, BLS12_377::scalar_t* d_coefficients, BLS12_377::scalar_t* d_domain, unsigned domain_size, 
                                                    unsigned n, unsigned batch_size, BLS12_377::scalar_t *coset_powers, size_t device_id = 0)
{
    try
    {
        return evaluate_batch(d_out, d_coefficients, d_domain, domain_size, n, batch_size, true, coset_powers);
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int evaluate_points_on_coset_cuda_bls12_377(BLS12_377::projective_t* d_out, BLS12_377::projective_t *d_coefficients, BLS12_377::scalar_t *d_domain, unsigned domain_size,
                                             unsigned n, BLS12_377::scalar_t *coset_powers, size_t device_id = 0)
{
    try
    {
        return evaluate(d_out, d_coefficients, d_domain, domain_size, n, true, coset_powers);
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int evaluate_points_on_coset_batch_cuda_bls12_377(BLS12_377::projective_t* d_out, BLS12_377::projective_t* d_coefficients, BLS12_377::scalar_t* d_domain, unsigned domain_size, 
                                                   unsigned n, unsigned batch_size, BLS12_377::scalar_t *coset_powers, size_t device_id = 0)
{
    try
    {
        return evaluate_batch(d_out, d_coefficients, d_domain, domain_size, n, batch_size, true, coset_powers);
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int reverse_order_scalars_cuda_bls12_377(BLS12_377::scalar_t* arr, int n, size_t device_id = 0)
{
    try
    {
        uint32_t logn = uint32_t(log(n) / log(2));
        reverse_order(arr, n, logn);
        return 0;
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int reverse_order_scalars_batch_cuda_bls12_377(BLS12_377::scalar_t* arr, int n, int batch_size, size_t device_id = 0)
{
    try
    {
        uint32_t logn = uint32_t(log(n) / log(2));
        reverse_order_batch(arr, n, logn, batch_size);
        return 0;
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int reverse_order_points_cuda_bls12_377(BLS12_377::projective_t* arr, int n, size_t device_id = 0)
{
    try
    {
        uint32_t logn = uint32_t(log(n) / log(2));
        reverse_order(arr, n, logn);
        return 0;
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int reverse_order_points_batch_cuda_bls12_377(BLS12_377::projective_t* arr, int n, int batch_size, size_t device_id = 0)
{
    try
    {
        uint32_t logn = uint32_t(log(n) / log(2));
        reverse_order_batch(arr, n, logn, batch_size);
        return 0;
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}
#endif