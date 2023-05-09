#ifndef _BN254
#define _BN254
#include <cuda.h>
#include "lde.cuh"
#include "ntt.cuh"
#include "../vector_manipulation/ve_mod_mult.cuh"

extern "C" BN254::scalar_t* build_domain_cuda_bn254(uint32_t domain_size, uint32_t logn, bool inverse, size_t device_id = 0)
{
    try
    {
        if (inverse) {
            return fill_twiddle_factors_array(domain_size, BN254::scalar_t::omega_inv(logn));
        } else {
            return fill_twiddle_factors_array(domain_size, BN254::scalar_t::omega(logn));
        }
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return nullptr;
    }
}

extern "C" int ntt_cuda_bn254(BN254::scalar_t *arr, uint32_t n, bool inverse, size_t device_id = 0)
{
    try
    {
        return ntt_end2end_template<BN254::scalar_t,BN254::scalar_t>(arr, n, inverse); // TODO: pass device_id
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        
        return -1;        
    }
}

extern "C" int ecntt_cuda_bn254(BN254::projective_t *arr, uint32_t n, bool inverse, size_t device_id = 0)
{
    try
    {
        return ntt_end2end_template<BN254::projective_t,BN254::scalar_t>(arr, n, inverse); // TODO: pass device_id
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int ntt_batch_cuda_bn254(BN254::scalar_t *arr, uint32_t arr_size, uint32_t batch_size, bool inverse, size_t device_id = 0)
{
    try
    {
        return ntt_end2end_batch_template<BN254::scalar_t,BN254::scalar_t>(arr, arr_size, batch_size, inverse); // TODO: pass device_id
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int ecntt_batch_cuda_bn254(BN254::projective_t *arr, uint32_t arr_size, uint32_t batch_size, bool inverse, size_t device_id = 0)
{
    try
    {
        return ntt_end2end_batch_template<BN254::projective_t,BN254::scalar_t>(arr, arr_size, batch_size, inverse); // TODO: pass device_id
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int interpolate_scalars_cuda_bn254(BN254::scalar_t* d_out, BN254::scalar_t *d_evaluations, BN254::scalar_t *d_domain, unsigned n, unsigned device_id = 0)
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

extern "C" int interpolate_scalars_batch_cuda_bn254(BN254::scalar_t* d_out, BN254::scalar_t* d_evaluations, BN254::scalar_t* d_domain, unsigned n,
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

extern "C" int interpolate_points_cuda_bn254(BN254::projective_t* d_out, BN254::projective_t *d_evaluations, BN254::scalar_t *d_domain, unsigned n, size_t device_id = 0)
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

extern "C" int interpolate_points_batch_cuda_bn254(BN254::projective_t* d_out, BN254::projective_t* d_evaluations, BN254::scalar_t* d_domain,
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

extern "C" int evaluate_scalars_cuda_bn254(BN254::scalar_t* d_out, BN254::scalar_t *d_coefficients, BN254::scalar_t *d_domain, 
                                     unsigned domain_size, unsigned n, unsigned device_id = 0)
{
    try
    {
        BN254::scalar_t* _null = nullptr;
        return evaluate(d_out, d_coefficients, d_domain, domain_size, n, false, _null);
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int evaluate_scalars_batch_cuda_bn254(BN254::scalar_t* d_out, BN254::scalar_t* d_coefficients, BN254::scalar_t* d_domain, unsigned domain_size,
                                           unsigned n, unsigned batch_size, size_t device_id = 0)
{
    try
    {
        BN254::scalar_t* _null = nullptr;
        return evaluate_batch(d_out, d_coefficients, d_domain, domain_size, n, batch_size, false, _null);
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int evaluate_points_cuda_bn254(BN254::projective_t* d_out, BN254::projective_t *d_coefficients, BN254::scalar_t *d_domain, 
                                    unsigned domain_size, unsigned n, size_t device_id = 0)
{
    try
    {
        BN254::scalar_t* _null = nullptr;
        return evaluate(d_out, d_coefficients, d_domain, domain_size, n, false, _null);
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int evaluate_points_batch_cuda_bn254(BN254::projective_t* d_out, BN254::projective_t* d_coefficients, BN254::scalar_t* d_domain, unsigned domain_size,
                                          unsigned n, unsigned batch_size, size_t device_id = 0)
{
    try
    {
        BN254::scalar_t* _null = nullptr;
        return evaluate_batch(d_out, d_coefficients, d_domain, domain_size, n, batch_size, false, _null);
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int evaluate_scalars_on_coset_cuda_bn254(BN254::scalar_t* d_out, BN254::scalar_t *d_coefficients, BN254::scalar_t *d_domain, unsigned domain_size,
                                              unsigned n, BN254::scalar_t *coset_powers, unsigned device_id = 0)
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

extern "C" int evaluate_scalars_on_coset_batch_cuda_bn254(BN254::scalar_t* d_out, BN254::scalar_t* d_coefficients, BN254::scalar_t* d_domain, unsigned domain_size, 
                                                    unsigned n, unsigned batch_size, BN254::scalar_t *coset_powers, size_t device_id = 0)
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

extern "C" int evaluate_points_on_coset_cuda_bn254(BN254::projective_t* d_out, BN254::projective_t *d_coefficients, BN254::scalar_t *d_domain, unsigned domain_size,
                                             unsigned n, BN254::scalar_t *coset_powers, size_t device_id = 0)
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

extern "C" int evaluate_points_on_coset_batch_cuda_bn254(BN254::projective_t* d_out, BN254::projective_t* d_coefficients, BN254::scalar_t* d_domain, unsigned domain_size, 
                                                   unsigned n, unsigned batch_size, BN254::scalar_t *coset_powers, size_t device_id = 0)
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

extern "C" int reverse_order_scalars_cuda_bn254(BN254::scalar_t* arr, int n, size_t device_id = 0)
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

extern "C" int reverse_order_scalars_batch_cuda_bn254(BN254::scalar_t* arr, int n, int batch_size, size_t device_id = 0)
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

extern "C" int reverse_order_points_cuda_bn254(BN254::projective_t* arr, int n, size_t device_id = 0)
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

extern "C" int reverse_order_points_batch_cuda_bn254(BN254::projective_t* arr, int n, int batch_size, size_t device_id = 0)
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