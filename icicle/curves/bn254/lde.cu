#ifndef _BN254_LDE
#define _BN254_LDE
#include <cuda.h>
#include "../../appUtils/ntt/lde.cu"
#include "../../appUtils/ntt/ntt.cuh"
#include "../../appUtils/vector_manipulation/ve_mod_mult.cuh"
#include "curve_config.cuh"

extern "C" BN254::scalar_t* build_domain_cuda_bn254(uint32_t domain_size, uint32_t logn, bool inverse, size_t device_id = 0, cudaStream_t stream = 0)
{
    try
    {
        cudaStreamCreate(&stream);
        if (inverse) {
            return fill_twiddle_factors_array(domain_size, BN254::scalar_t::omega_inv(logn), stream);
        } else {
            return fill_twiddle_factors_array(domain_size, BN254::scalar_t::omega(logn), stream);
        }
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return nullptr;
    }
}

extern "C" int ntt_cuda_bn254(BN254::scalar_t *arr, uint32_t n, bool inverse, Decimation decimation, size_t device_id = 0, cudaStream_t stream = 0)
{
    try
    {
        cudaStreamCreate(&stream);
        return ntt_end2end_template<BN254::scalar_t,BN254::scalar_t>(arr, n, inverse, stream); // TODO: pass device_id
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        
        return -1;        
    }
}

extern "C" int ecntt_cuda_bn254(BN254::projective_t *arr, uint32_t n, bool inverse, Decimation decimation, size_t device_id = 0, cudaStream_t stream = 0)
{
    try
    {
        cudaStreamCreate(&stream);
        return ntt_end2end_template<BN254::projective_t,BN254::scalar_t>(arr, n, inverse, stream); // TODO: pass device_id
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int ntt_batch_cuda_bn254(BN254::scalar_t *arr, uint32_t arr_size, uint32_t batch_size, bool inverse, size_t device_id = 0, cudaStream_t stream = 0)
{
    try
    {
        cudaStreamCreate(&stream);
        return ntt_end2end_batch_template<BN254::scalar_t,BN254::scalar_t>(arr, arr_size, batch_size, inverse, stream); // TODO: pass device_id
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int ecntt_batch_cuda_bn254(BN254::projective_t *arr, uint32_t arr_size, uint32_t batch_size, bool inverse, size_t device_id = 0, cudaStream_t stream = 0)
{
    try
    {
        cudaStreamCreate(&stream);
        return ntt_end2end_batch_template<BN254::projective_t,BN254::scalar_t>(arr, arr_size, batch_size, inverse, stream); // TODO: pass device_id
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int interpolate_scalars_cuda_bn254(BN254::scalar_t* d_out, BN254::scalar_t *d_evaluations, BN254::scalar_t *d_domain, unsigned n, unsigned device_id = 0, cudaStream_t stream = 0)
{
    try
    {
        BN254::scalar_t* _null = nullptr;
        return interpolate(d_out, d_evaluations, d_domain, n, false, _null, stream);
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int interpolate_scalars_batch_cuda_bn254(BN254::scalar_t* d_out, BN254::scalar_t* d_evaluations, BN254::scalar_t* d_domain, unsigned n,
                                              unsigned batch_size, size_t device_id = 0, cudaStream_t stream = 0)
{
    try
    {
        BN254::scalar_t* _null = nullptr;
        cudaStreamCreate(&stream);
        return interpolate_batch(d_out, d_evaluations, d_domain, n, batch_size, false, _null, stream);
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int interpolate_scalars_on_coset_cuda_bn254(BN254::scalar_t* d_out, BN254::scalar_t *d_evaluations, BN254::scalar_t *d_domain, unsigned n, BN254::scalar_t *coset_powers, unsigned device_id = 0, cudaStream_t stream = 0)
{
    try
    {
        return interpolate(d_out, d_evaluations, d_domain, n, true, coset_powers, stream);
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int interpolate_scalars_batch_on_coset_cuda_bn254(BN254::scalar_t* d_out, BN254::scalar_t* d_evaluations, BN254::scalar_t* d_domain, unsigned n,
                                              unsigned batch_size, BN254::scalar_t* coset_powers, size_t device_id = 0, cudaStream_t stream = 0)
{
    try
    {
        cudaStreamCreate(&stream);
        return interpolate_batch(d_out, d_evaluations, d_domain, n, batch_size, true, coset_powers, stream);
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int interpolate_points_cuda_bn254(BN254::projective_t* d_out, BN254::projective_t *d_evaluations, BN254::scalar_t *d_domain, unsigned n, size_t device_id = 0, cudaStream_t stream = 0)
{
    try
    {
        BN254::scalar_t* _null = nullptr;
        return interpolate(d_out, d_evaluations, d_domain, n, false, _null, stream);
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int interpolate_points_batch_cuda_bn254(BN254::projective_t* d_out, BN254::projective_t* d_evaluations, BN254::scalar_t* d_domain,
                                             unsigned n, unsigned batch_size, size_t device_id = 0, cudaStream_t stream = 0)
{
    try
    {
        BN254::scalar_t* _null = nullptr;
        cudaStreamCreate(&stream);
        return interpolate_batch(d_out, d_evaluations, d_domain, n, batch_size, false, _null, stream);
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int evaluate_scalars_cuda_bn254(BN254::scalar_t* d_out, BN254::scalar_t *d_coefficients, BN254::scalar_t *d_domain, 
                                     unsigned domain_size, unsigned n, unsigned device_id = 0, cudaStream_t stream = 0)
{
    try
    {
        BN254::scalar_t* _null = nullptr;
        cudaStreamCreate(&stream);
        return evaluate(d_out, d_coefficients, d_domain, domain_size, n, false, _null, stream);
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int evaluate_scalars_batch_cuda_bn254(BN254::scalar_t* d_out, BN254::scalar_t* d_coefficients, BN254::scalar_t* d_domain, unsigned domain_size,
                                           unsigned n, unsigned batch_size, size_t device_id = 0, cudaStream_t stream = 0)
{
    try
    {
        BN254::scalar_t* _null = nullptr;
        cudaStreamCreate(&stream);
        return evaluate_batch(d_out, d_coefficients, d_domain, domain_size, n, batch_size, false, _null, stream);
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int evaluate_points_cuda_bn254(BN254::projective_t* d_out, BN254::projective_t *d_coefficients, BN254::scalar_t *d_domain, 
                                    unsigned domain_size, unsigned n, size_t device_id = 0, cudaStream_t stream = 0)
{
    try
    {
        BN254::scalar_t* _null = nullptr;
        cudaStreamCreate(&stream);
        return evaluate(d_out, d_coefficients, d_domain, domain_size, n, false, _null, stream);
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int evaluate_points_batch_cuda_bn254(BN254::projective_t* d_out, BN254::projective_t* d_coefficients, BN254::scalar_t* d_domain, unsigned domain_size,
                                          unsigned n, unsigned batch_size, size_t device_id = 0, cudaStream_t stream = 0)
{
    try
    {
        BN254::scalar_t* _null = nullptr;
        cudaStreamCreate(&stream);
        return evaluate_batch(d_out, d_coefficients, d_domain, domain_size, n, batch_size, false, _null, stream);
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int evaluate_scalars_on_coset_cuda_bn254(BN254::scalar_t* d_out, BN254::scalar_t *d_coefficients, BN254::scalar_t *d_domain, unsigned domain_size,
                                              unsigned n, BN254::scalar_t *coset_powers, unsigned device_id = 0, cudaStream_t stream = 0)
{
    try
    {
        cudaStreamCreate(&stream);
        return evaluate(d_out, d_coefficients, d_domain, domain_size, n, true, coset_powers, stream);
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int evaluate_scalars_on_coset_batch_cuda_bn254(BN254::scalar_t* d_out, BN254::scalar_t* d_coefficients, BN254::scalar_t* d_domain, unsigned domain_size, 
                                                    unsigned n, unsigned batch_size, BN254::scalar_t *coset_powers, size_t device_id = 0, cudaStream_t stream = 0)
{
    try
    {
        cudaStreamCreate(&stream);
        return evaluate_batch(d_out, d_coefficients, d_domain, domain_size, n, batch_size, true, coset_powers, stream);
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int evaluate_points_on_coset_cuda_bn254(BN254::projective_t* d_out, BN254::projective_t *d_coefficients, BN254::scalar_t *d_domain, unsigned domain_size,
                                             unsigned n, BN254::scalar_t *coset_powers, size_t device_id = 0, cudaStream_t stream = 0)
{
    try
    {
        cudaStreamCreate(&stream);
        return evaluate(d_out, d_coefficients, d_domain, domain_size, n, true, coset_powers, stream);
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int evaluate_points_on_coset_batch_cuda_bn254(BN254::projective_t* d_out, BN254::projective_t* d_coefficients, BN254::scalar_t* d_domain, unsigned domain_size, 
                                                   unsigned n, unsigned batch_size, BN254::scalar_t *coset_powers, size_t device_id = 0, cudaStream_t stream = 0)
{
    try
    {
        cudaStreamCreate(&stream);
        return evaluate_batch(d_out, d_coefficients, d_domain, domain_size, n, batch_size, true, coset_powers, stream);
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int ntt_inplace_batch_cuda_bn254(BN254::scalar_t* d_inout, BN254::scalar_t* d_twiddles,
                                           unsigned n, unsigned batch_size, bool inverse, size_t device_id = 0, cudaStream_t stream = 0)
{
    try
    {

        cudaStreamCreate(&stream);
        BN254::scalar_t* _null = nullptr;
        ntt_inplace_batch_template(d_inout, d_twiddles, n, batch_size, inverse, false, _null, stream, true);
        return CUDA_SUCCESS; //TODO: we should implement this https://leimao.github.io/blog/Proper-CUDA-Error-Checking/
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int ntt_inplace_coset_batch_cuda_bn254(BN254::scalar_t* d_inout, BN254::scalar_t* d_twiddles,
                                           unsigned n, unsigned batch_size, bool inverse, bool is_coset, BN254::scalar_t* coset, size_t device_id = 0, cudaStream_t stream = 0)
{
    try
    {
        cudaStreamCreate(&stream);
        ntt_inplace_batch_template(d_inout, d_twiddles, n, batch_size, inverse, is_coset, coset, stream, true);
        return CUDA_SUCCESS; //TODO: we should implement this https://leimao.github.io/blog/Proper-CUDA-Error-Checking/
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int reverse_order_scalars_cuda_bn254(BN254::scalar_t* arr, int n, size_t device_id = 0, cudaStream_t stream = 0)
{
    try
    {
        uint32_t logn = uint32_t(log(n) / log(2));
        cudaStreamCreate(&stream);
        reverse_order(arr, n, logn, stream);
        cudaStreamSynchronize(stream);
        return 0;
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int reverse_order_scalars_batch_cuda_bn254(BN254::scalar_t* arr, int n, int batch_size, size_t device_id = 0, cudaStream_t stream = 0)
{
    try
    {
        uint32_t logn = uint32_t(log(n) / log(2));
        cudaStreamCreate(&stream);
        reverse_order_batch(arr, n, logn, batch_size, stream);
        return 0;
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int reverse_order_points_cuda_bn254(BN254::projective_t* arr, int n, size_t device_id = 0, cudaStream_t stream = 0)
{
    try
    {
        uint32_t logn = uint32_t(log(n) / log(2));
        cudaStreamCreate(&stream);
        reverse_order(arr, n, logn, stream);
        return 0;
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int reverse_order_points_batch_cuda_bn254(BN254::projective_t* arr, int n, int batch_size, size_t device_id = 0, cudaStream_t stream = 0)
{
    try
    {
        uint32_t logn = uint32_t(log(n) / log(2));
        cudaStreamCreate(&stream);
        reverse_order_batch(arr, n, logn, batch_size, stream);
        return 0;
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}
#endif