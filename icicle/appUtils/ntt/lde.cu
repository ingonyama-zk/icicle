#include <cuda.h>
#include "lde.cuh"


extern "C" scalar_t* build_domain_cuda(uint32_t domain_size, uint32_t logn, bool inverse, size_t device_id = 0)
{
    try
    {
        if (inverse) {
            return fill_twiddle_factors_array(domain_size, scalar_t::omega_inv(logn));
        } else {
            return fill_twiddle_factors_array(domain_size, scalar_t::omega(logn));
        }
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return nullptr;
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

extern "C" scalar_t* interpolate_scalars_cuda(scalar_t *d_evaluations, scalar_t *d_domain, unsigned n, unsigned device_id = 0)
{
    try
    {
        return interpolate_scalars(d_evaluations, d_domain, n); // TODO: pass device_id
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return nullptr;
    }
}

extern "C" scalar_t* interpolate_scalars_batch_cuda(scalar_t* d_evaluations, scalar_t* d_domain, unsigned n,
                                                    unsigned batch_size, size_t device_id = 0)
{
    try
    {
        return interpolate_scalars_batch(d_evaluations, d_domain, n, batch_size); // TODO: pass device_id
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return nullptr;
    }
}

extern "C" projective_t* interpolate_points_cuda(projective_t *d_evaluations, scalar_t *d_domain, unsigned n, size_t device_id = 0)
{
    try
    {
        return interpolate_points(d_evaluations, d_domain, n); // TODO: pass device_id
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return nullptr;
    }
}

extern "C" projective_t* interpolate_points_batch_cuda(projective_t* d_evaluations, scalar_t* d_domain,
                                                       unsigned n, unsigned batch_size, size_t device_id = 0)
{
    try
    {
        return interpolate_points_batch(d_evaluations, d_domain, n, batch_size); // TODO: pass device_id
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return nullptr;
    }
}

extern "C" scalar_t* evaluate_scalars_cuda(scalar_t *d_coefficients, scalar_t *d_domain, 
                                           unsigned domain_size, unsigned n, unsigned device_id = 0)
{
    try
    {
        return evaluate_scalars(d_coefficients, d_domain, domain_size, n); // TODO: pass device_id
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return nullptr;
    }
}

extern "C" scalar_t* evaluate_scalars_batch_cuda(scalar_t* d_coefficients, scalar_t* d_domain, unsigned domain_size,
                                                 unsigned n, unsigned batch_size, size_t device_id = 0)
{
    try
    {
        return evaluate_scalars_batch(d_coefficients, d_domain, domain_size, n, batch_size); // TODO: pass device_id
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return nullptr;
    }
}

extern "C" projective_t* evaluate_points_cuda(projective_t *d_coefficients, scalar_t *d_domain, 
                                              unsigned domain_size, unsigned n, size_t device_id = 0)
{
    try
    {
        return evaluate_points(d_coefficients, d_domain, domain_size, n); // TODO: pass device_id
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return nullptr;
    }
}

extern "C" projective_t* evaluate_points_batch_cuda(projective_t* d_coefficients, scalar_t* d_domain, unsigned domain_size,
                                                    unsigned n, unsigned batch_size, size_t device_id = 0)
{
    try
    {
        return evaluate_points_batch(d_coefficients, d_domain, domain_size, n, batch_size); // TODO: pass device_id
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return nullptr;
    }
}

extern "C" scalar_t* evaluate_scalars_on_coset_cuda(scalar_t *d_coefficients, scalar_t *d_domain, unsigned domain_size,
                                                    unsigned n, scalar_t *coset_powers, unsigned device_id = 0)
{
    try
    {
        return evaluate_scalars_on_coset(d_coefficients, d_domain, domain_size, n, coset_powers); // TODO: pass device_id
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return nullptr;
    }
}

extern "C" scalar_t* evaluate_scalars_on_coset_batch_cuda(scalar_t* d_coefficients, scalar_t* d_domain, unsigned domain_size, 
                                                          unsigned n, unsigned batch_size, scalar_t *coset_powers, size_t device_id = 0)
{
    try
    {
        return evaluate_scalars_on_coset_batch(d_coefficients, d_domain, domain_size, n, batch_size, coset_powers); // TODO: pass device_id
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return nullptr;
    }
}

extern "C" projective_t* evaluate_points_on_coset_cuda(projective_t *d_coefficients, scalar_t *d_domain, unsigned domain_size,
                                                       unsigned n, scalar_t *coset_powers, size_t device_id = 0)
{
    try
    {
        return evaluate_points_on_coset(d_coefficients, d_domain, domain_size, n, coset_powers); // TODO: pass device_id
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return nullptr;
    }
}

extern "C" projective_t* evaluate_points_on_coset_batch_cuda(projective_t* d_coefficients, scalar_t* d_domain, unsigned domain_size, 
                                                             unsigned n, unsigned batch_size, scalar_t *coset_powers, size_t device_id = 0)
{
    try
    {
        return evaluate_points_on_coset_batch(d_coefficients, d_domain, domain_size, n, batch_size, coset_powers); // TODO: pass device_id
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return nullptr;
    }
}

extern "C" int reverse_order_scalars_cuda(scalar_t* arr, int n, size_t device_id = 0)
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

extern "C" int reverse_order_scalars_batch_cuda(scalar_t* arr, int n, int batch_size, size_t device_id = 0)
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

extern "C" int reverse_order_points_cuda(projective_t* arr, int n, int logn, size_t device_id = 0)
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

extern "C" int reverse_order_points_batch_cuda(projective_t* arr, int n, int logn, int batch_size, size_t device_id = 0)
{
    try
    {
        reverse_order_batch(arr, n, logn, batch_size);
        return 0;
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

// int main() {
//     size_t log_l = 2;
//     size_t l = 1 << log_l;
//     projective_t* a = (projective_t*) malloc(l * sizeof(projective_t));
//     for (int i = 0; i < l; i++)
//         a[i] = projective_t::rand_host();
//     projective_t* d_a;
//     cudaMalloc(&d_a, sizeof(projective_t) * l);
//     cudaMemcpy(d_a, a, sizeof(projective_t) * l, cudaMemcpyHostToDevice);
//     scalar_t* d_domain = build_domain_cuda(log_l, false, 0);
//     projective_t* d_res = interpolate_points_cuda(d_a, d_domain, l, 0);
//     projective_t* res = (projective_t*) malloc(l * sizeof(projective_t));
//     cudaMemcpy(res, d_res, sizeof(projective_t) * l, cudaMemcpyDeviceToHost);
//     cudaFree(d_a);
//     cudaFree(d_res);
//     std::cout << res[0] << std::endl;
//     return 0;
// }
