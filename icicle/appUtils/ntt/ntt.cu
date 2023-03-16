#include <cuda.h>
#include "ntt.cuh"

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