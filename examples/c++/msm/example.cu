#include <fstream>
#include <iostream>
#include <iomanip>

// select the curve. TODO: can we use curve names instead of IDs?
#define CURVE_ID 1

// include MSM template
#include "icicle/appUtils/msm/msm.cu"
using namespace curve_config;

int main(int argc, char* argv[])
{
  std::cout << "Icicle example: Muli-Scalar Multiplication (MSM)" << std::endl;
  std::cout << "Example parameters" << std::endl;
  int batch_size = 1;
  std::cout << "Batch size: " << batch_size << std::endl;
  unsigned msm_size = 1048576;
  std::cout << "MSM size: " << msm_size << std::endl;
  int N = batch_size * msm_size;
  
  std::cout << "Generating random inputs on-host" << std::endl;
  scalar_t* scalars = new scalar_t[N];
  affine_t* points = new affine_t[N];
  projective_t result;
  scalar_t::RandHostMany(scalars, N);
  projective_t::RandHostManyAffine(points, N);

  std::cout << "Configuring MSM to use on-host inputs" << std::endl;
  // Create a CUDA stream
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cudaEvent_t start, stop;
  float time;
  // Create two events to time the MSM kernel
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  // Create a device context
  device_context::DeviceContext ctx = {
    stream, // stream
    0,      // device_id
    0,      // mempool
  };
  // Create a MSM configuration
  msm::MSMConfig config = {
    ctx,   // DeviceContext
    0,     // points_size
    1,     // precompute_factor
    0,     // c
    0,     // bitsize
    10,    // large_bucket_factor
    batch_size,    
    false, // are_scalars_on_device
    false, // are_scalars_montgomery_form
    false, // are_points_on_device
    false, // are_points_montgomery_form
    false,  // are_results_on_device
    false, // is_big_triangle
    true,  // is_async
  };

  std::cout << "Running MSM kernel with on-host inputs" << std::endl;
  // Record the start event on the stream
  cudaEventRecord(start, stream);
  // Execute the MSM kernel
  msm::MSM<scalar_t, affine_t, projective_t>(scalars, points, msm_size, config, &result);
  // Record the stop event on the stream
  cudaEventRecord(stop, stream);
  // Wait for the stop event to complete
  cudaEventSynchronize(stop);
  // Calculate the elapsed time between the start and stop events
  cudaEventElapsedTime(&time, start, stop);
  // Destroy the events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  // Print the elapsed time
  std::cout << "Kernel runtime: " << std::fixed << std::setprecision(3) << time * 1e-3 << " sec." << std::endl;
  // Print the result
  std::cout << projective_t::to_affine(result) << std::endl;

  std::cout << "Copying inputs on-device" << std::endl;
  scalar_t* scalars_d;
  affine_t* points_d;
  projective_t* result_d;
  cudaMalloc(&scalars_d, sizeof(scalar_t) * N);
  cudaMalloc(&points_d, sizeof(affine_t) * N);
  cudaMalloc(&result_d, sizeof(projective_t));
  cudaMemcpy(scalars_d, scalars, sizeof(scalar_t) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(points_d, points, sizeof(affine_t) * N, cudaMemcpyHostToDevice);

  std::cout << "Reconfiguring MSM to use on-device inputs" << std::endl;
  config.are_results_on_device = true;
  config.are_scalars_on_device = true;
  config.are_points_on_device = true;

  std::cout << "Running MSM kernel with on-device inputs" << std::endl;
  // Create two events to time the MSM kernel
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  // Record the start event on the stream
  cudaEventRecord(start, stream);
  // Execute the MSM kernel
  msm::MSM<scalar_t, affine_t, projective_t>(scalars_d, points_d, msm_size, config, result_d);
  // Record the stop event on the stream
  cudaEventRecord(stop, stream);
  // Wait for the stop event to complete
  cudaEventSynchronize(stop);
  // Calculate the elapsed time between the start and stop events
  cudaEventElapsedTime(&time, start, stop);
  // Destroy the events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  // Print the elapsed time
  std::cout << "Kernel runtime: " << std::fixed << std::setprecision(3) << time * 1e-3 << " sec." << std::endl;
  // Copy the result back to the host
  cudaMemcpy(&result, result_d, sizeof(projective_t), cudaMemcpyDeviceToHost);
  // Print the result
  std::cout << projective_t::to_affine(result) << std::endl;
  // Free the device memory
  cudaFree(scalars_d);
  cudaFree(points_d);
  cudaFree(result_d);
  cudaStreamDestroy(stream);
  return 0;
}
