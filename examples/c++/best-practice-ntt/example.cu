#include <stdio.h>
#include <iostream>
#include <string>
#include <chrono>

#include "curves/params/bn254.cuh"
#include "api/bn254.h"
using namespace bn254;
using namespace ntt;


const std::string curve = "BN254";

typedef scalar_t S;
typedef scalar_t E;

const unsigned max_log_ntt_size = 27;
const unsigned nof_ntts = 1;

// on-host data
E* input;
E* output;

// on-device data
E* d_input;
E* d_output;

void initialize_input(const unsigned ntt_size, const unsigned nof_ntts, E * elements ) {
  // E::RandHostMany(elements, ntt_size * nof_ntts);
  for (unsigned i = 0; i < ntt_size * nof_ntts; i++) {
    elements[i] = E::from(i+1);
  }
}

using FpMilliseconds = std::chrono::duration<float, std::chrono::milliseconds::period>;
#define START_TIMER(timer) auto timer##_start = std::chrono::high_resolution_clock::now();
#define END_TIMER(timer, msg) printf("%s: %.0f ms\n", msg, FpMilliseconds(std::chrono::high_resolution_clock::now() - timer##_start).count());


int main(int argc, char** argv) {
  cudaDeviceReset();
  cudaDeviceProp deviceProperties;
  int deviceId=0;
  cudaGetDeviceProperties(&deviceProperties, deviceId);
  std::string gpu_full_name = deviceProperties.name;
  std::cout << gpu_full_name << std::endl;
  std::string gpu_name = gpu_full_name;
  int gpu_clock_mhz = deviceProperties.clockRate/1000.;

  std::cout << "Curve: " << curve << std::endl;

  cudaError_t err;
  
  auto ctx = device_context::get_default_device_context();
  S basic_root = S::omega(max_log_ntt_size);
  err = bn254_initialize_domain(&basic_root, ctx, /* fast twiddles */ true);
  NTTConfig<S> config = default_ntt_config<S>();
  config.batch_size = nof_ntts;
  
  const unsigned log_ntt_size = 25;
  std::cout << "log NTT size: " << log_ntt_size << std::endl;
  const unsigned ntt_size = 1 << log_ntt_size;
  config.ntt_algorithm = NttAlgorithm::MixedRadix; 
  config.batch_size = nof_ntts;
  // all data is on device, blocking calls
  config.are_inputs_on_device = false;
  config.are_outputs_on_device = false;
  config.is_async = false;
  
  std::cout << "Allocating on-host arrays" << std::endl;
  input = (E*) malloc(sizeof(E) * ntt_size * nof_ntts);
  output = (E*) malloc(sizeof(E) * ntt_size * nof_ntts);
  std::cout << "Initializing input data" << std::endl;
  initialize_input(ntt_size, nof_ntts, input );
  
  std::cout << "Running NTT with on-host data" << std::endl;
  START_TIMER(onhost);
  err = bn254_ntt_cuda(input, ntt_size, NTTDir::kForward, config, output);
  END_TIMER(onhost, "On-host NTT");
  if(err != cudaSuccess) {
    std::cout << "Error: " << cudaGetErrorString(err) << std  ::endl; 
  }

  std::cout << "Allocate on-device memory" << std::endl;
  cudaMallocAsync(&d_input, sizeof(E) * ntt_size * nof_ntts, config.ctx.stream);
  cudaMallocAsync(&d_output, sizeof(E) * ntt_size * nof_ntts, config.ctx.stream);

  std::cout << "Moving input data on device" << std::endl;
  START_TIMER(MemcpyHostToDevice);
  cudaMemcpyAsync(d_input, input, sizeof(E) * ntt_size, cudaMemcpyHostToDevice, config.ctx.stream);
  cudaStreamSynchronize(config.ctx.stream);
  END_TIMER(MemcpyHostToDevice, "Memcpy Host to Device");

  std::cout << "Running NTT with on-device data" << std::endl;
  config.are_inputs_on_device = true;
  config.are_outputs_on_device = true;
  START_TIMER(ondevice);
  err = bn254_ntt_cuda(d_input, ntt_size, NTTDir::kForward, config, d_output);
  END_TIMER(ondevice, "On-device NTT");
  if(err != cudaSuccess) {
    std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
  }

  std::cout << "Moving output data on host" << std::endl;
  START_TIMER(MemcpyDeviceToHost);
  cudaMemcpyAsync(output, d_output, sizeof(E) * ntt_size, cudaMemcpyDeviceToHost, config.ctx.stream);
  cudaStreamSynchronize(config.ctx.stream);
  END_TIMER(MemcpyDeviceToHost, "Memcpy Device to Host");


  std::cout << "Cleaning-up memory" << std::endl;  
  cudaFreeAsync(d_input, config.ctx.stream);
  cudaFreeAsync(d_output, config.ctx.stream);
  cudaStreamSynchronize(config.ctx.stream);
  free(input);
  free(output);

  std::cout << "Interleaving experiment" << std::endl;
  // allocate memory on host
  E * h_input[2];
  E * h_output[2]; 
  err = cudaHostAlloc((void**)&h_input[0], sizeof(E)*ntt_size, cudaHostAllocDefault);
  err = cudaHostAlloc((void**)&h_input[1], sizeof(E)*ntt_size, cudaHostAllocDefault);
  err = cudaHostAlloc((void**)&h_output[0], sizeof(E)*ntt_size, cudaHostAllocDefault);
  err = cudaHostAlloc((void**)&h_output[1], sizeof(E)*ntt_size, cudaHostAllocDefault);

  // allocate memory on device
  E * d_input[2];
  E * d_output[2];
  err = cudaMalloc((void**)&d_input[0], sizeof(E)*ntt_size);
  err = cudaMalloc((void**)&d_input[1], sizeof(E)*ntt_size);
  err = cudaMalloc((void**)&d_output[0], sizeof(E)*ntt_size);
  err = cudaMalloc((void**)&d_output[1], sizeof(E)*ntt_size);

  // initialize input data
  initialize_input(ntt_size, 1, h_input[0]);
  initialize_input(ntt_size, 1, h_input[1]);

  // Create CUDA streams for overlapping data transfers with kernel execution.
  cudaStream_t stream_compute, stream_transfer; 
  err = cudaStreamCreate(&stream_compute); 
  err = cudaStreamCreate(&stream_transfer);
  auto ctx_compute = device_context::DeviceContext{
    stream_compute, // stream
    0,              // device_id
    0,              // mempool
  };
  

  bn254_initialize_domain(&basic_root, ctx_compute, /* fast twiddles */ true);
  NTTConfig<S> config_compute = default_ntt_config<S>(ctx_compute);
  config_compute.ntt_algorithm = NttAlgorithm::MixedRadix;
  config_compute.batch_size = 1;
  config_compute.are_inputs_on_device = true;
  config_compute.are_outputs_on_device = true;
  config_compute.is_async = true;

  START_TIMER(interleaved);
  // Perform Asynchronous Memory Transfers and Kernel Execution
  cudaMemcpyAsync(d_input[0],  h_input[0],  sizeof(E)*ntt_size, cudaMemcpyHostToDevice, stream_transfer);
  cudaMemcpyAsync(h_output[0], d_output[0], sizeof(E)*ntt_size, cudaMemcpyDeviceToHost, stream_transfer);  
  bn254_ntt_cuda(d_input[1], ntt_size, NTTDir::kForward, config_compute, d_output[1]);

  // Wait for all streams to finish
  cudaStreamSynchronize(stream_transfer); 
  cudaStreamSynchronize(stream_compute);  
  END_TIMER(interleaved, "Interleaved NTT");
  
  // Clean-up
  
  cudaStreamDestroy(stream_transfer);

  std::cout << "Full Duplex Interleaving experiment" << std::endl;
  // separate streams for host-to-device and device-to-host transfers
  cudaStream_t stream_h2d, stream_d2h; 
  err = cudaStreamCreate(&stream_h2d);
  err = cudaStreamCreate(&stream_d2h);

  START_TIMER(fullduplex);
  // Perform Asynchronous Memory Transfers and Kernel Execution
  cudaMemcpyAsync(d_input[0],  h_input[0],  sizeof(E)*ntt_size, cudaMemcpyHostToDevice, stream_h2d);
  cudaMemcpyAsync(h_output[0], d_output[0], sizeof(E)*ntt_size, cudaMemcpyDeviceToHost, stream_d2h);  
  bn254_ntt_cuda(d_input[1], ntt_size, NTTDir::kForward, config_compute, d_output[1]);

  // Wait for all streams to finish
  cudaStreamSynchronize(stream_h2d); 
  cudaStreamSynchronize(stream_d2h); 
  cudaStreamSynchronize(stream_compute);  
  END_TIMER(fullduplex, "Full Duplex Interleaved NTT");

  // Clean-up
  cudaFree(d_input[0]); 
  cudaFree(d_input[1]); 
  cudaFree(d_output[0]); 
  cudaFree(d_output[1]);
  cudaFreeHost(h_input[0]); 
  cudaFreeHost(h_input[1]); 
  cudaFreeHost(h_output[0]); 
  cudaFreeHost(h_output[1]);

  std::cout << "Full Duplex Interleaving In-place experiment" << std::endl;

  
  // on-host in-place data
  E * h_inp[2];
  E * h_out[2];
  err = cudaHostAlloc((void**)&h_inp[0], sizeof(E)*ntt_size, cudaHostAllocDefault);
  err = cudaHostAlloc((void**)&h_inp[1], sizeof(E)*ntt_size, cudaHostAllocDefault);
  err = cudaHostAlloc((void**)&h_out[0], sizeof(E)*ntt_size, cudaHostAllocDefault);
  err = cudaHostAlloc((void**)&h_out[1], sizeof(E)*ntt_size, cudaHostAllocDefault);

  // on-device in-place data
  E * d_vec[2];
  err = cudaMalloc((void**)&d_vec[0], sizeof(E)*ntt_size);
  err = cudaMalloc((void**)&d_vec[1], sizeof(E)*ntt_size);

  // initialize input data
  initialize_input(ntt_size, 1, h_inp[0]);

  int nof_blocks = 32;
  std::cout << "Number of blocks: " << nof_blocks << std::endl;
  int block_size = ntt_size/nof_blocks;

  // Perform Asynchronous Memory Transfers and Kernel Execution
  // [0] memory transfers
  // [1] kernel execution
  
  START_TIMER(inplace);
  bn254_ntt_cuda(d_vec[1], ntt_size, NTTDir::kForward, config_compute, d_vec[1]);
  for (int i = 0; i <= nof_blocks; i++) {
    if (i < nof_blocks) {
      cudaMemcpyAsync(&h_out[0][i*block_size], &d_vec[0][i*block_size], sizeof(E)*block_size, cudaMemcpyDeviceToHost, stream_d2h);    
    }
    if (i>0) {
      cudaMemcpyAsync(&d_vec[0][(i-1)*block_size], &h_inp[0][(i-1)*block_size], sizeof(E)*block_size, cudaMemcpyHostToDevice, stream_h2d);
    }
    cudaStreamSynchronize(stream_d2h); 
    cudaStreamSynchronize(stream_h2d); 
  }
  cudaStreamSynchronize(stream_compute);  
  END_TIMER(inplace, "Full Duplex Interleaved In-place  NTT");
  // Clean-up
  cudaFree(d_vec[0]); 
  cudaFree(d_vec[1]);
  cudaFreeHost(h_inp[0]); 
  cudaFreeHost(h_inp[1]);
  cudaFreeHost(h_out[0]); 
  cudaFreeHost(h_out[1]);
  cudaStreamDestroy(stream_compute);
  cudaStreamDestroy(stream_d2h);
  cudaStreamDestroy(stream_h2d);
  return 0;
}
