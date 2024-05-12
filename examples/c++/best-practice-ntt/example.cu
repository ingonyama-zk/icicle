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


void initialize_input(const unsigned ntt_size, const unsigned nof_ntts, E * elements ) {
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

  
  // auto ctx = device_context::get_default_device_context();
  S basic_root = S::omega(max_log_ntt_size);
  
  const unsigned log_ntt_size = 22;
  const unsigned nof_ntts = 16;

  std::cout << "log NTT size: " << log_ntt_size << std::endl;
  const unsigned ntt_size = 1 << log_ntt_size;

  std::cout << "Batch size: " << nof_ntts << std::endl;

  // Create CUDA streams for overlapping data transfers with kernel execution.
  cudaStream_t stream_compute;
  cudaStreamCreate(&stream_compute); 
  auto ctx_compute = device_context::DeviceContext{
    stream_compute, // stream
    0,              // device_id
    0,              // mempool
  };
  

  bn254_initialize_domain(&basic_root, ctx_compute, /* fast twiddles */ true);
  NTTConfig<S> config_compute = default_ntt_config<S>(ctx_compute);
  config_compute.ntt_algorithm = NttAlgorithm::MixedRadix;
  config_compute.batch_size = nof_ntts;
  config_compute.are_inputs_on_device = true;
  config_compute.are_outputs_on_device = true;
  config_compute.is_async = true;

  // separate streams for host-to-device and device-to-host transfers
  cudaStream_t stream_h2d, stream_d2h; 
  cudaStreamCreate(&stream_h2d);
  cudaStreamCreate(&stream_d2h);


  std::cout << "Full Duplex Interleaving In-place NTT" << std::endl;
  int nof_blocks = 32;
  std::cout << "Number of blocks: " << nof_blocks << std::endl;
  int block_size = ntt_size*nof_ntts/nof_blocks;
  
  // on-host pinned data
  E * h_inp[2];
  E * h_out[2];
  cudaHostAlloc((void**)&h_inp[0], sizeof(E)*ntt_size*nof_ntts, cudaHostAllocDefault);
  cudaHostAlloc((void**)&h_inp[1], sizeof(E)*ntt_size*nof_ntts, cudaHostAllocDefault);
  cudaHostAlloc((void**)&h_out[0], sizeof(E)*ntt_size*nof_ntts, cudaHostAllocDefault);
  cudaHostAlloc((void**)&h_out[1], sizeof(E)*ntt_size*nof_ntts, cudaHostAllocDefault);

  // on-device in-place data
  E * d_vec[2];
  cudaMalloc((void**)&d_vec[0], sizeof(E)*ntt_size*nof_ntts);
  cudaMalloc((void**)&d_vec[1], sizeof(E)*ntt_size*nof_ntts);

  // initialize input data
  initialize_input(ntt_size, nof_ntts, h_inp[0]);
  initialize_input(ntt_size, nof_ntts, h_inp[1]);

  

  // Perform Asynchronous Memory Transfers and Kernel Execution
  // [0] memory transfers
  // [1] kernel execution
  cudaEvent_t compute_start, compute_stop;
  cudaEventCreate(&compute_start);
  cudaEventCreate(&compute_stop);

  
  for ( int run = 0; run < 10; run++ ) {
    int buffer_compute = run % 2;
    int buffer_transfer = (run + 1) % 2;
    std::cout << "Run: " << run << std::endl;
    std::cout << "Buffer Compute: " << buffer_compute << std::endl;
    std::cout << "Buffer Transfer: " << buffer_transfer << std::endl;
    START_TIMER(inplace);
    cudaEventRecord(compute_start, stream_compute);
    bn254_ntt_cuda(d_vec[buffer_compute], ntt_size, NTTDir::kForward, config_compute, d_vec[buffer_compute]);
    cudaEventRecord(compute_stop, stream_compute);
    for (int i = 0; i <= nof_blocks; i++) {
      if (i < nof_blocks) {
        cudaMemcpyAsync(&h_out[buffer_transfer][i*block_size], &d_vec[buffer_transfer][i*block_size], sizeof(E)*block_size, cudaMemcpyDeviceToHost, stream_d2h);    
      }
      if (i>0) {
        cudaMemcpyAsync(&d_vec[buffer_transfer][(i-1)*block_size], &h_inp[buffer_transfer][(i-1)*block_size], sizeof(E)*block_size, cudaMemcpyHostToDevice, stream_h2d);
      }
      cudaStreamSynchronize(stream_d2h); 
      cudaStreamSynchronize(stream_h2d); 
    }
    // cudaStreamSynchronize(stream_compute);  
    cudaEventSynchronize(compute_stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, compute_start, compute_stop);
    END_TIMER(inplace, "Full Duplex Interleaved In-place  NTT");
    std::cout << "NTT time: " << milliseconds << " ms" << std::endl;
  };
  
  // Clean-up
  cudaFree(d_vec[0]); 
  cudaFree(d_vec[1]);
  cudaFreeHost(h_inp[0]); 
  cudaFreeHost(h_inp[1]);
  cudaFreeHost(h_out[0]); 
  cudaFreeHost(h_out[1]);
  cudaEventDestroy(compute_start);
  cudaEventDestroy(compute_stop);
  cudaStreamDestroy(stream_compute);
  cudaStreamDestroy(stream_d2h);
  cudaStreamDestroy(stream_h2d);
  return 0;
}
