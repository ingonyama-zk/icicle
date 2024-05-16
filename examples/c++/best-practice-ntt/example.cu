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

  std::cout << "Curve: " << curve << std::endl;

  S basic_root = S::omega(max_log_ntt_size);

  // change these parameters to match the desired NTT size and batch size  
  const unsigned log_ntt_size = 22;
  const unsigned nof_ntts = 16;

  std::cout << "log NTT size: " << log_ntt_size << std::endl;
  const unsigned ntt_size = 1 << log_ntt_size;

  std::cout << "Batch size: " << nof_ntts << std::endl;

  // Create separate CUDA streams for overlapping data transfers and kernel execution.
  cudaStream_t stream_compute, stream_h2d, stream_d2h;
  cudaStreamCreate(&stream_compute); 
  cudaStreamCreate(&stream_h2d);
  cudaStreamCreate(&stream_d2h);

  // Create device context for NTT computation
  auto ctx_compute = device_context::DeviceContext{
    stream_compute, // stream
    0,              // device_id
    0,              // mempool
  };

  // Initialize NTT domain and configuration
  bn254_initialize_domain(&basic_root, ctx_compute, /* fast twiddles */ true);
  NTTConfig<S> config_compute = default_ntt_config<S>(ctx_compute);
  config_compute.ntt_algorithm = NttAlgorithm::MixedRadix;
  config_compute.batch_size = nof_ntts;
  config_compute.are_inputs_on_device = true;
  config_compute.are_outputs_on_device = true;
  config_compute.is_async = true;
  
  std::cout << "Concurrent Download, Upload, and Compute In-place NTT" << std::endl;
  int nof_blocks = 32;
  std::cout << "Number of blocks: " << nof_blocks << std::endl;
  int block_size = ntt_size*nof_ntts/nof_blocks;
  
  // on-host pinned data
  E * h_inp[2];
  E * h_out[2];
  for (int i = 0; i < 2; i++) {
    cudaHostAlloc((void**)&h_inp[i], sizeof(E)*ntt_size*nof_ntts, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_out[i], sizeof(E)*ntt_size*nof_ntts, cudaHostAllocDefault);
  }
  
  // on-device in-place data
  // we need two on-device vectors to overlap data transfers with NTT kernel execution
  E * d_vec[2];
  for (int i = 0; i < 2; i++) {
    cudaMalloc((void**)&d_vec[i], sizeof(E)*ntt_size*nof_ntts);
  }
  
  // initialize input data
  initialize_input(ntt_size, nof_ntts, h_inp[0]);
  initialize_input(ntt_size, nof_ntts, h_inp[1]);

  cudaEvent_t compute_start, compute_stop;
  cudaEventCreate(&compute_start);
  cudaEventCreate(&compute_stop);

  for ( int run = 0; run < 10; run++ ) {  
    int vec_compute = run % 2;
    int vec_transfer = (run + 1) % 2;
    std::cout << "Run: " << run << std::endl;
    std::cout << "Compute Vector: " << vec_compute << std::endl;
    std::cout << "Transfer Vector: " << vec_transfer << std::endl;
    START_TIMER(inplace);
    cudaEventRecord(compute_start, stream_compute);
    bn254_ntt_cuda(d_vec[vec_compute], ntt_size, NTTDir::kForward, config_compute, d_vec[vec_compute]);
    cudaEventRecord(compute_stop, stream_compute);
    // we have to delay upload to device relative to download from device by one block: preserve write after read
    for (int i = 0; i <= nof_blocks; i++) {
      if (i < nof_blocks) {
        cudaMemcpyAsync(&h_out[vec_transfer][i*block_size], &d_vec[vec_transfer][i*block_size], sizeof(E)*block_size, cudaMemcpyDeviceToHost, stream_d2h);    
      }
      if (i>0) {
        cudaMemcpyAsync(&d_vec[vec_transfer][(i-1)*block_size], &h_inp[vec_transfer][(i-1)*block_size], sizeof(E)*block_size, cudaMemcpyHostToDevice, stream_h2d);
      }
      // synchronize upload and download at the end of the block to ensure data integrity
      cudaStreamSynchronize(stream_d2h); 
      cudaStreamSynchronize(stream_h2d); 
    }
    // synchronize compute stream with the end of the computation
    cudaEventSynchronize(compute_stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, compute_start, compute_stop);
    END_TIMER(inplace, "Concurrent In-Place  NTT");
    std::cout << "NTT time: " << milliseconds << " ms" << std::endl;
  };
  
  // Clean-up
  for (int i = 0; i < 2; i++) {
    cudaFree(d_vec[i]); 
    cudaFreeHost(h_inp[i]); 
    cudaFreeHost(h_out[i]); 
  }
  cudaEventDestroy(compute_start);
  cudaEventDestroy(compute_stop);
  cudaStreamDestroy(stream_compute);
  cudaStreamDestroy(stream_d2h);
  cudaStreamDestroy(stream_h2d);
  return 0;
}
