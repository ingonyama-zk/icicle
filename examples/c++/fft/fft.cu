#include <iostream>
#include <iomanip>
#include <chrono>
#include <nvml.h>
#include <vector>

#define CURVE_ID 1
#include "curves/curve_config.cuh"
#include "utils/device_context.cuh"
#include "utils/vec_ops.cu"

using namespace curve_config;

typedef scalar_t T;

void print(std::string tag, T* b, int n) {
  std::cout << "=================" << std::endl;
  std::cout << "Printing " << tag << std::endl;

  for (int i = 0; i < n; i++) {
    std::cout << b[i] << " ";
  }
  std::cout << std::endl << "=================" << std::endl;
}

uint32_t reverse_bits(uint32_t x)
{
    x = ((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1);
    x = ((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2);
    x = ((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4);
    x = ((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8);
    return (x >> 16) | (x << 16);
}

void old_reverse_bit(T* b, uint n) {
  for (int i = 1, j = 0; i < n; i++)
  {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1)
      j ^= bit;
    j ^= bit;

    if (i < j) {
      // swap(b[i], b[j]);
      T tmp = b[i];
      b[i] = b[j];
      b[j] = tmp;
    }
  }
}

T* precompute_w(uint n, T root, T root_inv, uint root_pw, bool invert) {
  T* ws = (T*)malloc((n - 1) * sizeof(T));
  uint index = 0;

  for (int len = 2; len <= n; len <<= 1)
  {
    T wlen = invert ? root_inv : root;
    for (int i = len; i < root_pw; i <<= 1)
      wlen = wlen * wlen;

    T w = T::from(1);
    for (int j = 0; j < len / 2; j++)
    {
      ws[index++] = w;
      w = w * wlen;
    }
  }

  std::cout << "precompute_w index = " << index << std::endl;

  return ws;
}

void fft_cpu(T* b, uint n, T root, T root_inv, uint root_pw, T* ws, T* ws_inv, bool invert) {
  const int log_n = log2(n);
  for (int i = 0; i < n; i++) {
    uint rev = reverse_bits(i);
    rev = rev >> (32 - log_n);

    if (i < rev) {
      // std::cout << "Swapping " << i << " " << rev << std::endl;
      T tmp = b[i];
      b[i] = b[rev];
      b[rev] = tmp;
    }
  }
  // print("CPU value b after bit reverse", b, n);

  int ws_index = 0;
  for (int len = 2; len <= n; len <<= 1)
  {
    for (int i = 0; i < n; i += len)
    {
      for (int j = 0; j < len / 2; j++)
      {
        T w;
        if (!invert) {
          w = ws[ws_index + j];
        } else {
          w = ws_inv[ws_index + j];
        }

        T u = b[i + j];
        T v = b[i + j + len / 2] * w;
        b[i + j] = u + v;
        b[i + j + len / 2] = u - v;
      }
    }

    ws_index += len / 2;
  }

  if (invert) {
    T inv_n = T::inverse(T::from(n));
    for (int i = 0; i < n; i++) {
      b[i] = b[i] * inv_n;
    }
  }

  // print(b, n);
}

////////////////////////////////////////////////////////////////////////
__device__ uint32_t device_reverse_bits(uint32_t x)
{
    x = ((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1);
    x = ((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2);
    x = ((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4);
    x = ((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8);
    return (x >> 16) | (x << 16);
}

__global__ void swap_bits(T* b, uint n, uint log_n) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  for (int i = tid * 2; i < tid * 2 + 2; i++) {
    uint rev = device_reverse_bits(i);
    rev = rev >> (32 - log_n);

    if (i < rev) {
      T tmp = b[i];
      b[i] = b[rev];
      b[rev] = tmp;
    }
  }
}

__global__ void fft_kernel(T* b, uint n, T inv_n, uint pow, uint ws_index, T* ws, T* ws_inv, bool invert) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    int len = 1 << pow;
    int len2 = len >> 1; // len2 = len / 2
    int q = tid >> (pow - 1);
    int i = q * len;
    int j = tid - q * len2;

    T w;
    if (!invert) {
      w = ws[ws_index + j];
    } else {
      w = ws_inv[ws_index + j];
    }

    T u = b[i + j];
    T v = b[i + j + len / 2] * w;
    b[i + j] = u + v;
    b[i + j + len / 2] = u - v;
}

__global__ void invert_result(T* b, T inv_n) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  auto x = tid << 1;
  b[x] = b[x] * inv_n;
  b[x + 1] = b[x + 1] * inv_n;
}

T* fft_gpu(T* host_b, uint n, T* device_ws, T* device_ws_inv, bool invert) {
  cudaError_t err;
  T* device_b;

  T inv_n = T::inverse(T::from(n));

  auto start_time = std::chrono::high_resolution_clock::now();

  // allocate device array
  cudaMalloc((void**)&device_b, n * sizeof(T));

  // copy from host to device
  err = cudaMemcpy(device_b, host_b, n * sizeof(T), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "Failed to copy data from host to device - " << cudaGetErrorString(err) << std::endl;
    return NULL;
  }

  int cuda_device_ix = 0;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, cuda_device_ix);

  // Set the grid and block dimensions
  int worker_count = n >> 1;
  int num_threads = worker_count < prop.maxThreadsPerBlock ? worker_count : prop.maxThreadsPerBlock;
  int num_blocks = (worker_count + num_threads - 1) / num_threads;

  printf("GPU num_blocks = %d, num_threads = %d\n", num_blocks, num_threads);

  const int log_n = log2(n);
  // Swap bits
  swap_bits<<< num_blocks, num_threads  >>> (device_b, n, log_n);

  std::cout << "inv_n = " << inv_n << std::endl;

  // main loop
  int ws_index = 0;
  for (int pow = 1; ; pow++) {
    int len = 1 << pow;
    if (len > n) {
      break;
    }

    fft_kernel<<< num_blocks, num_threads  >>> (device_b, n, inv_n, pow, ws_index, device_ws, device_ws_inv, invert);

    ws_index += len >> 1;
  }

  // If this is interpolatio, invert the result
  if (invert) {
    invert_result<<< num_blocks, num_threads  >>> (device_b, inv_n);
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  std::cout << "GPU Kernel running time: " << duration.count() << " microseconds" << std::endl;

  T* host_result = (T*)malloc(n * sizeof(T));
  err = cudaMemcpy(host_result, device_b, n * sizeof(T), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    std::cerr << "Failed to copy data from device to host - " << cudaGetErrorString(err) << std::endl;
    return NULL;
  }

  return host_result;
}

T* run_gpu(std::vector<int> a) {
  cudaError_t err;

  const int log_n = log2(a.size());
  const int root_pw = 1 << log_n;

  int n = a.size();
  T* a_field = (T*)malloc(n * sizeof(T));
  for (int i = 0; i < a.size(); i++) {
    a_field[i] = T::from(a[i]);
  }

  T root = T::omega(log_n);
  T root_inv = T::inverse(root);

  std::cout << "root = " << root << std::endl;
  std::cout << "root_inv = " << root_inv << std::endl;

  T* ws = precompute_w(n, root, root_inv, root_pw, false);
  T* ws_inv = precompute_w(n, root, root_inv, root_pw, true);

  T* device_ws;
  T* device_ws_inv;
  cudaMalloc((void**)&device_ws, n * sizeof(T));
  cudaMalloc((void**)&device_ws_inv, n * sizeof(T));
  // copy from host to device
  err = cudaMemcpy(device_ws, ws, n * sizeof(T), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "Failed to copy data from host to device - " << cudaGetErrorString(err) << std::endl;
    return NULL;
  }
  err = cudaMemcpy(device_ws_inv, ws_inv, n * sizeof(T), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "Failed to copy data from host to device - " << cudaGetErrorString(err) << std::endl;
    return NULL;
  }

  // function call
  auto result = fft_gpu(a_field, n, device_ws, device_ws_inv, false);

  for (int i = 0; i < 8; i++) {
    std::cout << result[i] << std::endl;
  }

  return result;

  // return fft_gpu(result, n, device_ws, device_ws_inv, true);
}

T* run_cpu(std::vector<int> a) {
  const int log_n = log2(a.size());
  const int root_pw = 1 << log_n;

  int n = a.size();
  T* a_field = (T*)malloc(n * sizeof(T));
  for (int i = 0; i < a.size(); i++) {
    a_field[i] = T::from(a[i]);
  }

  std::cout << "n = " << n << " log_n = " << log_n << std::endl;

  T root = T::omega(log_n);
  T root_inv = T::inverse(root);

  std::cout << "root = " << root << std::endl;
  std::cout << "root_inv = " << root_inv << std::endl;

  // (uint n, T root, T root_inv, uint root_pw, bool invert)
  T* ws = precompute_w(n, root, root_inv, root_pw, false);
  T* ws_inv = precompute_w(n, root, root_inv, root_pw, true);

  std::cout << "Done precompute" << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  // Evaluate
  fft_cpu(a_field, n, root, root_inv, root_pw, ws, ws_inv, false);
  // print("CPU Evaluation", a_field, n);

  auto end1 = std::chrono::high_resolution_clock::now();
  auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start);
  std::cout << "CPU Duration 1 = " << duration1.count() << std::endl;

  // Interpolate
  fft_cpu(a_field, n, root, root_inv, root_pw, ws, ws_inv, true);

  auto end2 = std::chrono::high_resolution_clock::now();
  auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - end1);
  std::cout << "CPU Duration 2 = " << duration2.count() << std::endl;

  return a_field;
}

std::vector<int> gen_data() {
  std::vector<int> a = {3, 1, 4, 1, 5, 9, 2, 6};
  // std::vector<int> a;
  // for (int i = 0; i < 1 << 4; i++) {
  //   int random = rand() % 1000;
  //   a.push_back(random);
  // }

  for (int i = 0; i < 8; i++) {
    std::cout << a[i] << std::endl;
  }
  std::cout << "=========" << std::endl;

  return a;
}

int main(int argc, char** argv) {
  auto a = gen_data();

  // auto cpu_result = run_cpu(a);
  auto gpu_result = run_gpu(a);

  // for (int i = 0; i < a.size(); i++) {
  //   if (cpu_result[i] != gpu_result[i]) {
  //     std::cout << "Test fails at " << i << " cpu = " << cpu_result[i] << " gpu = " << gpu_result[i];
  //     std::cout << " original value = " << a[i] << std::endl;
  //     return 1;
  //   }
  // }

  // std::cout << "Test passed!" << std::endl;

  return 0;
}
