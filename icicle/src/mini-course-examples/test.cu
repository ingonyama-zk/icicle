#include "fields/id.h"
// #define FIELD_ID 2
#define CURVE_ID 3
#include "curves/curve_config.cuh"
// #include "fields/field_config.cuh"

#include <chrono>
#include <iostream>
#include <vector>

#include "fields/field.cuh"
#include "curves/projective.cuh"
#include "gpu-utils/device_context.cuh"

#include "kernels.cu"

class Dummy_Scalar
{
public:
  static constexpr unsigned NBITS = 32;

  unsigned x;
  unsigned p = 10;
  // unsigned p = 1<<30;

  static HOST_DEVICE_INLINE Dummy_Scalar zero() { return {0}; }

  static HOST_DEVICE_INLINE Dummy_Scalar one() { return {1}; }

  friend HOST_INLINE std::ostream& operator<<(std::ostream& os, const Dummy_Scalar& scalar)
  {
    os << scalar.x;
    return os;
  }

  HOST_DEVICE_INLINE unsigned get_scalar_digit(unsigned digit_num, unsigned digit_width) const
  {
    return (x >> (digit_num * digit_width)) & ((1 << digit_width) - 1);
  }

  friend HOST_DEVICE_INLINE Dummy_Scalar operator+(Dummy_Scalar p1, const Dummy_Scalar& p2)
  {
    return {(p1.x + p2.x) % p1.p};
  }

  friend HOST_DEVICE_INLINE bool operator==(const Dummy_Scalar& p1, const Dummy_Scalar& p2) { return (p1.x == p2.x); }

  friend HOST_DEVICE_INLINE bool operator==(const Dummy_Scalar& p1, const unsigned p2) { return (p1.x == p2); }

  static HOST_DEVICE_INLINE Dummy_Scalar neg(const Dummy_Scalar& scalar) { return {scalar.p - scalar.x}; }
  static HOST_INLINE Dummy_Scalar rand_host()
  {
    return {(unsigned)rand() % 10};
    // return {(unsigned)rand()};
  }
};


typedef curve_config::scalar_t test_scalar;
typedef curve_config::projective_t test_projective;
typedef curve_config::affine_t test_affine;

// typedef Dummy_Scalar test_t;
// typedef test_projective test_t;
typedef test_scalar test_t;

void queryGPUProperties() {
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount returned " << static_cast<int>(error_id) << " -> " << cudaGetErrorString(error_id) << std::endl;
        std::cerr << "Result = FAIL" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (deviceCount == 0) {
        std::cout << "There are no available device(s) that support CUDA." << std::endl;
    } else {
        std::cout << "Detected " << deviceCount << " CUDA Capable device(s)." << std::endl;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaSetDevice(dev);

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout << "Device " << dev << ": \"" << deviceProp.name << "\"" << std::endl;
        std::cout << "  CUDA Capability Major/Minor version number: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Total amount of global memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Number of multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "  Total amount of global memory: " << deviceProp.totalGlobalMem << " bytes" << std::endl;
        std::cout << "  Total amount of shared memory per block: " << deviceProp.sharedMemPerBlock << " bytes" << std::endl;
        std::cout << "  Total amount of shared memory per multiprocessor: " << deviceProp.sharedMemPerMultiprocessor << " bytes" << std::endl;
        std::cout << "  Total number of registers available per block: " << deviceProp.regsPerBlock << std::endl;
        std::cout << "  Total number of registers available per multiprocessor: " << deviceProp.regsPerMultiprocessor << std::endl;
        std::cout << "  Warp size: " << deviceProp.warpSize << std::endl;
        std::cout << "  Maximum number of threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Maximum number of threads per multiprocessor: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "  Maximum sizes of each dimension of a block: " << deviceProp.maxThreadsDim[0] << " x " 
                  << deviceProp.maxThreadsDim[1] << " x " << deviceProp.maxThreadsDim[2] << std::endl;
        std::cout << "  Maximum sizes of each dimension of a grid: " << deviceProp.maxGridSize[0] << " x " 
                  << deviceProp.maxGridSize[1] << " x " << deviceProp.maxGridSize[2] << std::endl;
        std::cout << "  Clock rate: " << deviceProp.clockRate / 1000 << " MHz" << std::endl;
        std::cout << "  Memory clock rate: " << deviceProp.memoryClockRate / 1000 << " MHz" << std::endl;
        std::cout << "  Memory bus width: " << deviceProp.memoryBusWidth << " bits" << std::endl;
        std::cout << "  Peak memory bandwidth: " 
                  << 2.0 * deviceProp.memoryClockRate * (deviceProp.memoryBusWidth / 8) / 1.0e6 << " GB/s" << std::endl;
    }
}

int main()
{

  queryGPUProperties();

  int N = 1<<20;
  // int N = 300;

  test_t* A_h = new test_t[N];
  test_t* B_h = new test_t[N];
  test_t* C_h = new test_t[N];
  test_t* Cb_h = new test_t[N];

  for (int i = 0; i < N; i++)
  {
    A_h[i] = i<100? test_t::rand_host() : A_h[i-100];
    B_h[i] = i<100? test_t::rand_host() : B_h[i-100];
  }
  
  test_t *A_d,*B_d,*C_d;
  test_t *Cb_d;


  cudaMalloc(&A_d, sizeof(test_t) * N);
  cudaMalloc(&B_d, sizeof(test_t) * N);
  cudaMalloc(&C_d, sizeof(test_t) * N);
  cudaMalloc(&Cb_d, sizeof(test_t) * N);
  
  cudaMemcpy(A_d, A_h, sizeof(test_t) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, sizeof(test_t) * N, cudaMemcpyHostToDevice);

  // int THREADS = 256;
  // int BLOCKS = (N + THREADS - 1)/THREADS;
  // add_elements_kernel<<<BLOCKS, THREADS>>>(A_d, B_d, C_d, N);
  // cudaDeviceSynchronize();
  // // printf("cuda error %d\n", cudaGetLastError());
  // std::cout << "cuda err: " << cudaGetErrorString(cudaGetLastError()) << std::endl;

  // THREADS = 256;
  // BLOCKS = (N + THREADS - 1)/THREADS;
  // bugged_add_elements_kernel<<<BLOCKS, THREADS>>>(A_d, B_d, Cb_d, N);
  // cudaDeviceSynchronize();
  // // printf("cuda error %d\n", cudaGetLastError());
  // std::cout << "cuda err: " << cudaGetErrorString(cudaGetLastError()) << std::endl;

  // int THREADS = 128;
  // int BLOCKS = (N/4 + THREADS - 1)/THREADS;
  // // fake_ntt_kernel<<<BLOCKS, THREADS, sizeof(test_t)*THREADS>>>(A_d, C_d, N);
  // fake_ntt_kernel<<<BLOCKS, THREADS, sizeof(test_t)*THREADS*4>>>(A_d, C_d, N/4);
  // cudaDeviceSynchronize();
  // // printf("cuda error %d\n", cudaGetLastError());
  // std::cout << "cuda err: " << cudaGetErrorString(cudaGetLastError()) << std::endl;

  // THREADS = 128;
  // BLOCKS = (N/4 + THREADS - 1)/THREADS;
  // // fake_ntt_kernel<<<BLOCKS, THREADS, sizeof(test_t)*THREADS>>>(A_d, C_d, N);
  // bugged_fake_ntt_kernel<<<BLOCKS, THREADS, sizeof(test_t)*THREADS*4>>>(A_d, Cb_d, N/4);
  // // bugged_fake_ntt_kernel<<<1, 1, sizeof(test_t)*THREADS*4>>>(A_d, Cb_d, N/4);
  // cudaDeviceSynchronize();
  // // printf("cuda error %d\n", cudaGetLastError());
  // std::cout << "cuda err: " << cudaGetErrorString(cudaGetLastError()) << std::endl;

  cudaMemcpy(C_h, C_d, sizeof(test_t) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(Cb_h, Cb_d, sizeof(test_t) * N, cudaMemcpyDeviceToHost);

  // printf("A: ");
  // for (size_t i = 0; i < 8; i++)
  // {
  //   std::cout << A_h[i] << ", ";
  // }
  // printf("\n");
  // printf("C test: ");
  // for (size_t i = 0; i < 8; i++)
  // {
  //   std::cout << Cb_h[i] << ", ";
  // }
  // printf("\n");
  // printf("C ref: ");
  // for (size_t i = 0; i < 8; i++)
  // {
  //   std::cout << C_d[i] << ", ";
  //   // std::cout << C_h[i] << ", ";
  // }
  // printf("\n");

  return 0;
}