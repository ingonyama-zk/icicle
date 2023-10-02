#include "../../primitives/field.cuh"
#include "../../primitives/projective.cuh"
#include "../../utils/cuda_utils.cuh"
#include "msm.cu"
#include <chrono>
#include <iostream>
#include <vector>
#include "../../curves/bn254/curve_config.cuh"

using namespace BN254;

class Dummy_Scalar
{
public:
  static constexpr unsigned NBITS = 32;

  unsigned x;
  unsigned p = 10;

  static HOST_DEVICE_INLINE Dummy_Scalar zero() { return {0}; }

  static HOST_DEVICE_INLINE Dummy_Scalar one() { return {1}; }

  friend HOST_INLINE std::ostream& operator<<(std::ostream& os, const Dummy_Scalar& scalar)
  {
    os << scalar.x;
    return os;
  }

  HOST_DEVICE_INLINE unsigned get_scalar_digit(unsigned digit_num, unsigned digit_width)
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
  }
};

class Dummy_Projective
{
public:
  Dummy_Scalar x;

  static HOST_DEVICE_INLINE Dummy_Projective zero() { return {0}; }

  static HOST_DEVICE_INLINE Dummy_Projective one() { return {1}; }

  static HOST_DEVICE_INLINE Dummy_Projective to_affine(const Dummy_Projective& point) { return {point.x}; }

  static HOST_DEVICE_INLINE Dummy_Projective from_affine(const Dummy_Projective& point) { return {point.x}; }

  static HOST_DEVICE_INLINE Dummy_Projective neg(const Dummy_Projective& point) { return {Dummy_Scalar::neg(point.x)}; }

  friend HOST_DEVICE_INLINE Dummy_Projective operator+(Dummy_Projective p1, const Dummy_Projective& p2)
  {
    return {p1.x + p2.x};
  }

  friend HOST_INLINE std::ostream& operator<<(std::ostream& os, const Dummy_Projective& point)
  {
    os << point.x;
    return os;
  }

  friend HOST_DEVICE_INLINE Dummy_Projective operator*(Dummy_Scalar scalar, const Dummy_Projective& point)
  {
    Dummy_Projective res = zero();
#ifdef CUDA_ARCH
#pragma unroll
#endif
    for (int i = 0; i < Dummy_Scalar::NBITS; i++) {
      if (i > 0) { res = res + res; }
      if (scalar.get_scalar_digit(Dummy_Scalar::NBITS - i - 1, 1)) { res = res + point; }
    }
    return res;
  }

  friend HOST_DEVICE_INLINE bool operator==(const Dummy_Projective& p1, const Dummy_Projective& p2)
  {
    return (p1.x == p2.x);
  }

  static HOST_DEVICE_INLINE bool is_zero(const Dummy_Projective& point) { return point.x == 0; }

  static HOST_INLINE Dummy_Projective rand_host()
  {
    return {(unsigned)rand() % 10};
  }
};

typedef scalar_t test_scalar;
typedef projective_t test_projective;
typedef affine_t test_affine;

int main()
{
  unsigned batch_size = 1;
  unsigned msm_size = 65535+1;
  unsigned N = batch_size * msm_size;

  test_scalar* scalars = new test_scalar[N];
  test_affine* points = new test_affine[N];

  for (unsigned i = 0; i < N; i++) {
    points[i] = (i % msm_size < 10) ? test_projective::to_affine(test_projective::rand_host()) : points[i - 10];
    scalars[i] = test_scalar::rand_host();
  }
  std::cout << "finished generating" << std::endl;

  test_projective large_res[batch_size * 2];
  test_scalar* scalars_d;
  test_affine* points_d;
  test_projective* large_res_d;

  cudaMalloc(&scalars_d, sizeof(test_scalar) * msm_size);
  cudaMalloc(&points_d, sizeof(test_affine) * msm_size);
  cudaMalloc(&large_res_d, sizeof(test_projective));
  cudaMemcpy(scalars_d, scalars, sizeof(test_scalar) * msm_size, cudaMemcpyHostToDevice);
  cudaMemcpy(points_d, points, sizeof(test_affine) * msm_size, cudaMemcpyHostToDevice);

  std::cout << "finished copying" << std::endl;

  cudaStream_t stream1;
  cudaStream_t stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  auto begin1 = std::chrono::high_resolution_clock::now();
  large_msm<test_scalar, test_projective, test_affine>(scalars, points, msm_size, large_res, false, true, 1, stream1);
  auto end1 = std::chrono::high_resolution_clock::now();
  auto elapsed1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - begin1);
  printf("Big Triangle : %.3f seconds.\n", elapsed1.count() * 1e-9);
  auto begin = std::chrono::high_resolution_clock::now();
  large_msm<test_scalar, test_projective, test_affine>(
    scalars_d, points_d, msm_size, large_res_d, true, false, 1, stream2);
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
  printf("On Device No Big Triangle: %.3f seconds.\n", elapsed.count() * 1e-9);
  cudaStreamSynchronize(stream1);
  cudaStreamSynchronize(stream2);
  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);

  std::cout << test_projective::to_affine(large_res[0]) << std::endl;

  cudaMemcpy(&large_res[1], large_res_d, sizeof(test_projective), cudaMemcpyDeviceToHost);
  std::cout << test_projective::to_affine(large_res[1]) << std::endl;

  return 0;
}