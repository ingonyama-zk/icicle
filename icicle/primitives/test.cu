#include <cuda_runtime.h>
#include <gtest/gtest.h>

// TODO: change the curve depending on env variable
#include "../curves/bls12_381.cuh"
#include "projective.cuh"
#include "field.cuh"


typedef Field<fp_config> scalar_field;
typedef Field<fq_config> base_field;
typedef Projective<base_field, scalar_field, group_generator, weierstrass_b> proj;

template <class T>
int device_populate_random(T* d_elements, unsigned n) {
    T h_elements[n];
    for (unsigned i = 0; i < n; i++)
        h_elements[i] = T::rand_host();
    return cudaMemcpy(d_elements, h_elements, sizeof(T) * n, cudaMemcpyHostToDevice);
}

template <class T>
int device_set(T* d_elements, T el, unsigned n) {
    T h_elements[n];
    for (unsigned i = 0; i < n; i++)
        h_elements[i] = el;
    return cudaMemcpy(d_elements, h_elements, sizeof(T) * n, cudaMemcpyHostToDevice);
}

class PrimitivesTest : public ::testing::Test {
protected:
  static const unsigned n = 1 << 5;

  proj *points1{};
  proj *points2{};
  proj *zeroes{};
  proj *res1{};
  proj *res2{};

  PrimitivesTest() {
    assert(!cudaDeviceReset());
    assert(!cudaMallocManaged(&points1, n * sizeof(proj)));
    assert(!cudaMallocManaged(&points2, n * sizeof(proj)));
    assert(!cudaMallocManaged(&zeroes, n * sizeof(proj)));
    assert(!cudaMallocManaged(&res1, n * sizeof(proj)));
    assert(!cudaMallocManaged(&res2, n * sizeof(proj)));
  }

  ~PrimitivesTest() override {
    cudaFree(points1);
    cudaFree(points2);
    cudaFree(zeroes);
    cudaFree(res1);
    cudaFree(res2);
    cudaDeviceReset();
  }

  void SetUp() override {
    ASSERT_EQ(device_populate_random<proj>(points1, n), cudaSuccess);
    ASSERT_EQ(device_populate_random<proj>(points2, n), cudaSuccess);
    ASSERT_EQ(device_set<proj>(zeroes, proj::zero(), n), cudaSuccess);
    ASSERT_EQ(cudaMemset(res1, 0, n * sizeof(proj)), cudaSuccess);
    ASSERT_EQ(cudaMemset(res2, 0, n * sizeof(proj)), cudaSuccess);
  }
};

TEST_F(PrimitivesTest, RandomPointsAreOnCurve) {
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED1(proj::is_on_curve, points1[i]);
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
