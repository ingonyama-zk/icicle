#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include "test_kernels.cuh"
#include <iostream>

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
  scalar_field *scalars1{};
  scalar_field *scalars2{};
  proj *zero_points{};
  scalar_field *zero_scalars{};
  scalar_field *one_scalars{};
  affine *aff_points{};
  proj *res_points1{};
  proj *res_points2{};
  scalar_field *res_scalars1{};
  scalar_field *res_scalars2{};
  scalar_field::wide *res_scalars_wide{};
  scalar_field::wide *res_scalars_wide_full{};

  PrimitivesTest() {
    assert(!cudaDeviceReset());
    assert(!cudaMallocManaged(&points1, n * sizeof(proj)));
    assert(!cudaMallocManaged(&points2, n * sizeof(proj)));
    assert(!cudaMallocManaged(&scalars1, n * sizeof(scalar_field)));
    assert(!cudaMallocManaged(&scalars2, n * sizeof(scalar_field)));
    assert(!cudaMallocManaged(&zero_points, n * sizeof(proj)));
    assert(!cudaMallocManaged(&zero_scalars, n * sizeof(scalar_field)));
    assert(!cudaMallocManaged(&one_scalars, n * sizeof(scalar_field)));
    assert(!cudaMallocManaged(&aff_points, n * sizeof(affine)));
    assert(!cudaMallocManaged(&res_points1, n * sizeof(proj)));
    assert(!cudaMallocManaged(&res_points2, n * sizeof(proj)));
    assert(!cudaMallocManaged(&res_scalars1, n * sizeof(scalar_field)));
    assert(!cudaMallocManaged(&res_scalars2, n * sizeof(scalar_field)));
    assert(!cudaMallocManaged(&res_scalars_wide, n * sizeof(scalar_field::wide)));
    assert(!cudaMallocManaged(&res_scalars_wide_full, n * sizeof(scalar_field::wide)));

  }

  ~PrimitivesTest() override {
    cudaFree(points1);
    cudaFree(points2);
    cudaFree(scalars1);
    cudaFree(scalars2);
    cudaFree(zero_points);
    cudaFree(zero_scalars);
    cudaFree(one_scalars);
    cudaFree(aff_points);
    cudaFree(res_points1);
    cudaFree(res_points2);
    cudaFree(res_scalars1);
    cudaFree(res_scalars2);

    cudaFree(res_scalars_wide);
    cudaFree(res_scalars_wide_full);

    cudaDeviceReset();
  }

  void SetUp() override {
    ASSERT_EQ(device_populate_random<proj>(points1, n), cudaSuccess);
    ASSERT_EQ(device_populate_random<proj>(points2, n), cudaSuccess);
    ASSERT_EQ(device_populate_random<scalar_field>(scalars1, n), cudaSuccess);
    ASSERT_EQ(device_populate_random<scalar_field>(scalars2, n), cudaSuccess);
    ASSERT_EQ(device_set<proj>(zero_points, proj::zero(), n), cudaSuccess);
    ASSERT_EQ(device_set<scalar_field>(zero_scalars, scalar_field::zero(), n), cudaSuccess);
    ASSERT_EQ(device_set<scalar_field>(one_scalars, scalar_field::one(), n), cudaSuccess);
    ASSERT_EQ(cudaMemset(aff_points, 0, n * sizeof(affine)), cudaSuccess);
    ASSERT_EQ(cudaMemset(res_points1, 0, n * sizeof(proj)), cudaSuccess);
    ASSERT_EQ(cudaMemset(res_points2, 0, n * sizeof(proj)), cudaSuccess);
    ASSERT_EQ(cudaMemset(res_scalars1, 0, n * sizeof(scalar_field)), cudaSuccess);
    ASSERT_EQ(cudaMemset(res_scalars2, 0, n * sizeof(scalar_field)), cudaSuccess);
    
    ASSERT_EQ(cudaMemset(res_scalars_wide, 0, n * sizeof(scalar_field::wide)), cudaSuccess);
    ASSERT_EQ(cudaMemset(res_scalars_wide_full, 0, n * sizeof(scalar_field::wide)), cudaSuccess);
  }
};

TEST_F(PrimitivesTest, FieldAdditionSubtractionCancel) {
  ASSERT_EQ(vec_add(scalars1, scalars2, res_scalars1, n), cudaSuccess);
  ASSERT_EQ(vec_sub(res_scalars1, scalars2, res_scalars2, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(scalars1[i], res_scalars2[i]);
}

TEST_F(PrimitivesTest, FieldZeroAddition) {
  ASSERT_EQ(vec_add(scalars1, zero_scalars, res_scalars1, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(scalars1[i], res_scalars1[i]);
}

TEST_F(PrimitivesTest, FieldAdditionHostDeviceEq) {
  ASSERT_EQ(vec_add(scalars1, scalars2, res_scalars1, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(scalars1[i] + scalars2[i], res_scalars1[i]);
}

TEST_F(PrimitivesTest, FieldMultiplicationByOne) {
  ASSERT_EQ(vec_mul(scalars1, one_scalars, res_scalars1, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(scalars1[i], res_scalars1[i]);
}

TEST_F(PrimitivesTest, FieldMultiplicationByMinusOne) {
  ASSERT_EQ(vec_neg(one_scalars, res_scalars1, n), cudaSuccess);
  ASSERT_EQ(vec_mul(scalars1, res_scalars1, res_scalars2, n), cudaSuccess);
  ASSERT_EQ(vec_add(scalars1, res_scalars2, res_scalars1, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(res_scalars1[i], zero_scalars[i]);
}

TEST_F(PrimitivesTest, FieldMultiplicationByZero) {
  ASSERT_EQ(vec_mul(scalars1, zero_scalars, res_scalars1, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(zero_scalars[i], res_scalars1[i]);
}

TEST_F(PrimitivesTest, FieldMultiplicationInverseCancel) {
  ASSERT_EQ(vec_mul(scalars1, scalars2, res_scalars1, n), cudaSuccess);
  ASSERT_EQ(field_vec_inv(scalars2, res_scalars2, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(scalars1[i], res_scalars1[i] * res_scalars2[i]);
}

TEST_F(PrimitivesTest, FieldMultiplicationHostDeviceEq) {
  ASSERT_EQ(vec_mul(scalars1, scalars2, res_scalars1, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(scalars1[i] * scalars2[i], res_scalars1[i]);
}

TEST_F(PrimitivesTest, FieldMultiplicationByTwoEqSum) {
  ASSERT_EQ(vec_add(one_scalars, one_scalars, res_scalars1, n), cudaSuccess);
  ASSERT_EQ(vec_mul(res_scalars1, scalars1, res_scalars2, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(res_scalars2[i], scalars1[i] + scalars1[i]);
}

TEST_F(PrimitivesTest, FieldSqrHostDeviceEq) {
  ASSERT_EQ(field_vec_sqr(scalars1, res_scalars1, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(scalars1[i] * scalars1[i], res_scalars1[i]);
}

TEST_F(PrimitivesTest, FieldMultiplicationSqrEq) {
  ASSERT_EQ(vec_mul(scalars1, scalars1, res_scalars1, n), cudaSuccess);
  ASSERT_EQ(field_vec_sqr(scalars1, res_scalars2, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(res_scalars1[i], res_scalars2[i]);
}

TEST_F(PrimitivesTest, ECRandomPointsAreOnCurve) {
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED1(proj::is_on_curve, points1[i]);
}

TEST_F(PrimitivesTest, ECPointAdditionSubtractionCancel) {
  ASSERT_EQ(vec_add(points1, points2, res_points1, n), cudaSuccess);
  ASSERT_EQ(vec_sub(res_points1, points2, res_points2, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(points1[i], res_points2[i]);
}

TEST_F(PrimitivesTest, ECPointZeroAddition) {
  ASSERT_EQ(vec_add(points1, zero_points, res_points1, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(points1[i], res_points1[i]);
}

TEST_F(PrimitivesTest, ECPointAdditionHostDeviceEq) {
  ASSERT_EQ(vec_add(points1, points2, res_points1, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(points1[i] + points2[i], res_points1[i]);
}

TEST_F(PrimitivesTest, ECScalarMultiplicationHostDeviceEq) {
  ASSERT_EQ(vec_mul(scalars1, points1, res_points1, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(scalars1[i] * points1[i], res_points1[i]);
}

TEST_F(PrimitivesTest, ECScalarMultiplicationByOne) {
  ASSERT_EQ(vec_mul(one_scalars, points1, res_points1, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(points1[i], res_points1[i]);
}

TEST_F(PrimitivesTest, ECScalarMultiplicationByMinusOne) {
  ASSERT_EQ(vec_neg(one_scalars, res_scalars1, n), cudaSuccess);
  ASSERT_EQ(vec_mul(res_scalars1, points1, res_points1, n), cudaSuccess);
  ASSERT_EQ(vec_neg(points1, res_points2, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(res_points1[i], res_points2[i]);
}

TEST_F(PrimitivesTest, ECScalarMultiplicationByTwo) {
  ASSERT_EQ(vec_add(one_scalars, one_scalars, res_scalars1, n), cudaSuccess);
  ASSERT_EQ(vec_mul(res_scalars1, points1, res_points1, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ((one_scalars[i] + one_scalars[i]) * points1[i], res_points1[i]);
}

TEST_F(PrimitivesTest, ECScalarMultiplicationInverseCancel) {
  ASSERT_EQ(vec_mul(scalars1, points1, res_points1, n), cudaSuccess);
  ASSERT_EQ(field_vec_inv(scalars1, res_scalars1, n), cudaSuccess);
  ASSERT_EQ(vec_mul(res_scalars1, res_points1, res_points2, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(points1[i], res_points2[i]);
}

TEST_F(PrimitivesTest, ECScalarMultiplicationIsDistributiveOverMultiplication) {
  ASSERT_EQ(vec_mul(scalars1, points1, res_points1, n), cudaSuccess);
  ASSERT_EQ(vec_mul(scalars2, res_points1, res_points2, n), cudaSuccess);
  ASSERT_EQ(vec_mul(scalars1, scalars2, res_scalars1, n), cudaSuccess);
  ASSERT_EQ(vec_mul(res_scalars1, points1, res_points1, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(res_points1[i], res_points2[i]);
}

TEST_F(PrimitivesTest, ECScalarMultiplicationIsDistributiveOverAddition) {
  ASSERT_EQ(vec_mul(scalars1, points1, res_points1, n), cudaSuccess);
  ASSERT_EQ(vec_mul(scalars2, points1, res_points2, n), cudaSuccess);
  ASSERT_EQ(vec_add(scalars1, scalars2, res_scalars1, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(res_scalars1[i] * points1[i], res_points1[i] + res_points2[i]);
}

TEST_F(PrimitivesTest, ECProjectiveToAffine) {
  ASSERT_EQ(point_vec_to_affine(points1, aff_points, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(points1[i], proj::from_affine(aff_points[i]));
}

TEST_F(PrimitivesTest, ECMixedPointAddition) {
  ASSERT_EQ(point_vec_to_affine(points2, aff_points, n), cudaSuccess);
  ASSERT_EQ(vec_add(points1, aff_points, res_points1, n), cudaSuccess);
  ASSERT_EQ(vec_add(points1, points2, res_points2, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(res_points1[i], res_points2[i]);
}

TEST_F(PrimitivesTest, ECMixedAdditionOfNegatedPointEqSubtraction) {
  ASSERT_EQ(point_vec_to_affine(points2, aff_points, n), cudaSuccess);
  ASSERT_EQ(vec_sub(points1, aff_points, res_points1, n), cudaSuccess);
  ASSERT_EQ(vec_neg(points2, res_points2, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(res_points1[i], points1[i] + res_points2[i]);
}

TEST_F(PrimitivesTest, MP_LSB_MULT) {


  ASSERT_EQ(mp_lsb_mult(scalars1, scalars2, res_scalars_wide), cudaSuccess);
  std::cout << "GPU lsb mult output  = 0x";
  for (int i=0; i<2*scalar_field::TLC; i++)
  {
    std::cout << std::hex << res_scalars_wide[0].limbs_storage.limbs[i];
  }
  std::cout << std::endl;


  ASSERT_EQ(mp_mult(scalars1, scalars2, res_scalars_wide_full), cudaSuccess);
  std::cout << "GPU full mult output = 0x";
  for (int i=0; i<2*scalar_field::TLC; i++)
  {
    std::cout << std::hex << res_scalars_wide_full[0].limbs_storage.limbs[i];
  }
  std::cout << std::endl;

  for (int i=0; i<2*scalar_field::TLC; i++)
  {
    if (res_scalars_wide_full[0].limbs_storage.limbs[i] == res_scalars_wide[0].limbs_storage.limbs[i])
    std::cout << "matched index = " << i << std::endl;
  }

}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
