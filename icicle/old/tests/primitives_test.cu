#include "primitives/test_kernels.cuh"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>

template <class T>
int device_populate_random(T* d_elements, unsigned n)
{
  T h_elements[n];
  for (unsigned i = 0; i < n; i++)
    h_elements[i] = T::rand_host();
  return cudaMemcpy(d_elements, h_elements, sizeof(T) * n, cudaMemcpyHostToDevice);
}

template <class T>
int device_set(T* d_elements, T el, unsigned n)
{
  T h_elements[n];
  for (unsigned i = 0; i < n; i++)
    h_elements[i] = el;
  return cudaMemcpy(d_elements, h_elements, sizeof(T) * n, cudaMemcpyHostToDevice);
}

class PrimitivesTest : public ::testing::Test
{
protected:
  static const unsigned n = 1 << 4;

  projective_t* points1{};
  projective_t* points2{};
  g2_projective_t* g2_points1{};
  g2_projective_t* g2_points2{};
  scalar_t* scalars1{};
  scalar_t* scalars2{};
  projective_t* zero_points{};
  g2_projective_t* g2_zero_points{};
  scalar_t* zero_scalars{};
  scalar_t* one_scalars{};
  affine_t* aff_points{};
  g2_affine_t* g2_aff_points{};
  projective_t* res_points1{};
  projective_t* res_points2{};
  g2_projective_t* g2_res_points1{};
  g2_projective_t* g2_res_points2{};
  scalar_t* res_scalars1{};
  scalar_t* res_scalars2{};
  scalar_t::Wide* res_scalars_wide{};
  scalar_t::Wide* res_scalars_wide_full{};

  PrimitivesTest()
  {
    assert(!cudaDeviceReset());
    assert(!cudaMallocManaged(&points1, n * sizeof(projective_t)));
    assert(!cudaMallocManaged(&points2, n * sizeof(projective_t)));
    assert(!cudaMallocManaged(&g2_points1, n * sizeof(g2_projective_t)));
    assert(!cudaMallocManaged(&g2_points2, n * sizeof(g2_projective_t)));
    assert(!cudaMallocManaged(&scalars1, n * sizeof(scalar_t)));
    assert(!cudaMallocManaged(&scalars2, n * sizeof(scalar_t)));
    assert(!cudaMallocManaged(&zero_points, n * sizeof(projective_t)));
    assert(!cudaMallocManaged(&g2_zero_points, n * sizeof(g2_projective_t)));
    assert(!cudaMallocManaged(&zero_scalars, n * sizeof(scalar_t)));
    assert(!cudaMallocManaged(&one_scalars, n * sizeof(scalar_t)));
    assert(!cudaMallocManaged(&aff_points, n * sizeof(affine_t)));
    assert(!cudaMallocManaged(&g2_aff_points, n * sizeof(g2_affine_t)));
    assert(!cudaMallocManaged(&res_points1, n * sizeof(projective_t)));
    assert(!cudaMallocManaged(&res_points2, n * sizeof(projective_t)));
    assert(!cudaMallocManaged(&g2_res_points1, n * sizeof(g2_projective_t)));
    assert(!cudaMallocManaged(&g2_res_points2, n * sizeof(g2_projective_t)));
    assert(!cudaMallocManaged(&res_scalars1, n * sizeof(scalar_t)));
    assert(!cudaMallocManaged(&res_scalars2, n * sizeof(scalar_t)));
  }

  ~PrimitivesTest() override
  {
    cudaFree(points1);
    cudaFree(points2);
    cudaFree(g2_points1);
    cudaFree(g2_points2);
    cudaFree(scalars1);
    cudaFree(scalars2);
    cudaFree(zero_points);
    cudaFree(g2_zero_points);
    cudaFree(zero_scalars);
    cudaFree(one_scalars);
    cudaFree(aff_points);
    cudaFree(g2_aff_points);
    cudaFree(res_points1);
    cudaFree(res_points2);
    cudaFree(g2_res_points1);
    cudaFree(g2_res_points2);
    cudaFree(res_scalars1);
    cudaFree(res_scalars2);

    cudaDeviceReset();
  }

  void SetUp() override
  {
    ASSERT_EQ(device_populate_random<projective_t>(points1, n), cudaSuccess);
    ASSERT_EQ(device_populate_random<projective_t>(points2, n), cudaSuccess);
    ASSERT_EQ(device_populate_random<g2_projective_t>(g2_points1, n), cudaSuccess);
    ASSERT_EQ(device_populate_random<g2_projective_t>(g2_points2, n), cudaSuccess);
    ASSERT_EQ(device_populate_random<scalar_t>(scalars1, n), cudaSuccess);
    ASSERT_EQ(device_populate_random<scalar_t>(scalars2, n), cudaSuccess);
    ASSERT_EQ(device_set<projective_t>(zero_points, projective_t::zero(), n), cudaSuccess);
    ASSERT_EQ(device_set<g2_projective_t>(g2_zero_points, g2_projective_t::zero(), n), cudaSuccess);
    ASSERT_EQ(device_set<scalar_t>(zero_scalars, scalar_t::zero(), n), cudaSuccess);
    ASSERT_EQ(device_set<scalar_t>(one_scalars, scalar_t::one(), n), cudaSuccess);
    ASSERT_EQ(cudaMemset(aff_points, 0, n * sizeof(affine_t)), cudaSuccess);
    ASSERT_EQ(cudaMemset(g2_aff_points, 0, n * sizeof(g2_affine_t)), cudaSuccess);
    ASSERT_EQ(cudaMemset(res_points1, 0, n * sizeof(projective_t)), cudaSuccess);
    ASSERT_EQ(cudaMemset(res_points2, 0, n * sizeof(projective_t)), cudaSuccess);
    ASSERT_EQ(cudaMemset(g2_res_points1, 0, n * sizeof(g2_projective_t)), cudaSuccess);
    ASSERT_EQ(cudaMemset(g2_res_points2, 0, n * sizeof(g2_projective_t)), cudaSuccess);
    ASSERT_EQ(cudaMemset(res_scalars1, 0, n * sizeof(scalar_t)), cudaSuccess);
    ASSERT_EQ(cudaMemset(res_scalars2, 0, n * sizeof(scalar_t)), cudaSuccess);
  }
};

TEST_F(PrimitivesTest, FieldAdditionSubtractionCancel)
{
  ASSERT_EQ(vec_add(scalars1, scalars2, res_scalars1, n), cudaSuccess);
  ASSERT_EQ(vec_sub(res_scalars1, scalars2, res_scalars2, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(scalars1[i], res_scalars2[i]);
}

TEST_F(PrimitivesTest, FieldZeroAddition)
{
  ASSERT_EQ(vec_add(scalars1, zero_scalars, res_scalars1, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(scalars1[i], res_scalars1[i]);
}

TEST_F(PrimitivesTest, FieldAdditionHostDeviceEq)
{
  ASSERT_EQ(vec_add(scalars1, scalars2, res_scalars1, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(scalars1[i] + scalars2[i], res_scalars1[i]);
}

TEST_F(PrimitivesTest, FieldMultiplicationByOne)
{
  ASSERT_EQ(vec_mul(scalars1, one_scalars, res_scalars1, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(scalars1[i], res_scalars1[i]);
}

TEST_F(PrimitivesTest, FieldMultiplicationByMinusOne)
{
  ASSERT_EQ(vec_neg(one_scalars, res_scalars1, n), cudaSuccess);
  ASSERT_EQ(vec_mul(scalars1, res_scalars1, res_scalars2, n), cudaSuccess);
  ASSERT_EQ(vec_add(scalars1, res_scalars2, res_scalars1, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(res_scalars1[i], zero_scalars[i]);
}

TEST_F(PrimitivesTest, FieldMultiplicationByZero)
{
  ASSERT_EQ(vec_mul(scalars1, zero_scalars, res_scalars1, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(zero_scalars[i], res_scalars1[i]);
}

TEST_F(PrimitivesTest, FieldMultiplicationInverseCancel)
{
  ASSERT_EQ(vec_mul(scalars1, scalars2, res_scalars1, n), cudaSuccess);
  ASSERT_EQ(field_vec_inv(scalars2, res_scalars2, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(scalars1[i], res_scalars1[i] * res_scalars2[i]);
}

TEST_F(PrimitivesTest, FieldMultiplicationHostDeviceEq)
{
  ASSERT_EQ(vec_mul(scalars1, scalars2, res_scalars1, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(scalars1[i] * scalars2[i], res_scalars1[i]);
}

TEST_F(PrimitivesTest, FieldMultiplicationByTwoEqSum)
{
  ASSERT_EQ(vec_add(one_scalars, one_scalars, res_scalars1, n), cudaSuccess);
  ASSERT_EQ(vec_mul(res_scalars1, scalars1, res_scalars2, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(res_scalars2[i], scalars1[i] + scalars1[i]);
}

TEST_F(PrimitivesTest, FieldSqrHostDeviceEq)
{
  ASSERT_EQ(field_vec_sqr(scalars1, res_scalars1, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(scalars1[i] * scalars1[i], res_scalars1[i]);
}

TEST_F(PrimitivesTest, FieldMultiplicationSqrEq)
{
  ASSERT_EQ(vec_mul(scalars1, scalars1, res_scalars1, n), cudaSuccess);
  ASSERT_EQ(field_vec_sqr(scalars1, res_scalars2, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(res_scalars1[i], res_scalars2[i]);
}

TEST_F(PrimitivesTest, ECRandomPointsAreOnCurve)
{
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED1(projective_t::is_on_curve, points1[i]);
}

TEST_F(PrimitivesTest, ECPointAdditionSubtractionCancel)
{
  ASSERT_EQ(vec_add(points1, points2, res_points1, n), cudaSuccess);
  ASSERT_EQ(vec_sub(res_points1, points2, res_points2, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(points1[i], res_points2[i]);
}

TEST_F(PrimitivesTest, ECPointZeroAddition)
{
  ASSERT_EQ(vec_add(points1, zero_points, res_points1, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(points1[i], res_points1[i]);
}

TEST_F(PrimitivesTest, ECPointAdditionHostDeviceEq)
{
  ASSERT_EQ(vec_add(points1, points2, res_points1, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(points1[i] + points2[i], res_points1[i]);
}

TEST_F(PrimitivesTest, ECScalarMultiplicationHostDeviceEq)
{
  ASSERT_EQ(vec_mul(scalars1, points1, res_points1, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(scalars1[i] * points1[i], res_points1[i]);
}

TEST_F(PrimitivesTest, ECScalarMultiplicationByOne)
{
  ASSERT_EQ(vec_mul(one_scalars, points1, res_points1, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(points1[i], res_points1[i]);
}

TEST_F(PrimitivesTest, ECScalarMultiplicationByMinusOne)
{
  ASSERT_EQ(vec_neg(one_scalars, res_scalars1, n), cudaSuccess);
  ASSERT_EQ(vec_mul(res_scalars1, points1, res_points1, n), cudaSuccess);
  ASSERT_EQ(vec_neg(points1, res_points2, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(res_points1[i], res_points2[i]);
}

TEST_F(PrimitivesTest, ECScalarMultiplicationByTwo)
{
  ASSERT_EQ(vec_add(one_scalars, one_scalars, res_scalars1, n), cudaSuccess);
  ASSERT_EQ(vec_mul(res_scalars1, points1, res_points1, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ((one_scalars[i] + one_scalars[i]) * points1[i], res_points1[i]);
}

TEST_F(PrimitivesTest, ECScalarMultiplicationInverseCancel)
{
  ASSERT_EQ(vec_mul(scalars1, points1, res_points1, n), cudaSuccess);
  ASSERT_EQ(field_vec_inv(scalars1, res_scalars1, n), cudaSuccess);
  ASSERT_EQ(vec_mul(res_scalars1, res_points1, res_points2, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(points1[i], res_points2[i]);
}

TEST_F(PrimitivesTest, ECScalarMultiplicationIsDistributiveOverMultiplication)
{
  ASSERT_EQ(vec_mul(scalars1, points1, res_points1, n), cudaSuccess);
  ASSERT_EQ(vec_mul(scalars2, res_points1, res_points2, n), cudaSuccess);
  ASSERT_EQ(vec_mul(scalars1, scalars2, res_scalars1, n), cudaSuccess);
  ASSERT_EQ(vec_mul(res_scalars1, points1, res_points1, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(res_points1[i], res_points2[i]);
}

TEST_F(PrimitivesTest, ECScalarMultiplicationIsDistributiveOverAddition)
{
  ASSERT_EQ(vec_mul(scalars1, points1, res_points1, n), cudaSuccess);
  ASSERT_EQ(vec_mul(scalars2, points1, res_points2, n), cudaSuccess);
  ASSERT_EQ(vec_add(scalars1, scalars2, res_scalars1, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(res_scalars1[i] * points1[i], res_points1[i] + res_points2[i]);
}

TEST_F(PrimitivesTest, ECProjectiveToAffine)
{
  ASSERT_EQ(point_vec_to_affine(points1, aff_points, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(points1[i], projective_t::from_affine(aff_points[i]));
}

TEST_F(PrimitivesTest, ECMixedPointAddition)
{
  ASSERT_EQ(point_vec_to_affine(points2, aff_points, n), cudaSuccess);
  ASSERT_EQ(vec_add(points1, aff_points, res_points1, n), cudaSuccess);
  ASSERT_EQ(vec_add(points1, points2, res_points2, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(res_points1[i], res_points2[i]);
}

TEST_F(PrimitivesTest, ECMixedAdditionOfNegatedPointEqSubtraction)
{
  ASSERT_EQ(point_vec_to_affine(points2, aff_points, n), cudaSuccess);
  ASSERT_EQ(vec_sub(points1, aff_points, res_points1, n), cudaSuccess);
  ASSERT_EQ(vec_neg(points2, res_points2, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(res_points1[i], points1[i] + res_points2[i]);
}

TEST_F(PrimitivesTest, G2ECRandomPointsAreOnCurve)
{
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED1(g2_projective_t::is_on_curve, g2_points1[i]);
}

TEST_F(PrimitivesTest, G2ECPointAdditionSubtractionCancel)
{
  ASSERT_EQ(vec_add(g2_points1, g2_points2, g2_res_points1, n), cudaSuccess);
  ASSERT_EQ(vec_sub(g2_res_points1, g2_points2, g2_res_points2, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(g2_points1[i], g2_res_points2[i]);
}

TEST_F(PrimitivesTest, G2ECPointZeroAddition)
{
  ASSERT_EQ(vec_add(g2_points1, g2_zero_points, g2_res_points1, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(g2_points1[i], g2_res_points1[i]);
}

TEST_F(PrimitivesTest, G2ECPointAdditionHostDeviceEq)
{
  ASSERT_EQ(vec_add(g2_points1, g2_points2, g2_res_points1, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(g2_points1[i] + g2_points2[i], g2_res_points1[i]);
}

TEST_F(PrimitivesTest, G2ECScalarMultiplicationHostDeviceEq)
{
  ASSERT_EQ(vec_mul(scalars1, g2_points1, g2_res_points1, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(scalars1[i] * g2_points1[i], g2_res_points1[i]);
}

TEST_F(PrimitivesTest, G2ECScalarMultiplicationByOne)
{
  ASSERT_EQ(vec_mul(one_scalars, points1, res_points1, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(g2_points1[i], g2_res_points1[i]);
}

TEST_F(PrimitivesTest, G2ECScalarMultiplicationByMinusOne)
{
  ASSERT_EQ(vec_neg(one_scalars, res_scalars1, n), cudaSuccess);
  ASSERT_EQ(vec_mul(res_scalars1, g2_points1, g2_res_points1, n), cudaSuccess);
  ASSERT_EQ(vec_neg(g2_points1, g2_res_points2, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(g2_res_points1[i], g2_res_points2[i]);
}

TEST_F(PrimitivesTest, G2ECScalarMultiplicationByTwo)
{
  ASSERT_EQ(vec_add(one_scalars, one_scalars, res_scalars1, n), cudaSuccess);
  ASSERT_EQ(vec_mul(res_scalars1, g2_points1, g2_res_points1, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ((one_scalars[i] + one_scalars[i]) * g2_points1[i], g2_res_points1[i]);
}

TEST_F(PrimitivesTest, G2ECScalarMultiplicationInverseCancel)
{
  ASSERT_EQ(vec_mul(scalars1, g2_points1, g2_res_points1, n), cudaSuccess);
  ASSERT_EQ(field_vec_inv(scalars1, res_scalars1, n), cudaSuccess);
  ASSERT_EQ(vec_mul(res_scalars1, g2_res_points1, g2_res_points2, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(g2_points1[i], g2_res_points2[i]);
}

TEST_F(PrimitivesTest, G2ECScalarMultiplicationIsDistributiveOverMultiplication)
{
  ASSERT_EQ(vec_mul(scalars1, g2_points1, g2_res_points1, n), cudaSuccess);
  ASSERT_EQ(vec_mul(scalars2, g2_res_points1, g2_res_points2, n), cudaSuccess);
  ASSERT_EQ(vec_mul(scalars1, scalars2, res_scalars1, n), cudaSuccess);
  ASSERT_EQ(vec_mul(res_scalars1, g2_points1, g2_res_points1, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(g2_res_points1[i], g2_res_points2[i]);
}

TEST_F(PrimitivesTest, G2ECScalarMultiplicationIsDistributiveOverAddition)
{
  ASSERT_EQ(vec_mul(scalars1, g2_points1, g2_res_points1, n), cudaSuccess);
  ASSERT_EQ(vec_mul(scalars2, g2_points1, g2_res_points2, n), cudaSuccess);
  ASSERT_EQ(vec_add(scalars1, scalars2, res_scalars1, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(res_scalars1[i] * g2_points1[i], g2_res_points1[i] + g2_res_points2[i]);
}

TEST_F(PrimitivesTest, G2ECProjectiveToAffine)
{
  ASSERT_EQ(point_vec_to_affine(g2_points1, g2_aff_points, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(g2_points1[i], g2_projective_t::from_affine(g2_aff_points[i]));
}

TEST_F(PrimitivesTest, G2ECMixedPointAddition)
{
  ASSERT_EQ(point_vec_to_affine(g2_points2, g2_aff_points, n), cudaSuccess);
  ASSERT_EQ(vec_add(g2_points1, g2_aff_points, g2_res_points1, n), cudaSuccess);
  ASSERT_EQ(vec_add(g2_points1, g2_points2, g2_res_points2, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(g2_res_points1[i], g2_res_points2[i]);
}

TEST_F(PrimitivesTest, G2ECMixedAdditionOfNegatedPointEqSubtraction)
{
  ASSERT_EQ(point_vec_to_affine(g2_points2, g2_aff_points, n), cudaSuccess);
  ASSERT_EQ(vec_sub(g2_points1, g2_aff_points, g2_res_points1, n), cudaSuccess);
  ASSERT_EQ(vec_neg(g2_points2, g2_res_points2, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(g2_res_points1[i], g2_points1[i] + g2_res_points2[i]);
}