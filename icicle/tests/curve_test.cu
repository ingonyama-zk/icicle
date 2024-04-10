#include "utils/test_functions.cuh"
#include "curves/curve_config.cuh"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>

using namespace curve_config;

template <class P>
class CurveTest : public ::testing::Test
{
protected:
  using TestP = P;

  static const unsigned n = 1 << 4;

  P* points1{};
  P* points2{};
  typename P::Scalar* scalars1{};
  typename P::Scalar* scalars2{};
  P* zero_points{};
  typename P::Scalar* one_scalars{};
  typename P::Aff* aff_points{};
  P* res_points1{};
  P* res_points2{};
  typename P::Scalar* res_scalars{};

  CurveTest()
  {
    assert(!cudaDeviceReset());
    assert(!cudaMallocManaged(&points1, n * sizeof(P)));
    assert(!cudaMallocManaged(&points2, n * sizeof(P)));
    assert(!cudaMallocManaged(&scalars1, n * sizeof(typename P::Scalar)));
    assert(!cudaMallocManaged(&scalars2, n * sizeof(typename P::Scalar)));
    assert(!cudaMallocManaged(&zero_points, n * sizeof(P)));
    assert(!cudaMallocManaged(&one_scalars, n * sizeof(typename P::Scalar)));
    assert(!cudaMallocManaged(&aff_points, n * sizeof(typename P::Aff)));
    assert(!cudaMallocManaged(&res_points1, n * sizeof(P)));
    assert(!cudaMallocManaged(&res_points2, n * sizeof(P)));
    assert(!cudaMallocManaged(&res_scalars, n * sizeof(typename P::Scalar)));
  }

  ~CurveTest() override
  {
    cudaFree(points1);
    cudaFree(points2);
    cudaFree(scalars1);
    cudaFree(scalars2);
    cudaFree(zero_points);
    cudaFree(one_scalars);
    cudaFree(aff_points);
    cudaFree(res_points1);
    cudaFree(res_points2);
    cudaFree(res_scalars);

    cudaDeviceReset();
  }

  void SetUp() override
  {
    ASSERT_EQ(device_populate_random<P>(points1, n), cudaSuccess);
    ASSERT_EQ(device_populate_random<P>(points2, n), cudaSuccess);
    ASSERT_EQ(device_populate_random<typename P::Scalar>(scalars1, n), cudaSuccess);
    ASSERT_EQ(device_populate_random<typename P::Scalar>(scalars2, n), cudaSuccess);
    ASSERT_EQ(device_set<P>(zero_points, P::zero(), n), cudaSuccess);
    ASSERT_EQ(device_set<typename P::Scalar>(one_scalars, P::Scalar::one(), n), cudaSuccess);
    ASSERT_EQ(cudaMemset(aff_points, 0, n * sizeof(typename P::Aff)), cudaSuccess);
    ASSERT_EQ(cudaMemset(res_points1, 0, n * sizeof(P)), cudaSuccess);
    ASSERT_EQ(cudaMemset(res_points2, 0, n * sizeof(P)), cudaSuccess);
    ASSERT_EQ(cudaMemset(res_scalars, 0, n * sizeof(typename P::Scalar)), cudaSuccess);
  }
};

#ifdef G2
typedef testing::Types<projective_t, g2_projective_t> CTImplementations;
#else
typedef testing::Types<projective_t> CTImplementations;
#endif

TYPED_TEST_SUITE(CurveTest, CTImplementations);

TYPED_TEST(CurveTest, ECRandomPointsAreOnCurve)
{
  using TestP = typename TestFixture::TestP;
  for (unsigned i = 0; i < this->n; i++)
    ASSERT_PRED1(TestP::is_on_curve, this->points1[i]);
}

TYPED_TEST(CurveTest, ECPointAdditionSubtractionCancel)
{
  ASSERT_EQ(vec_add(this->points1, this->points2, this->res_points1, this->n), cudaSuccess);
  ASSERT_EQ(vec_sub(this->res_points1, this->points2, this->res_points2, this->n), cudaSuccess);
  for (unsigned i = 0; i < this->n; i++)
    ASSERT_EQ(this->points1[i], this->res_points2[i]);
}

TYPED_TEST(CurveTest, ECPointZeroAddition)
{
  ASSERT_EQ(vec_add(this->points1, this->zero_points, this->res_points1, this->n), cudaSuccess);
  for (unsigned i = 0; i < this->n; i++)
    ASSERT_EQ(this->points1[i], this->res_points1[i]);
}

TYPED_TEST(CurveTest, ECPointAdditionHostDeviceEq)
{
  ASSERT_EQ(vec_add(this->points1, this->points2, this->res_points1, this->n), cudaSuccess);
  for (unsigned i = 0; i < this->n; i++)
    ASSERT_EQ(this->points1[i] + this->points2[i], this->res_points1[i]);
}

TYPED_TEST(CurveTest, ECScalarMultiplicationHostDeviceEq)
{
  ASSERT_EQ(vec_mul(this->scalars1, this->points1, this->res_points1, this->n), cudaSuccess);
  for (unsigned i = 0; i < this->n; i++)
    ASSERT_EQ(this->scalars1[i] * this->points1[i], this->res_points1[i]);
}

TYPED_TEST(CurveTest, ECScalarMultiplicationByOne)
{
  ASSERT_EQ(vec_mul(this->one_scalars, this->points1, this->res_points1, this->n), cudaSuccess);
  for (unsigned i = 0; i < this->n; i++)
    ASSERT_EQ(this->points1[i], this->res_points1[i]);
}

TYPED_TEST(CurveTest, ECScalarMultiplicationByMinusOne)
{
  ASSERT_EQ(vec_neg(this->one_scalars, this->res_scalars, this->n), cudaSuccess);
  ASSERT_EQ(vec_mul(this->res_scalars, this->points1, this->res_points1, this->n), cudaSuccess);
  ASSERT_EQ(vec_neg(this->points1, this->res_points2, this->n), cudaSuccess);
  for (unsigned i = 0; i < this->n; i++)
    ASSERT_EQ(this->res_points1[i], this->res_points2[i]);
}

TYPED_TEST(CurveTest, ECScalarMultiplicationByTwo)
{
  ASSERT_EQ(vec_add(this->one_scalars, this->one_scalars, this->res_scalars, this->n), cudaSuccess);
  ASSERT_EQ(vec_mul(this->res_scalars, this->points1, this->res_points1, this->n), cudaSuccess);
  for (unsigned i = 0; i < this->n; i++)
    ASSERT_EQ((this->one_scalars[i] + this->one_scalars[i]) * this->points1[i], this->res_points1[i]);
}

TYPED_TEST(CurveTest, ECScalarMultiplicationInverseCancel)
{
  ASSERT_EQ(vec_mul(this->scalars1, this->points1, this->res_points1, this->n), cudaSuccess);
  ASSERT_EQ(field_vec_inv(this->scalars1, this->res_scalars, this->n), cudaSuccess);
  ASSERT_EQ(vec_mul(this->res_scalars, this->res_points1, this->res_points2, this->n), cudaSuccess);
  for (unsigned i = 0; i < this->n; i++)
    ASSERT_EQ(this->points1[i], this->res_points2[i]);
}

TYPED_TEST(CurveTest, ECScalarMultiplicationIsDistributiveOverMultiplication)
{
  ASSERT_EQ(vec_mul(this->scalars1, this->points1, this->res_points1, this->n), cudaSuccess);
  ASSERT_EQ(vec_mul(this->scalars2, this->res_points1, this->res_points2, this->n), cudaSuccess);
  ASSERT_EQ(vec_mul(this->scalars1, this->scalars2, this->res_scalars, this->n), cudaSuccess);
  ASSERT_EQ(vec_mul(this->res_scalars, this->points1, this->res_points1, this->n), cudaSuccess);
  for (unsigned i = 0; i < this->n; i++)
    ASSERT_EQ(this->res_points1[i], this->res_points2[i]);
}

TYPED_TEST(CurveTest, ECScalarMultiplicationIsDistributiveOverAddition)
{
  ASSERT_EQ(vec_mul(this->scalars1, this->points1, this->res_points1, this->n), cudaSuccess);
  ASSERT_EQ(vec_mul(this->scalars2, this->points1, this->res_points2, this->n), cudaSuccess);
  ASSERT_EQ(vec_add(this->scalars1, this->scalars2, this->res_scalars, this->n), cudaSuccess);
  for (unsigned i = 0; i < this->n; i++)
    ASSERT_EQ(this->res_scalars[i] * this->points1[i], this->res_points1[i] + this->res_points2[i]);
}

TYPED_TEST(CurveTest, ECProjectiveToAffine)
{
  using TestP = typename TestFixture::TestP;
  ASSERT_EQ(point_vec_to_affine(this->points1, this->aff_points, this->n), cudaSuccess);
  for (unsigned i = 0; i < this->n; i++)
    ASSERT_EQ(this->points1[i], TestP::from_affine(this->aff_points[i]));
}

TYPED_TEST(CurveTest, ECMixedPointAddition)
{
  ASSERT_EQ(point_vec_to_affine(this->points2, this->aff_points, this->n), cudaSuccess);
  ASSERT_EQ(vec_add(this->points1, this->aff_points, this->res_points1, this->n), cudaSuccess);
  ASSERT_EQ(vec_add(this->points1, this->points2, this->res_points2, this->n), cudaSuccess);
  for (unsigned i = 0; i < this->n; i++)
    ASSERT_EQ(this->res_points1[i], this->res_points2[i]);
}

TYPED_TEST(CurveTest, ECMixedAdditionOfNegatedPointEqSubtraction)
{
  ASSERT_EQ(point_vec_to_affine(this->points2, this->aff_points, this->n), cudaSuccess);
  ASSERT_EQ(vec_sub(this->points1, this->aff_points, this->res_points1, this->n), cudaSuccess);
  ASSERT_EQ(vec_neg(this->points2, this->res_points2, this->n), cudaSuccess);
  for (unsigned i = 0; i < this->n; i++)
    ASSERT_EQ(this->res_points1[i], this->points1[i] + this->res_points2[i]);
}