#include "utils/test_functions.cuh"
#include "fields/field_config.cuh"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>

using namespace field_config;

class FieldTest : public ::testing::Test
{
protected:
  static const unsigned n = 1 << 4;

  scalar_t* scalars1{};
  scalar_t* scalars2{};
  scalar_t* zero_scalars{};
  scalar_t* one_scalars{};
  scalar_t* res_scalars1{};
  scalar_t* res_scalars2{};

  FieldTest()
  {
    assert(!cudaDeviceReset());
    assert(!cudaMallocManaged(&scalars1, n * sizeof(scalar_t)));
    assert(!cudaMallocManaged(&scalars2, n * sizeof(scalar_t)));
    assert(!cudaMallocManaged(&zero_scalars, n * sizeof(scalar_t)));
    assert(!cudaMallocManaged(&one_scalars, n * sizeof(scalar_t)));
    assert(!cudaMallocManaged(&res_scalars1, n * sizeof(scalar_t)));
    assert(!cudaMallocManaged(&res_scalars2, n * sizeof(scalar_t)));
  }

  ~FieldTest() override
  {
    cudaFree(scalars1);
    cudaFree(scalars2);
    cudaFree(zero_scalars);
    cudaFree(one_scalars);
    cudaFree(res_scalars1);
    cudaFree(res_scalars2);

    cudaDeviceReset();
  }

  void SetUp() override
  {
    ASSERT_EQ(device_populate_random<scalar_t>(scalars1, n), cudaSuccess);
    ASSERT_EQ(device_populate_random<scalar_t>(scalars2, n), cudaSuccess);
    ASSERT_EQ(device_set(zero_scalars, scalar_t::zero(), n), cudaSuccess);
    ASSERT_EQ(device_set(one_scalars, scalar_t::one(), n), cudaSuccess);
    ASSERT_EQ(cudaMemset(res_scalars1, 0, n * sizeof(scalar_t)), cudaSuccess);
    ASSERT_EQ(cudaMemset(res_scalars2, 0, n * sizeof(scalar_t)), cudaSuccess);
  }
};

TEST_F(FieldTest, FieldAdditionSubtractionCancel)
{
  ASSERT_EQ(vec_add(scalars1, scalars2, res_scalars1, n), cudaSuccess);
  ASSERT_EQ(vec_sub(res_scalars1, scalars2, res_scalars2, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(scalars1[i], res_scalars2[i]);
}

TEST_F(FieldTest, FieldZeroAddition)
{
  ASSERT_EQ(vec_add(scalars1, zero_scalars, res_scalars1, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(scalars1[i], res_scalars1[i]);
}

TEST_F(FieldTest, FieldAdditionHostDeviceEq)
{
  ASSERT_EQ(vec_add(scalars1, scalars2, res_scalars1, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(scalars1[i] + scalars2[i], res_scalars1[i]);
}

TEST_F(FieldTest, FieldMultiplicationByOne)
{
  ASSERT_EQ(vec_mul(scalars1, one_scalars, res_scalars1, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(scalars1[i], res_scalars1[i]);
}

TEST_F(FieldTest, FieldMultiplicationByMinusOne)
{
  ASSERT_EQ(vec_neg(one_scalars, res_scalars1, n), cudaSuccess);
  ASSERT_EQ(vec_mul(scalars1, res_scalars1, res_scalars2, n), cudaSuccess);
  ASSERT_EQ(vec_add(scalars1, res_scalars2, res_scalars1, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(res_scalars1[i], zero_scalars[i]);
}

TEST_F(FieldTest, FieldMultiplicationByZero)
{
  ASSERT_EQ(vec_mul(scalars1, zero_scalars, res_scalars1, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(zero_scalars[i], res_scalars1[i]);
}

TEST_F(FieldTest, FieldMultiplicationInverseCancel)
{
  ASSERT_EQ(vec_mul(scalars1, scalars2, res_scalars1, n), cudaSuccess);
  ASSERT_EQ(field_vec_inv(scalars2, res_scalars2, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(scalars1[i], res_scalars1[i] * res_scalars2[i]);
}

TEST_F(FieldTest, FieldMultiplicationHostDeviceEq)
{
  ASSERT_EQ(vec_mul(scalars1, scalars2, res_scalars1, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(scalars1[i] * scalars2[i], res_scalars1[i]);
}

TEST_F(FieldTest, FieldMultiplicationByTwoEqSum)
{
  ASSERT_EQ(vec_add(one_scalars, one_scalars, res_scalars1, n), cudaSuccess);
  ASSERT_EQ(vec_mul(res_scalars1, scalars1, res_scalars2, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(res_scalars2[i], scalars1[i] + scalars1[i]);
}

TEST_F(FieldTest, FieldSqrHostDeviceEq)
{
  ASSERT_EQ(field_vec_sqr(scalars1, res_scalars1, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(scalars1[i] * scalars1[i], res_scalars1[i]);
}

TEST_F(FieldTest, FieldMultiplicationSqrEq)
{
  ASSERT_EQ(vec_mul(scalars1, scalars1, res_scalars1, n), cudaSuccess);
  ASSERT_EQ(field_vec_sqr(scalars1, res_scalars2, n), cudaSuccess);
  for (unsigned i = 0; i < n; i++)
    ASSERT_EQ(res_scalars1[i], res_scalars2[i]);
}
