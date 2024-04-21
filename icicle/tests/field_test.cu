#include "utils/test_functions.cuh"
#include "fields/field_config.cuh"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>

using namespace field_config;

template <class T>
class FieldTest : public ::testing::Test
{
protected:
  static const unsigned n = 1 << 4;

  T* scalars1{};
  T* scalars2{};
  T* zero_scalars{};
  T* one_scalars{};
  T* res_scalars1{};
  T* res_scalars2{};

  FieldTest()
  {
    assert(!cudaDeviceReset());
    assert(!cudaMallocManaged(&scalars1, n * sizeof(T)));
    assert(!cudaMallocManaged(&scalars2, n * sizeof(T)));
    assert(!cudaMallocManaged(&zero_scalars, n * sizeof(T)));
    assert(!cudaMallocManaged(&one_scalars, n * sizeof(T)));
    assert(!cudaMallocManaged(&res_scalars1, n * sizeof(T)));
    assert(!cudaMallocManaged(&res_scalars2, n * sizeof(T)));
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
    ASSERT_EQ(device_populate_random<T>(scalars1, n), cudaSuccess);
    ASSERT_EQ(device_populate_random<T>(scalars2, n), cudaSuccess);
    ASSERT_EQ(device_set(zero_scalars, T::zero(), n), cudaSuccess);
    ASSERT_EQ(device_set(one_scalars, T::one(), n), cudaSuccess);
    ASSERT_EQ(cudaMemset(res_scalars1, 0, n * sizeof(T)), cudaSuccess);
    ASSERT_EQ(cudaMemset(res_scalars2, 0, n * sizeof(T)), cudaSuccess);
  }
};

#ifdef EXT_FIELD
typedef testing::Types<scalar_t, extension_t> FTImplementations;
#else
typedef testing::Types<scalar_t> FTImplementations;
#endif

TYPED_TEST_SUITE(FieldTest, FTImplementations);

TYPED_TEST(FieldTest, FieldAdditionSubtractionCancel)
{
  ASSERT_EQ(vec_add(this->scalars1, this->scalars2, this->res_scalars1, this->n), cudaSuccess);
  ASSERT_EQ(vec_sub(this->res_scalars1, this->scalars2, this->res_scalars2, this->n), cudaSuccess);
  for (unsigned i = 0; i < this->n; i++)
    ASSERT_EQ(this->scalars1[i], this->res_scalars2[i]);
}

TYPED_TEST(FieldTest, FieldZeroAddition)
{
  ASSERT_EQ(vec_add(this->scalars1, this->zero_scalars, this->res_scalars1, this->n), cudaSuccess);
  for (unsigned i = 0; i < this->n; i++)
    ASSERT_EQ(this->scalars1[i], this->res_scalars1[i]);
}

TYPED_TEST(FieldTest, FieldAdditionHostDeviceEq)
{
  ASSERT_EQ(vec_add(this->scalars1, this->scalars2, this->res_scalars1, this->n), cudaSuccess);
  for (unsigned i = 0; i < this->n; i++)
    ASSERT_EQ(this->scalars1[i] + this->scalars2[i], this->res_scalars1[i]);
}

TYPED_TEST(FieldTest, FieldMultiplicationByOne)
{
  ASSERT_EQ(vec_mul(this->scalars1, this->one_scalars, this->res_scalars1, this->n), cudaSuccess);
  for (unsigned i = 0; i < this->n; i++)
    ASSERT_EQ(this->scalars1[i], this->res_scalars1[i]);
}

TYPED_TEST(FieldTest, FieldMultiplicationByMinusOne)
{
  ASSERT_EQ(vec_neg(this->one_scalars, this->res_scalars1, this->n), cudaSuccess);
  ASSERT_EQ(vec_mul(this->scalars1, this->res_scalars1, this->res_scalars2, this->n), cudaSuccess);
  ASSERT_EQ(vec_add(this->scalars1, this->res_scalars2, this->res_scalars1, this->n), cudaSuccess);
  for (unsigned i = 0; i < this->n; i++)
    ASSERT_EQ(this->res_scalars1[i], this->zero_scalars[i]);
}

TYPED_TEST(FieldTest, FieldMultiplicationByZero)
{
  ASSERT_EQ(vec_mul(this->scalars1, this->zero_scalars, this->res_scalars1, this->n), cudaSuccess);
  for (unsigned i = 0; i < this->n; i++)
    ASSERT_EQ(this->zero_scalars[i], this->res_scalars1[i]);
}

TYPED_TEST(FieldTest, FieldMultiplicationInverseCancel)
{
  ASSERT_EQ(vec_mul(this->scalars1, this->scalars2, this->res_scalars1, this->n), cudaSuccess);
  ASSERT_EQ(field_vec_inv(this->scalars2, this->res_scalars2, this->n), cudaSuccess);
  for (unsigned i = 0; i < this->n; i++)
    ASSERT_EQ(this->scalars1[i], this->res_scalars1[i] * this->res_scalars2[i]);
}

TYPED_TEST(FieldTest, FieldMultiplicationHostDeviceEq)
{
  ASSERT_EQ(vec_mul(this->scalars1, this->scalars2, this->res_scalars1, this->n), cudaSuccess);
  for (unsigned i = 0; i < this->n; i++)
    ASSERT_EQ(this->scalars1[i] * this->scalars2[i], this->res_scalars1[i]);
}

TYPED_TEST(FieldTest, FieldMultiplicationByTwoEqSum)
{
  ASSERT_EQ(vec_add(this->one_scalars, this->one_scalars, this->res_scalars1, this->n), cudaSuccess);
  ASSERT_EQ(vec_mul(this->res_scalars1, this->scalars1, this->res_scalars2, this->n), cudaSuccess);
  for (unsigned i = 0; i < this->n; i++)
    ASSERT_EQ(this->res_scalars2[i], this->scalars1[i] + this->scalars1[i]);
}

TYPED_TEST(FieldTest, FieldSqrHostDeviceEq)
{
  ASSERT_EQ(field_vec_sqr(this->scalars1, this->res_scalars1, this->n), cudaSuccess);
  for (unsigned i = 0; i < this->n; i++)
    ASSERT_EQ(this->scalars1[i] * this->scalars1[i], this->res_scalars1[i]);
}

TYPED_TEST(FieldTest, FieldMultiplicationSqrEq)
{
  ASSERT_EQ(vec_mul(this->scalars1, this->scalars1, this->res_scalars1, this->n), cudaSuccess);
  ASSERT_EQ(field_vec_sqr(this->scalars1, this->res_scalars2, this->n), cudaSuccess);
  for (unsigned i = 0; i < this->n; i++)
    ASSERT_EQ(this->res_scalars1[i], this->res_scalars2[i]);
}
