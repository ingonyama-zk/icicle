#include "test_kernels.cuh"
#include <boost/multiprecision/cpp_int.hpp>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>
namespace mp = boost::multiprecision;

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

mp::int1024_t convert_to_boost_mp(uint32_t* a, uint32_t length)
{
  mp::int1024_t res = 0;
  for (uint32_t i = 0; i < length; i++) {
    res += (mp::int1024_t)(a[i]) << 32 * i;
  }
  return res;
}

class PrimitivesTest : public ::testing::Test
{
protected:
  static const unsigned n = 1 << 4;

  projective_t* points1{};
  projective_t* points2{};
  g2_projective_t* g2_points1{};
  g2_projective_t* g2_points2{};
  scalar_field_t* scalars1{};
  scalar_field_t* scalars2{};
  projective_t* zero_points{};
  g2_projective_t* g2_zero_points{};
  scalar_field_t* zero_scalars{};
  scalar_field_t* one_scalars{};
  affine_t* aff_points{};
  g2_affine_t* g2_aff_points{};
  projective_t* res_points1{};
  projective_t* res_points2{};
  g2_projective_t* g2_res_points1{};
  g2_projective_t* g2_res_points2{};
  scalar_field_t* res_scalars1{};
  scalar_field_t* res_scalars2{};
  scalar_field_t::Wide* res_scalars_wide{};
  scalar_field_t::Wide* res_scalars_wide_full{};

  PrimitivesTest()
  {
    assert(!cudaDeviceReset());
    assert(!cudaMallocManaged(&points1, n * sizeof(projective_t)));
    assert(!cudaMallocManaged(&points2, n * sizeof(projective_t)));
    assert(!cudaMallocManaged(&g2_points1, n * sizeof(g2_projective_t)));
    assert(!cudaMallocManaged(&g2_points2, n * sizeof(g2_projective_t)));
    assert(!cudaMallocManaged(&scalars1, n * sizeof(scalar_field_t)));
    assert(!cudaMallocManaged(&scalars2, n * sizeof(scalar_field_t)));
    assert(!cudaMallocManaged(&zero_points, n * sizeof(projective_t)));
    assert(!cudaMallocManaged(&g2_zero_points, n * sizeof(g2_projective_t)));
    assert(!cudaMallocManaged(&zero_scalars, n * sizeof(scalar_field_t)));
    assert(!cudaMallocManaged(&one_scalars, n * sizeof(scalar_field_t)));
    assert(!cudaMallocManaged(&aff_points, n * sizeof(affine_t)));
    assert(!cudaMallocManaged(&g2_aff_points, n * sizeof(g2_affine_t)));
    assert(!cudaMallocManaged(&res_points1, n * sizeof(projective_t)));
    assert(!cudaMallocManaged(&res_points2, n * sizeof(projective_t)));
    assert(!cudaMallocManaged(&g2_res_points1, n * sizeof(g2_projective_t)));
    assert(!cudaMallocManaged(&g2_res_points2, n * sizeof(g2_projective_t)));
    assert(!cudaMallocManaged(&res_scalars1, n * sizeof(scalar_field_t)));
    assert(!cudaMallocManaged(&res_scalars2, n * sizeof(scalar_field_t)));
    assert(!cudaMallocManaged(&res_scalars_wide, n * sizeof(scalar_field_t::Wide)));
    assert(!cudaMallocManaged(&res_scalars_wide_full, n * sizeof(scalar_field_t::Wide)));
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

    cudaFree(res_scalars_wide);
    cudaFree(res_scalars_wide_full);

    cudaDeviceReset();
  }

  void SetUp() override
  {
    ASSERT_EQ(device_populate_random<projective_t>(points1, n), cudaSuccess);
    ASSERT_EQ(device_populate_random<projective_t>(points2, n), cudaSuccess);
    ASSERT_EQ(device_populate_random<g2_projective_t>(g2_points1, n), cudaSuccess);
    ASSERT_EQ(device_populate_random<g2_projective_t>(g2_points2, n), cudaSuccess);
    ASSERT_EQ(device_populate_random<scalar_field_t>(scalars1, n), cudaSuccess);
    ASSERT_EQ(device_populate_random<scalar_field_t>(scalars2, n), cudaSuccess);
    ASSERT_EQ(device_set<projective_t>(zero_points, projective_t::zero(), n), cudaSuccess);
    ASSERT_EQ(device_set<g2_projective_t>(g2_zero_points, g2_projective_t::zero(), n), cudaSuccess);
    ASSERT_EQ(device_set<scalar_field_t>(zero_scalars, scalar_field_t::zero(), n), cudaSuccess);
    ASSERT_EQ(device_set<scalar_field_t>(one_scalars, scalar_field_t::one(), n), cudaSuccess);
    ASSERT_EQ(cudaMemset(aff_points, 0, n * sizeof(affine_t)), cudaSuccess);
    ASSERT_EQ(cudaMemset(g2_aff_points, 0, n * sizeof(g2_affine_t)), cudaSuccess);
    ASSERT_EQ(cudaMemset(res_points1, 0, n * sizeof(projective_t)), cudaSuccess);
    ASSERT_EQ(cudaMemset(res_points2, 0, n * sizeof(projective_t)), cudaSuccess);
    ASSERT_EQ(cudaMemset(g2_res_points1, 0, n * sizeof(g2_projective_t)), cudaSuccess);
    ASSERT_EQ(cudaMemset(g2_res_points2, 0, n * sizeof(g2_projective_t)), cudaSuccess);
    ASSERT_EQ(cudaMemset(res_scalars1, 0, n * sizeof(scalar_field_t)), cudaSuccess);
    ASSERT_EQ(cudaMemset(res_scalars2, 0, n * sizeof(scalar_field_t)), cudaSuccess);
    ASSERT_EQ(cudaMemset(res_scalars_wide, 0, n * sizeof(scalar_field_t::Wide)), cudaSuccess);
    ASSERT_EQ(cudaMemset(res_scalars_wide_full, 0, n * sizeof(scalar_field_t::Wide)), cudaSuccess);
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

TEST_F(PrimitivesTest, MP_LSB_MULT)
{
  // LSB multiply, check correctness of first TLC + 1 digits result.
  ASSERT_EQ(mp_lsb_mult(scalars1, scalars2, res_scalars_wide), cudaSuccess);
  std::cout << "first GPU lsb mult output  = 0x";
  for (int i = 0; i < 2 * scalar_field_t::TLC; i++) {
    std::cout << std::hex << res_scalars_wide[0].limbs_storage.limbs[i];
  }
  std::cout << std::endl;

  ASSERT_EQ(mp_mult(scalars1, scalars2, res_scalars_wide_full), cudaSuccess);
  std::cout << "first GPU full mult output = 0x";
  for (int i = 0; i < 2 * scalar_field_t::TLC; i++) {
    std::cout << std::hex << res_scalars_wide_full[0].limbs_storage.limbs[i];
  }
  std::cout << std::endl;
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < scalar_field_t::TLC + 1; i++) {
      ASSERT_EQ(res_scalars_wide_full[j].limbs_storage.limbs[i], res_scalars_wide[j].limbs_storage.limbs[i]);
    }
  }
}

TEST_F(PrimitivesTest, MP_MSB_MULT)
{
  // MSB multiply, take n msb bits of multiplication, assert that the error is up to 1.
  ASSERT_EQ(mp_msb_mult(scalars1, scalars2, res_scalars_wide), cudaSuccess);
  std::cout << "first GPU msb mult output  = 0x";
  for (int i = 2 * scalar_field_t::TLC - 1; i >= 0; i--) {
    std::cout << std::hex << res_scalars_wide[0].limbs_storage.limbs[i] << " ";
  }
  std::cout << std::endl;

  ASSERT_EQ(mp_mult(scalars1, scalars2, res_scalars_wide_full), cudaSuccess);
  std::cout << "first GPU full mult output = 0x";
  for (int i = 2 * scalar_field_t::TLC - 1; i >= 0; i--) {
    std::cout << std::hex << res_scalars_wide_full[0].limbs_storage.limbs[i] << " ";
  }

  std::cout << std::endl;

  for (int i = 0; i < 2 * scalar_field_t::TLC - 1; i++) {
    if (res_scalars_wide_full[0].limbs_storage.limbs[i] == res_scalars_wide[0].limbs_storage.limbs[i])
      std::cout << "matched word idx = " << i << std::endl;
  }
}

TEST_F(PrimitivesTest, INGO_MP_MULT)
{
  // MSB multiply, take n msb bits of multiplication, assert that the error is up to 1.
  ASSERT_EQ(ingo_mp_mult(scalars1, scalars2, res_scalars_wide), cudaSuccess);
  std::cout << "INGO   = 0x";
  for (int i = 0; i < 2 * scalar_field_t::TLC; i++) {
    std::cout << std::hex << res_scalars_wide[0].limbs_storage.limbs[i] << " ";
  }
  std::cout << std::endl;

  ASSERT_EQ(mp_mult(scalars1, scalars2, res_scalars_wide_full), cudaSuccess);
  std::cout << "ZKSYNC = 0x";
  for (int i = 0; i < 2 * scalar_field_t::TLC; i++) {
    std::cout << std::hex << res_scalars_wide_full[0].limbs_storage.limbs[i] << " ";
  }

  std::cout << std::endl;

  for (int i = 0; i < 2 * scalar_field_t::TLC - 1; i++) {
    if (res_scalars_wide_full[0].limbs_storage.limbs[i] == res_scalars_wide[0].limbs_storage.limbs[i])
      std::cout << "matched word idx = " << i << std::endl;
  }
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < 2 * scalar_field_t::TLC - 1; i++) {
      ASSERT_EQ(res_scalars_wide_full[j].limbs_storage.limbs[i], res_scalars_wide[j].limbs_storage.limbs[i]);
    }
  }
}

TEST_F(PrimitivesTest, INGO_MP_MSB_MULT)
{
  // MSB multiply, take n msb bits of multiplication, assert that the error is up to 1.
  ASSERT_EQ(ingo_mp_msb_mult(scalars1, scalars2, res_scalars_wide, n), cudaSuccess);
  std::cout << "INGO MSB   = 0x";
  for (int i = 2 * scalar_field_t::TLC - 1; i >= 0; i--) {
    std::cout << std::hex << res_scalars_wide[0].limbs_storage.limbs[i] << " ";
  }
  std::cout << std::endl;

  ASSERT_EQ(mp_mult(scalars1, scalars2, res_scalars_wide_full), cudaSuccess);
  std::cout << "ZKSYNC = 0x";
  for (int i = 2 * scalar_field_t::TLC - 1; i >= 0; i--) {
    std::cout << std::hex << res_scalars_wide_full[0].limbs_storage.limbs[i] << " ";
  }

  std::cout << std::endl;

  // for (int i=scalar_field::TLC; i < 2*scalar_field::TLC - 1; i++)
  // {
  //   ASSERT_EQ(in_bound, true);
  // }
  // for (int j=0; j<n; j++)
  // {
  //   for (int i=0; i < 2*scalar_field::TLC - 1; i++)
  //   {
  //     ASSERT_EQ(res_scalars_wide_full[j].limbs_storage.limbs[i], res_scalars_wide[j].limbs_storage.limbs[i]);
  //   }
  // }
  // mp testing
  mp::int1024_t scalar_1_mp = 0;
  mp::int1024_t scalar_2_mp = 0;
  mp::int1024_t res_mp = 0;
  mp::int1024_t res_gpu = 0;
  uint32_t num_limbs = scalar_field_t::TLC;

  for (int j = 0; j < n; j++) {
    uint32_t* scalar1_limbs = scalars1[j].limbs_storage.limbs;
    uint32_t* scalar2_limbs = scalars2[j].limbs_storage.limbs;
    scalar_1_mp = convert_to_boost_mp(scalar1_limbs, num_limbs);
    scalar_2_mp = convert_to_boost_mp(scalar2_limbs, num_limbs);
    res_mp = scalar_1_mp * scalar_2_mp;
    res_mp = res_mp >> (num_limbs * 32);
    res_gpu = convert_to_boost_mp(&(res_scalars_wide[j]).limbs_storage.limbs[num_limbs], num_limbs);
    std::cout << "res  mp = " << res_mp << std::endl;
    std::cout << "res gpu = " << res_gpu << std::endl;
    std::cout << "error = " << res_mp - res_gpu << std::endl;
    bool upper_bound = res_gpu <= res_mp;
    bool lower_bound = res_gpu > (res_mp - num_limbs);
    bool in_bound = upper_bound && lower_bound;

    ASSERT_EQ(in_bound, true);
  }
}

TEST_F(PrimitivesTest, INGO_MP_MOD_MULT)
{
  std::cout << " taking num limbs " << std::endl;
  uint32_t num_limbs = scalar_field_t::TLC;
  std::cout << " calling gpu... = " << std::endl;
  ASSERT_EQ(ingo_mp_mod_mult(scalars1, scalars2, res_scalars1, n), cudaSuccess);
  std::cout << " gpu call done " << std::endl;
  // mp testing
  mp::int1024_t scalar_1_mp = 0;
  mp::int1024_t scalar_2_mp = 0;
  mp::int1024_t res_mp = 0;
  mp::int1024_t res_gpu = 0;
  mp::int1024_t p = convert_to_boost_mp(scalar_field_t::get_modulus().limbs, num_limbs);
  std::cout << " p = " << p << std::endl;

  for (int j = 0; j < n; j++) {
    uint32_t* scalar1_limbs = scalars1[j].limbs_storage.limbs;
    uint32_t* scalar2_limbs = scalars2[j].limbs_storage.limbs;
    scalar_1_mp = convert_to_boost_mp(scalar1_limbs, num_limbs);
    scalar_2_mp = convert_to_boost_mp(scalar2_limbs, num_limbs);
    // std::cout << " s1 = " << scalar_1_mp << std::endl;
    // std::cout << " s2 = " << scalar_2_mp << std::endl;
    res_mp = (scalar_1_mp * scalar_2_mp) % p;
    res_gpu = convert_to_boost_mp((res_scalars1[j]).limbs_storage.limbs, num_limbs);
    std::cout << "res  mp = " << res_mp << std::endl;
    std::cout << "res gpu = " << res_gpu << std::endl;
    std::cout << "error = " << res_mp - res_gpu << std::endl;
    ASSERT_EQ(res_gpu, res_mp);
  }
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

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
