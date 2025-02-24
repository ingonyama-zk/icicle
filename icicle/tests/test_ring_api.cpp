
#include "test_mod_arithmetic_api.h"

// Derive all ModArith tests and add ring specific tests here
template <typename T>
class RingTest : public ModArithTest<T>
{
};

using RingTestBase = ModArithTestBase;
TYPED_TEST_SUITE(RingTest, FTImplementations);

// Note: this is testing host arithmetic. Other tests against CPU backend should guarantee correct device arithmetic too
TYPED_TEST(RingTest, RingSanityTest)
{
  auto a = TypeParam::rand_host();
  auto b = TypeParam::rand_host();
  auto a_neg = TypeParam::neg(a);
  ASSERT_EQ(a + TypeParam::zero(), a);
  ASSERT_EQ(a + b - a, b);
  ASSERT_EQ(a + a_neg, TypeParam::zero());
  ASSERT_EQ(a * TypeParam::zero(), TypeParam::zero());
  ASSERT_EQ(a * TypeParam::from(2), a + a);

  TypeParam invertible_element = TypeParam::rand_host();
  while (!TypeParam::has_inverse(invertible_element)) {
    invertible_element = TypeParam::rand_host();
  }
  auto invertible_element_inv = TypeParam::inverse(invertible_element);
  ASSERT_EQ(invertible_element * a * invertible_element_inv, a);
  ASSERT_EQ(invertible_element * invertible_element_inv, TypeParam::one());
}

// TODO remove once all tests compile for RNS type
TEST_F(RingTestBase, RingSanityRNSTest)
{
  auto a = scalar_rns_t::rand_host();
  auto b = scalar_rns_t::rand_host();
  auto a_neg = scalar_rns_t::neg(a);
  ASSERT_EQ(a + scalar_rns_t::zero(), a);
  ASSERT_EQ(a + b - a, b);
  ASSERT_EQ(a + a_neg, scalar_rns_t::zero());
  ASSERT_EQ(a * scalar_rns_t::zero(), scalar_rns_t::zero());
  ASSERT_EQ(a * scalar_rns_t::from(2), a + a);

  scalar_rns_t invertible_element = scalar_rns_t::rand_host();
  while (!scalar_rns_t::has_inverse(invertible_element)) {
    invertible_element = scalar_rns_t::rand_host();
  }
  auto invertible_element_inv = scalar_rns_t::inverse(invertible_element);
  ASSERT_EQ(invertible_element * a * invertible_element_inv, a);
  ASSERT_EQ(invertible_element * invertible_element_inv, scalar_rns_t::one());
}

// TODO remove once all tests compile for RNS type
TEST_F(RingTestBase, RingSanityRNS)
{
  auto a = scalar_rns_t::one();
  auto b = scalar_rns_t::one();
  auto sum = a + b;
  auto mul = sum * sum;
  auto sub = mul - sum;
  auto minus_one = scalar_rns_t::zero() - scalar_rns_t::one();
  std::cout << "a=" << a << std::endl;
  std::cout << "b=" << b << std::endl;
  std::cout << "sum=a+b=" << sum << std::endl;
  std::cout << "mul=sum*sum=" << mul << std::endl;
  std::cout << "sub=mul-sum=" << sub << std::endl;
  std::cout << "minus_one=" << minus_one << std::endl;

  const int size = 1 << 22;
  // RNS
  auto rns_input_a = std::vector<scalar_rns_t>(size);
  auto rns_input_b = std::vector<scalar_rns_t>(size);
  auto rns_output = std::vector<scalar_rns_t>(size);

  scalar_rns_t::rand_host_many(rns_input_a.data(), size);
  scalar_rns_t::rand_host_many(rns_input_b.data(), size);
  START_TIMER(rns);
  for (int i = 0; i < size; ++i) {
    rns_output[i] = rns_input_a[i] * rns_input_b[i];
  }
  END_TIMER_AVERAGE(rns, "rns mult", true /*=enable*/, size);
  // END_TIMER(rns, "rns mult", true /*=enable*/);

  // DIRECT
  auto direct_input_a = std::vector<scalar_t>(size);
  auto direct_input_b = std::vector<scalar_t>(size);
  auto direct_output = std::vector<scalar_t>(size);

  scalar_t::rand_host_many(direct_input_a.data(), size);
  scalar_t::rand_host_many(direct_input_b.data(), size);
  START_TIMER(direct);
  for (int i = 0; i < size; ++i) {
    direct_output[i] = direct_input_a[i] * direct_input_b[i];
  }
  END_TIMER_AVERAGE(direct, "direct mult", true /*=enable*/, size);
  // END_TIMER(direct, "direct mult", true /*=enable*/);

  // TODO Yuval: convert rns to direct to compare
  static_assert(sizeof(scalar_rns_t) == sizeof(scalar_t), "RNS and direct scalar_t should have the same size");
  ASSERT_EQ(0, memcmp(rns_output.data(), direct_output.data(), size * sizeof(scalar_t)));
}