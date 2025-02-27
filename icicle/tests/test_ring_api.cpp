
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

TEST_F(RingTestBase, RingRnsConversion)
{
  // I need an invertible element here
  scalar_t r = scalar_t::rand_host();
  while (!scalar_t::has_inverse(r)) {
    r = scalar_t::rand_host();
  }

  // check r * r^-1 = 1 in rns and direct
  auto r_inv = scalar_t::inverse(r);
  ASSERT_EQ(r * r_inv, scalar_t::one());

  // convert direct to rns and check 'r * r^-1 = 1' in rns
  scalar_rns_t r_rns = scalar_rns_t::from_direct(r);
  scalar_rns_t r_inv_rns_converted = scalar_rns_t::from_direct(r_inv);
  scalar_rns_t r_inv_rns_computed = scalar_rns_t::inverse(r_rns);
  ASSERT_EQ(r_inv_rns_converted, r_inv_rns_computed);
  ASSERT_EQ(r_rns * r_inv_rns_converted, scalar_rns_t::one());
  ICICLE_LOG_INFO << "r=" << r << ", r_rns=" << r_rns << ", r_inv=" << r_inv << ", r_inv_rns=" << r_inv_rns_converted;

  // Constructor from direct
  scalar_rns_t r_rns_from_zq = r; // here we convert r to rns implicitly
  scalar_rns_t r_rns_from_zq_direct = scalar_rns_t::from_direct(r);
  ASSERT_EQ(r_rns_from_zq, r_rns_from_zq_direct);

  // Convert r in-place
  scalar_t r_backup = r;
  // convert r inplace
  scalar_rns_t& r_rns_casted = (scalar_rns_t&)r;
  scalar_rns_t::convert_direct_to_rns(&r.limbs_storage, &r_rns_casted.limbs_storage);
  ASSERT_EQ(r_rns_casted, r_rns);
  ICICLE_LOG_INFO << "r=" << r << ", r_rns=" << r_rns_casted;
  // convert rns back to direct
  ASSERT_NE(r, r_backup);
  scalar_rns_t::convert_rns_to_direct(&r_rns_casted.limbs_storage, &r.limbs_storage);
  ASSERT_EQ(r, r_backup);
}