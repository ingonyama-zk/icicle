

#include "test_ring_field.h"

// Extending the common tests to ring specific tests

using Zq = scalar_t;

#include "icicle/fields/params_gen.h"

TYPED_TEST(RingAndFieldTest, RootsOfUnity)
{
  constexpr int logn = 24;
  // auto w = TypeParam::omega(logn);
  // ASSERT_EQ(TypeParam::pow(w, 1 << logn), TypeParam::one());

  constexpr int limbs_count = 2;
  // constexpr storage<2> modulus = {0x00000001, 0xffffffff}; // goldilocks
  constexpr storage<2> modulus = {0xf7000001, 0x3b880000}; // labrador
  constexpr unsigned modulus_bit_count =
    32 * (limbs_count - 1) + params_gen::floorlog2(modulus.limbs[limbs_count - 1]) + 1;
  ICICLE_LOG_INFO << "modulus_bit_count=" << modulus_bit_count;
  storage<limbs_count> m = params_gen::template get_m<limbs_count, 2 * modulus_bit_count>(modulus);

  unsigned num_of_reductions = params_gen::template num_of_reductions<limbs_count, 2 * modulus_bit_count>(modulus, m);
  ICICLE_LOG_INFO << "num_of_reductions=" << num_of_reductions;
  ICICLE_LOG_INFO << "m=" << scalar_t{field_config::zq_config::m};
  ICICLE_LOG_INFO << "slack_bits=" << scalar_t::slack_bits;
  ICICLE_LOG_INFO << "neg_mod=" << scalar_t{field_config::zq_config::neg_modulus};
  ICICLE_LOG_INFO << "mod=" << scalar_t{field_config::zq_config::modulus};
}