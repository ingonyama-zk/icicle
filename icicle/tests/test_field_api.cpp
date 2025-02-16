
#include "test_ring_field.h"

// Add field specific tests here

// TODO Hadar: this is a workaround for 'storage<18 - scalar_t::TLC>' failing due to 17 limbs not supported.
//             It means we skip fields such as babybear!
#ifndef EXT_FIELD
TEST_F(RingAndFieldTestBase, FieldStorageReduceSanityTest)
{
  /*
  SR - storage reduce
  check that:
  1. SR(x1) + SR(x1) = SR(x1+x2)
  2. SR(INV(SR(x))*x) = 1
  */
  START_TIMER(StorageSanity)
  for (int i = 0; i < 1000; i++) {
    if constexpr (scalar_t::TLC == 1) {
      storage<18> a =                                          // 18 because we support up to 576 bits
        scalar_t::template rand_storage<18>(17);               // 17 so we don't have carry after addition
      storage<18> b = scalar_t::template rand_storage<18>(17); // 17 so we don't have carry after addition
      storage<18> sum = {};
      const storage<3> c =
        scalar_t::template rand_storage<3>(); // 3 because we don't support higher odd number of limbs yet
      storage<4> product = {};
      host_math::template add_sub_limbs<18, false, false, true>(a, b, sum);
      auto c_red = scalar_t::from(c);
      auto c_inv = scalar_t::inverse(c_red);
      host_math::multiply_raw<3, 1, true>(
        c, c_inv.limbs_storage, product); // using 32-bit multiplication for small fields
      ASSERT_EQ(scalar_t::from(a) + scalar_t::from(b), scalar_t::from(sum));
      ASSERT_EQ(scalar_t::from(product), scalar_t::one());
      std::byte* a_bytes = reinterpret_cast<std::byte*>(a.limbs);
      std::byte* b_bytes = reinterpret_cast<std::byte*>(b.limbs);
      std::byte* sum_bytes = reinterpret_cast<std::byte*>(sum.limbs);
      std::byte* product_bytes = reinterpret_cast<std::byte*>(product.limbs);
      ASSERT_EQ(scalar_t::from(a), scalar_t::from(a_bytes, 18 * 4));
      ASSERT_EQ(scalar_t::from(a_bytes, 18 * 4) + scalar_t::from(b_bytes, 18 * 4), scalar_t::from(sum_bytes, 18 * 4));
      ASSERT_EQ(scalar_t::from(product_bytes, 4 * 4), scalar_t::one());
    } else {
      storage<18> a =                                          // 18 because we support up to 576 bits
        scalar_t::template rand_storage<18>(17);               // 17 so we don't have carry after addition
      storage<18> b = scalar_t::template rand_storage<18>(17); // 17 so we don't have carry after addition
      storage<18> sum = {};
      const storage<18 - scalar_t::TLC> c =
        scalar_t::template rand_storage<18 - scalar_t::TLC>(); // -TLC so we don't overflow in multiplication
      storage<18> product = {};
      host_math::template add_sub_limbs<18, false, false, true>(a, b, sum);
      auto c_red = scalar_t::from(c);
      auto c_inv = scalar_t::inverse(c_red);
      host_math::multiply_raw(c, c_inv.limbs_storage, product);
      ASSERT_EQ(scalar_t::from(a) + scalar_t::from(b), scalar_t::from(sum));
      ASSERT_EQ(scalar_t::from(product), scalar_t::one());
      std::byte* a_bytes = reinterpret_cast<std::byte*>(a.limbs);
      std::byte* b_bytes = reinterpret_cast<std::byte*>(b.limbs);
      std::byte* sum_bytes = reinterpret_cast<std::byte*>(sum.limbs);
      std::byte* product_bytes = reinterpret_cast<std::byte*>(product.limbs);
      ASSERT_EQ(scalar_t::from(a), scalar_t::from(a_bytes, 18 * 4));
      ASSERT_EQ(scalar_t::from(a_bytes, 18 * 4) + scalar_t::from(b_bytes, 18 * 4), scalar_t::from(sum_bytes, 18 * 4));
      ASSERT_EQ(scalar_t::from(product_bytes, 18 * 4), scalar_t::one());
    }
  }
  END_TIMER(StorageSanity, "storage sanity", true);
}
#endif // ! EXT_FIELD