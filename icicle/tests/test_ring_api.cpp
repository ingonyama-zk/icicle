
#include "test_mod_arithmetic_api.h"
#include "icicle/balanced_decomposition.h"

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
  scalar_rns_t r_rns = scalar_rns_t::from_direct(r); // static method to convert direct to rns
  scalar_rns_t r_inv_rns_converted = scalar_rns_t::from_direct(r_inv);
  scalar_rns_t r_inv_rns_computed = scalar_rns_t::inverse(r_rns);
  ASSERT_EQ(r_inv_rns_converted, r_inv_rns_computed);
  ASSERT_EQ(r_rns * r_inv_rns_converted, scalar_rns_t::one());

  // Constructor from direct
  scalar_rns_t r_rns_from_zq = r; // here we convert r to rns implicitly by constructing from Zq type
  scalar_rns_t r_rns_from_zq_direct = scalar_rns_t::from_direct(r);
  ASSERT_EQ(r_rns_from_zq, r_rns_from_zq_direct);

  // Convert r in-place
  scalar_t r_backup = r;
  scalar_rns_t& r_rns_casted = (scalar_rns_t&)r;
  scalar_rns_t::convert_direct_to_rns(
    &r.limbs_storage, &r_rns_casted.limbs_storage); // convert using given memory, possibly inplace
  ASSERT_EQ(r_rns_casted, r_rns);
  // convert rns back to direct
  ASSERT_NE(r, r_backup);
  ASSERT_EQ(r_backup, r_rns_casted.to_direct()); // create a new Zq element from rns
  scalar_rns_t::convert_rns_to_direct(
    &r_rns_casted.limbs_storage, &r.limbs_storage); // convert using the given memory, possibly inplace
  ASSERT_EQ(r, r_backup);
}

TEST_F(RingTestBase, VectorRnsConversion)
{
  const size_t N = 1 << 16;
  auto direct_input = std::vector<scalar_t>(N);
  auto direct_output = std::vector<scalar_t>(N);
  auto rns_input = std::vector<scalar_rns_t>(N);
  auto rns_output = std::vector<scalar_rns_t>(N);

  scalar_t::rand_host_many(direct_input.data(), N);

  for (auto device : s_registered_devices) {
    ICICLE_CHECK(icicle_set_device(device));

    // (1) compute element-wise power in direct representation
    ICICLE_CHECK(
      vector_mul(direct_input.data(), direct_input.data(), N, VecOpsConfig{}, direct_output.data())); // Direct
    // (2) convert to rns and recompute in rns
    ICICLE_CHECK(convert_to_rns(direct_input.data(), N, VecOpsConfig{}, rns_input.data()));
    ICICLE_CHECK(vector_mul(rns_input.data(), rns_input.data(), N, VecOpsConfig{}, rns_output.data())); // RNS
    // (3) assert results are different
    ASSERT_NE(0, memcmp(rns_output.data(), direct_output.data(), sizeof(scalar_t) * N));
    // (4) convert back from rns (inplace) and compare to the direct output
    // Note that we convert in-place so the rns type remains but the underlying data is not rns anymore!
    ICICLE_CHECK(convert_from_rns(rns_output.data(), N, VecOpsConfig{}, (scalar_t*)rns_output.data()));
    ASSERT_EQ(0, memcmp(rns_output.data(), direct_output.data(), sizeof(scalar_t) * N));
  }
}

TEST_F(RingTestBase, BalancedDecomposition)
{
  static_assert(field_t::TLC == 2, "Decomposition assumes q ~64b");
  constexpr auto q_storage = field_t::get_modulus();
  const int64_t q = *(int64_t*)&q_storage; // Note this is valid since TLC == 2
  ICICLE_ASSERT(q > 0) << "Expecting at least one slack bit to use int64 arithmetic";

  const size_t size = 1 << 20;
  auto input = std::vector<field_t>(size);
  field_t::rand_host_many(input.data(), size);
  auto recomposed = std::vector<field_t>(size);

  const auto q_sqrt = static_cast<uint32_t>(std::sqrt(q));
  const auto bases = std::vector<uint32_t>{2, 3, 4, 27, 60, q_sqrt};

  for (auto device : s_registered_devices) {
    ICICLE_CHECK(icicle_set_device(device));

    field_t *d_input, *d_decomposed, *d_recomposed;
    ICICLE_CHECK(icicle_malloc((void**)&d_input, size * sizeof(field_t)));
    ICICLE_CHECK(icicle_malloc((void**)&d_recomposed, size * sizeof(field_t)));
    ICICLE_CHECK(icicle_copy(d_input, input.data(), size * sizeof(field_t)));

    auto cfg = VecOpsConfig{};
    cfg.is_a_on_device = true;
    cfg.is_result_on_device = true;

    for (const auto base : bases) {
      // Number of digits needed to represent an element mod q in balanced base-b representation
      const size_t digits_per_element = balanced_decomposition::compute_nof_digits<field_t>(base);
      const size_t decomposed_size = size * digits_per_element;
      auto decomposed = std::vector<field_t>(decomposed_size);

      std::stringstream timer_label_decompose, timer_label_recompose;
      timer_label_decompose << "Decomposition [device=" << device << ", base=" << base << "]";
      timer_label_recompose << "Recomposition [device=" << device << ", base=" << base << "]";

      ICICLE_CHECK(icicle_malloc((void**)&d_decomposed, decomposed_size * sizeof(field_t)));

      // Decompose into balanced digits
      START_TIMER(decomposition);
      ICICLE_CHECK(balanced_decomposition::decompose(d_input, size, base, cfg, d_decomposed, decomposed_size));
      END_TIMER(decomposition, timer_label_decompose.str().c_str(), true);

      // Verify that all digits lie in the correct balanced range (-b/2, b/2]
      ICICLE_CHECK(icicle_copy(decomposed.data(), d_decomposed, decomposed_size * sizeof(field_t)));
      for (size_t i = 0; i < decomposed_size; ++i) {
        const int64_t digit = *reinterpret_cast<int64_t*>(&decomposed[i]);

        // Since field_t wraps into [0, q), digits near q are actually negative
        const bool is_positive_digit = digit <= base / 2;
        const bool is_negative_digit = (base % 2 == 0) ? (q - digit) < base / 2 : (q - digit) <= base / 2;
        const bool is_balanced = is_positive_digit || is_negative_digit;

        ASSERT_TRUE(is_balanced) << "Digit " << digit << " is out of expected balanced range for base=" << base;
      }

      // Recompose and compare to original input
      START_TIMER(recomposition);
      ICICLE_CHECK(balanced_decomposition::recompose(d_decomposed, decomposed_size, base, cfg, d_recomposed, size));
      END_TIMER(recomposition, timer_label_recompose.str().c_str(), true);

      ICICLE_CHECK(icicle_copy(recomposed.data(), d_recomposed, size * sizeof(field_t)));
      ASSERT_EQ(0, memcmp(input.data(), recomposed.data(), sizeof(field_t) * size))
        << "Recomposition failed for base=" << base;

      icicle_free(d_decomposed);
    } // base loop

    icicle_free(d_input);
    icicle_free(d_recomposed);
  } // device loop
}

TEST_F(RingTestBase, BalancedDecompositionErrorCases)
{
  static_assert(field_t::TLC == 2, "Decomposition assumes q ~64b");
  constexpr auto q_storage = field_t::get_modulus();
  const int64_t q = *(int64_t*)&q_storage; // Note this is valid since TLC == 2
  ICICLE_ASSERT(q > 0) << "Expecting at least one slack bit to use int64 arithmetic";

  const size_t size = 1 << 10;
  auto input = std::vector<field_t>(size);
  field_t::rand_host_many(input.data(), size);
  auto recomposed = std::vector<field_t>(size);
  const auto bases = std::vector<uint32_t>{2, 4, 16, 179};

  for (auto device : s_registered_devices) {
    ICICLE_CHECK(icicle_set_device(device));

    const auto cfg = VecOpsConfig{};

    // Number of digits needed to represent an element mod q in balanced base-b representation
    const uint32_t base = rand_uint_32b();
    const size_t digits_per_element = balanced_decomposition::compute_nof_digits<field_t>(base);
    const size_t decomposed_size = size * digits_per_element;
    auto decomposed = std::vector<field_t>(decomposed_size);

    // (1) Error: output size too small
    ASSERT_NE(
      eIcicleError::SUCCESS,
      balanced_decomposition::decompose(input.data(), size, base, cfg, decomposed.data(), decomposed_size - 1));
    ASSERT_NE(
      eIcicleError::SUCCESS,
      balanced_decomposition::recompose(decomposed.data(), decomposed_size - 1, base, cfg, input.data(), size));

    field_t* nullptr_field_t = nullptr;
    // (2) Error: output is null
    ASSERT_NE(
      eIcicleError::SUCCESS,
      balanced_decomposition::decompose(input.data(), size, base, cfg, nullptr_field_t, decomposed_size));
    ASSERT_NE(
      eIcicleError::SUCCESS,
      balanced_decomposition::recompose(decomposed.data(), decomposed_size, base, cfg, nullptr_field_t, size));

    // (3) Error: input is null
    ASSERT_NE(
      eIcicleError::SUCCESS,
      balanced_decomposition::decompose(nullptr_field_t, size, base, cfg, decomposed.data(), decomposed_size));
    ASSERT_NE(
      eIcicleError::SUCCESS,
      balanced_decomposition::recompose(nullptr_field_t, decomposed_size, base, cfg, input.data(), size));

    // (4) Error: base is 1
    ASSERT_NE(
      eIcicleError::SUCCESS,
      balanced_decomposition::decompose(input.data(), size, 1 /*=base*/, cfg, decomposed.data(), decomposed_size));
    ASSERT_NE(
      eIcicleError::SUCCESS,
      balanced_decomposition::recompose(decomposed.data(), decomposed_size, 1 /*=base*/, cfg, input.data(), size));

  } // device loop
}
