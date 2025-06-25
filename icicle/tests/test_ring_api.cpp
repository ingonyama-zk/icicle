#include "test_mod_arithmetic_api.h"
#include "test_matrix_api.h"
#include "icicle/balanced_decomposition.h"
#include "icicle/jl_projection.h"
#include "icicle/norm.h"
#include "icicle/random_sampling.h"
#include "icicle/negacyclic_ntt.h"
#include "icicle/fields/field_config.h"
#include "icicle/fields/field.h"
#include <chrono>

using namespace field_config;
using namespace icicle;

using uint128_t = __uint128_t;

// Helper function for norm checking
static uint64_t abs_centered(uint64_t val, uint64_t q)
{
  if (val > q / 2) { val = q - val; }
  return val;
}

// Derive all ModArith tests and add ring specific tests here
template <typename T>
class RingTest : public ModArithTest<T>
{
};

// This function performs a negacyclic convolution multiplication
static PolyRing Rq_mul(const PolyRing& a, const PolyRing& b)
{
  PolyRing c;
  constexpr size_t degree = PolyRing::d;
  const Zq* a_zq = reinterpret_cast<const Zq*>(&a);
  const Zq* b_zq = reinterpret_cast<const Zq*>(&b);
  Zq* c_zq = reinterpret_cast<Zq*>(&c);
  // zero initialize c
  for (size_t k = 0; k < degree; ++k)
    c_zq[k] = Zq::zero();

  // Manual negacyclic convolution: c_k = sum_{i+j ≡ k mod n} a_i * b_j,
  // with negation when i+j >= n
  for (size_t i = 0; i < degree; ++i) {
    for (size_t j = 0; j < degree; ++j) {
      size_t ij = i + j;
      size_t k = ij % degree;
      Zq prod = a_zq[i] * b_zq[j];
      if (ij >= degree) {
        c_zq[k] = c_zq[k] - prod; // negacyclic
      } else {
        c_zq[k] = c_zq[k] + prod;
      }
    }
  }
  return c;
}

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
    // (4) convert back from rns (inplace) and compare to the the direct output
    // Note that we convert in-place so the rns type remains but the underlying data is not rns anymore!
    ICICLE_CHECK(convert_from_rns(rns_output.data(), N, VecOpsConfig{}, (scalar_t*)rns_output.data()));
    ASSERT_EQ(0, memcmp(rns_output.data(), direct_output.data(), sizeof(scalar_t) * N));
  }
}

TEST_F(RingTestBase, BalancedDecompositionZQ)
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

TEST_F(RingTestBase, BalancedDecompositionZqErrorCases)
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

// This test verifies that balanced decomposition of a PolyRing is implemented correctly by recomposing manually
TEST_F(RingTestBase, BalancedDecompositionPolyRing)
{
  static_assert(PolyRing::Base::TLC == 2, "Decomposition assumes q ~64-bit");

  // Get q from PolyRing::Base as signed 64-bit for safe arithmetic
  constexpr auto q_storage = PolyRing::Base::get_modulus();
  const int64_t q = *(int64_t*)&q_storage;
  ICICLE_ASSERT(q > 0) << "Expecting positive modulus q to allow int64 arithmetic";

  // Generate a random input polynomial over PolyRing
  constexpr size_t size = 7;
  std::vector<PolyRing> input_polynomials(size);
  PolyRing::rand_host_many(input_polynomials.data(), size);

  for (auto device : s_registered_devices) {
    ICICLE_CHECK(icicle_set_device(device));

    VecOpsConfig cfg{};
    const uint32_t q_sqrt = static_cast<uint32_t>(std::sqrt(q));
    const std::vector<uint32_t> bases = {2, 3, 16, (1 << 20) + 1, q_sqrt};

    for (uint32_t base : bases) {
      // Compute the number of digits for the given base
      const size_t num_digits = balanced_decomposition::compute_nof_digits<Zq>(base);
      std::vector<PolyRing> decomposed_polynomials(size * num_digits);

      // Perform balanced decomposition into digits. Output is digit-major
      ICICLE_CHECK(balanced_decomposition::decompose(
        input_polynomials.data(), input_polynomials.size(), base, cfg, decomposed_polynomials.data(),
        decomposed_polynomials.size()));

      // Recompose the original polynomial from digits
      std::vector<PolyRing> recomposed(size);
      memset(recomposed.data(), 0, sizeof(PolyRing) * size);
      // Generate repeated powers of base: each digit level gets the same base^i for all polynomials
      Zq power = Zq::from(1);
      std::vector<Zq> powers(size * num_digits);
      for (size_t digit_idx = 0; digit_idx < num_digits; ++digit_idx) {
        Zq current = power;
        for (size_t poly_idx = 0; poly_idx < size; ++poly_idx) {
          powers[digit_idx * size + poly_idx] = current;
        }
        power = power * Zq::from(base);
      }

      // Multiply each decomposed digit by its base power
      std::vector<PolyRing> scaled_digits(size * num_digits);
      ICICLE_CHECK(
        vector_mul(decomposed_polynomials.data(), powers.data(), size * num_digits, {}, scaled_digits.data()));

      // Accumulate scaled digits into recomposed result
      for (size_t digit_idx = 0; digit_idx < num_digits; ++digit_idx) {
        ICICLE_CHECK(vector_add(
          (const Zq*)recomposed.data(), (const Zq*)(scaled_digits.data() + digit_idx * size), size * PolyRing::d, {},
          (Zq*)recomposed.data()));
      }

      // Verify recomposed polynomial matches the original input
      ASSERT_EQ(0, memcmp(recomposed.data(), input_polynomials.data(), size * sizeof(PolyRing)));
    }
  }
}

// This test verifies that batch balanced decomposition and recomposition
// on device memory correctly reconstruct the original PolyRing polynomials.
// It also checks that the decomposition satisfies the L∞ bound.
TEST_F(RingTestBase, BalancedDecompositionPolyRingBatch)
{
  static_assert(PolyRing::Base::TLC == 2, "Decomposition assumes q ~64-bit");

  constexpr size_t degree = PolyRing::d;
  constexpr size_t size = 1 << 10; // Number of PolyRing polynomials

  // Get modulus q as signed integer for arithmetic safety
  constexpr auto q_storage = PolyRing::Base::get_modulus();
  const int64_t q = *(const int64_t*)&q_storage;
  ICICLE_ASSERT(q > 0) << "Expecting positive q to allow int64 arithmetic";

  // Generate random input polynomials over PolyRing
  std::vector<PolyRing> input(size);
  PolyRing::rand_host_many(input.data(), size);

  std::vector<PolyRing> recomposed(size);
  const std::vector<uint32_t> bases = {2, 3, 16, 155, 1024, static_cast<uint32_t>(std::sqrt(q))};

  for (auto device : s_registered_devices) {
    ICICLE_CHECK(icicle_set_device(device));

    PolyRing *d_input = nullptr, *d_decomposed = nullptr, *d_recomposed = nullptr;
    ICICLE_CHECK(icicle_malloc((void**)&d_input, size * sizeof(PolyRing)));
    ICICLE_CHECK(icicle_malloc((void**)&d_recomposed, size * sizeof(PolyRing)));
    ICICLE_CHECK(icicle_copy(d_input, input.data(), size * sizeof(PolyRing)));

    VecOpsConfig cfg{};
    cfg.is_a_on_device = true;
    cfg.is_result_on_device = true;

    for (uint32_t base : bases) {
      const size_t digits_per_coeff = balanced_decomposition::compute_nof_digits<Zq>(base);
      const size_t decomposed_size = size * digits_per_coeff;

      std::stringstream label_decompose, label_recompose;
      label_decompose << "PolyRing Decomposition [device=" << device << ", base=" << base << "]";
      label_recompose << "PolyRing Recomposition [device=" << device << ", base=" << base << "]";

      ICICLE_CHECK(icicle_malloc((void**)&d_decomposed, decomposed_size * sizeof(PolyRing)));

      // --- Step 1: Decomposition ---
      START_TIMER(decompose);
      ICICLE_CHECK(balanced_decomposition::decompose(d_input, size, base, cfg, d_decomposed, decomposed_size));
      END_TIMER(decompose, label_decompose.str().c_str(), true);

      // --- Step 2: Norm Bound Check (L∞) ---
      {
        VecOpsConfig norm_cfg{};
        norm_cfg.is_a_on_device = true;
        norm_cfg.batch_size = digits_per_coeff * size;

        std::vector<char> is_norm_bound(norm_cfg.batch_size, false);
        norm::check_norm_bound(
          reinterpret_cast<Zq*>(d_decomposed), degree, eNormType::LInfinity, base / 2 + 1, norm_cfg,
          reinterpret_cast<bool*>(is_norm_bound.data()));

        for (size_t i = 0; i < norm_cfg.batch_size; ++i) {
          ASSERT_TRUE(is_norm_bound[i]) << "Decomposed PolyRing polynomial " << i
                                        << " exceeds expected balanced range for base = " << base;
        }
      }

      // --- Step 3: Recomposition and Validation ---
      START_TIMER(recompose);
      ICICLE_CHECK(balanced_decomposition::recompose(d_decomposed, decomposed_size, base, cfg, d_recomposed, size));
      END_TIMER(recompose, label_recompose.str().c_str(), true);

      ICICLE_CHECK(icicle_copy(recomposed.data(), d_recomposed, size * sizeof(PolyRing)));

      ASSERT_EQ(0, memcmp(input.data(), recomposed.data(), size * sizeof(PolyRing)))
        << "Recomposition mismatch for base = " << base;

      icicle_free(d_decomposed);
    }

    icicle_free(d_input);
    icicle_free(d_recomposed);
  }
}

TEST_F(RingTestBase, JLProjectionTest)
{
  static_assert(field_t::TLC == 2, "Decomposition assumes q ~64b");
  constexpr auto q_storage = field_t::get_modulus();
  const int64_t q = *(int64_t*)&q_storage; // Note this is valid since TLC == 2
  ICICLE_ASSERT(q > 0) << "Expecting at least one slack bit to use int64 arithmetic";

  const size_t N = 1 << 16;       // Input vector size
  const size_t output_size = 256; // JL projected size
  const int max_trials = 10;      // JL projection output is bound by sqrt(128)*norm(input) with probability 0.5.
                                  // Therefore we allow repeating the check a few times.

  std::vector<field_t> input(N);
  std::vector<field_t> output(output_size);

  // generate random values in [0, sqrt(q)]. We assume input is low norm. Otherwise we may wrap around and the JL
  // lemma won't hold.
  const int64_t sqrt_q = static_cast<int64_t>(std::sqrt(q));
  for (auto& x : input) {
    uint64_t val = rand_uint_32b() % (sqrt_q + 1); // uniform in [0, sqrt_q]
    x = field_t::from(val);
  }

  auto norm_squared = [&](const std::vector<field_t>& v) -> double {
    double sum = 0.0;
    for (const auto& x : v) {
      const int64_t val = *reinterpret_cast<const int64_t*>(&x);
      const int64_t centered = (val > q / 2) ? val - q : val; // Convert to signed representative
      sum += static_cast<double>(centered) * static_cast<double>(centered);
    }
    return sum;
  };

  const double input_norm = std::sqrt(norm_squared(input));
  ASSERT_GT(input_norm, 0.0) << "Input norm is zero, invalid for JL test";

  const auto cfg = VecOpsConfig{};
  for (auto device : s_registered_devices) {
    if (device != "CPU") continue; // TODO implement for CUDA too
    ICICLE_CHECK(icicle_set_device(device));
    std::stringstream timer_label;
    timer_label << "JL-projection [device=" << device << "]";

    bool passed = false;
    for (int trial = 0; trial < max_trials; ++trial) {
      std::byte seed[32];
      for (auto& b : seed) {
        b = static_cast<std::byte>(rand_uint_32b() % 256);
      }

      START_TIMER(projection);
      ICICLE_CHECK(jl_projection(input.data(), input.size(), seed, sizeof(seed), cfg, output.data(), output.size()));
      END_TIMER(projection, timer_label.str().c_str(), true);

      const double output_norm = std::sqrt(norm_squared(output));
      ASSERT_GT(output_norm, 0.0) << "JL projection output norm is zero (trial " << trial << ")";

      const double bound = std::sqrt(128.0) * input_norm;
      passed = (output_norm <= bound);

      ICICLE_LOG_INFO << "Input norm = " << input_norm << ", Output norm = " << output_norm
                      << ", Ratio = " << (output_norm / input_norm) << ", Bound = " << bound
                      << ", Passed = " << (passed ? "true" : "false");
      if (passed) break;
    }

    ASSERT_TRUE(passed) << "JL projection norm exceeded sqrt(128)*input_norm in all " << max_trials << " trials";
  }
}

TEST_F(RingTestBase, JLprojectionGetRowsTest)
{
  const size_t N = 1 << 10;       // Input vector size
  const size_t output_size = 256; // Number of JL projection rows

  std::vector<field_t> input(N, field_t::one()); // Input vector: all ones
  std::vector<field_t> projected(output_size);   // Output from jl_projection
  std::vector<field_t> matrix(output_size * N);  // Raw JL matrix rows (row-major)
  std::vector<field_t> expected(output_size);    // Expected output computed via matrix row sums

  std::byte seed[32];
  for (auto& b : seed) {
    b = static_cast<std::byte>(rand_uint_32b() % 256);
  }

  const auto cfg = VecOpsConfig{};

  for (const auto& device : s_registered_devices) {
    if (device != "CPU") continue; // TODO: Extend to CUDA

    ICICLE_CHECK(icicle_set_device(device));

    std::stringstream projection_timer_label, generate_timer_label;
    projection_timer_label << "JL-projection [device=" << device << "]";
    generate_timer_label << "JL-generate [device=" << device << "]";

    // Step 1: Compute projection via JL API
    START_TIMER(projection);
    ICICLE_CHECK(jl_projection(input.data(), N, seed, sizeof(seed), cfg, projected.data(), output_size));
    END_TIMER(projection, projection_timer_label.str().c_str(), true);

    // Step 2: Generate JL matrix rows explicitly
    START_TIMER(generate);
    ICICLE_CHECK(get_jl_matrix_rows(
      seed, sizeof(seed),
      N,           // row_size = input dimension
      0,           // start_row
      output_size, // num_rows
      cfg,
      matrix.data() // Output: [num_rows x row_size]
      ));
    END_TIMER(generate, generate_timer_label.str().c_str(), true);

    // Step 3: Since input = {1,1,...,1}, matrix-vector product is just summing each row
    VecOpsConfig sum_cfg{};
    sum_cfg.batch_size = output_size;
    ICICLE_CHECK(vector_sum(matrix.data(), N, sum_cfg, expected.data()));

    // Step 4: Compare expected vs projected
    for (size_t i = 0; i < output_size; ++i) {
      ASSERT_EQ(projected[i], expected[i])
        << "Mismatch at output[" << i << "]: projected = " << projected[i] << ", expected = " << expected[i];
    }
  }
}

// This test verifies the JL-projection lemma: projecting an input vector of Rq polynomials
// via Zq yields the same value as the constant term of an inner product in Rq with conjugated rows.
TEST_F(RingTestBase, JLprojectionLemma)
{
  const size_t input_size = 8;        // Number of Rq polynomials in the input
  const size_t projected_size = 16;   // Number of projected output values
  const size_t d = PolyRing::d;       // Degree of each Rq polynomial
  const size_t row_size = input_size; // Each JL row is composed of `input_size` Rq polynomials

  // Randomize input polynomials
  std::vector<PolyRing> input(input_size);
  PolyRing::rand_host_many(input.data(), input_size);

  // Prepare random seed
  std::byte seed[32];
  for (auto& b : seed) {
    b = static_cast<std::byte>(rand_uint_32b() & 0xFF);
  }

  for (const auto& device : s_registered_devices) {
    if (device == "CUDA") continue; // TODO: implement CUDA backend
    ICICLE_CHECK(icicle_set_device(device));

    // Project using flat Zq view (as if input is Zq vector)
    std::vector<field_t> projected(projected_size);
    ICICLE_CHECK(jl_projection(
      reinterpret_cast<const Zq*>(input.data()), input_size * d, seed, sizeof(seed), {}, projected.data(),
      projected_size));

    // Check the JL-lemma
    for (size_t row_idx = 0; row_idx < projected_size; ++row_idx) {
      std::vector<PolyRing> jl_row_conj(row_size);

      // Generate JL matrix row as Rq polynomials, with conjugation
      ICICLE_CHECK(get_jl_matrix_rows<PolyRing>(
        seed, sizeof(seed), row_size, row_idx, 1, /* 1 row */
        true /* conjugate */, {}, jl_row_conj.data()));

      // compute inner product in Rq domain. Note that we avoid negacyclic-NTT to avoid unnecessary dependency
      PolyRing inner_product_ntt = {0};
      for (size_t i = 0; i < row_size; ++i) {
        inner_product_ntt = inner_product_ntt + Rq_mul(input[i], jl_row_conj[i]);
      }

      // Validate that the constant term equals the Zq projection result
      const field_t constant_term = inner_product_ntt.values[0];
      EXPECT_EQ(constant_term, projected[row_idx]) << "Mismatch at row " << row_idx;
    }
  }
}

TEST_F(RingTestBase, NormBounded)
{
  static_assert(field_t::TLC == 2, "Norm checking assumes q ~64b");
  constexpr auto q_storage = field_t::get_modulus();
  const int64_t q = *(int64_t*)&q_storage; // Note this is valid since TLC == 2
  ICICLE_ASSERT(q > 0) << "Expecting at least one slack bit to use int64 arithmetic";
  auto square_root = static_cast<uint32_t>(std::sqrt(q));

  const size_t size = 1 << 10;
  auto input = std::vector<field_t>(size);

  for (size_t i = 0; i < size; ++i) {
    int32_t val = rand_uint_32b();
    if (val > square_root) { val = val % square_root; }
    input[i] = field_t::from(val);
  }

  for (auto device : s_registered_devices) {
    ICICLE_CHECK(icicle_set_device(device));

    bool output;

    // Test L2 norm
    {
      uint128_t actual_norm_squared = 0;
      for (size_t i = 0; i < size; ++i) {
        int64_t val = abs_centered(*(int64_t*)&input[i], q);
        actual_norm_squared += static_cast<uint64_t>(val) * static_cast<uint64_t>(val);
      }

      uint64_t bound = static_cast<uint64_t>(std::sqrt(actual_norm_squared)) + 1;
      ICICLE_CHECK(norm::check_norm_bound(input.data(), size, eNormType::L2, bound, VecOpsConfig{}, &output));
      ASSERT_TRUE(output) << "L2 norm check failed with bound " << bound << " on device " << device;

      bound = static_cast<uint64_t>(std::sqrt(actual_norm_squared)) - 1;
      ICICLE_CHECK(norm::check_norm_bound(input.data(), size, eNormType::L2, bound, VecOpsConfig{}, &output));
      ASSERT_FALSE(output) << "L2 norm check should fail with bound " << bound << " on device " << device;
    }

    // Test L-infinity norm
    {
      // Compute actual L-infinity norm
      uint64_t actual_norm = 0;
      for (size_t i = 0; i < size; ++i) {
        uint64_t val = abs_centered(*(int64_t*)&input[i], q);
        actual_norm = std::max(actual_norm, val);
      }

      // Test with bound just above actual norm
      uint64_t bound = actual_norm + 1;
      ICICLE_CHECK(norm::check_norm_bound(input.data(), size, eNormType::LInfinity, bound, VecOpsConfig{}, &output));
      ASSERT_TRUE(output) << "L-infinity norm check failed with bound " << bound << " on device " << device;

      // Test with bound just below actual norm
      bound = actual_norm - 1;
      ICICLE_CHECK(norm::check_norm_bound(input.data(), size, eNormType::LInfinity, bound, VecOpsConfig{}, &output));
      ASSERT_FALSE(output) << "L-infinity norm check should fail with bound " << bound << " on device " << device;
    }

    // Test error cases
    {
      field_t* nullptr_field_t = nullptr;
      bool* nullptr_bool = nullptr;

      // Test null input
      ASSERT_NE(
        eIcicleError::SUCCESS,
        norm::check_norm_bound(nullptr_field_t, size, eNormType::L2, 100, VecOpsConfig{}, &output));

      // Test null output
      ASSERT_NE(
        eIcicleError::SUCCESS,
        norm::check_norm_bound(input.data(), size, eNormType::L2, 100, VecOpsConfig{}, nullptr_bool));

      // Test zero size
      ASSERT_NE(
        eIcicleError::SUCCESS, norm::check_norm_bound(input.data(), 0, eNormType::L2, 100, VecOpsConfig{}, &output));

      // Test with values exceeding sqrt(q)
      auto invalid_input = std::vector<field_t>(size);
      invalid_input[0] = field_t::from(square_root + 1);
      ASSERT_NE(
        eIcicleError::SUCCESS,
        norm::check_norm_bound(invalid_input.data(), size, eNormType::L2, 100, VecOpsConfig{}, &output));
    }
  }
}

TEST_F(RingTestBase, NormRelative)
{
  static_assert(field_t::TLC == 2, "Norm checking assumes q ~64b");
  constexpr auto q_storage = field_t::get_modulus();
  const int64_t q = *(int64_t*)&q_storage; // Note this is valid since TLC == 2
  ICICLE_ASSERT(q > 0) << "Expecting at least one slack bit to use int64 arithmetic";

  auto square_root = static_cast<uint32_t>(std::sqrt(q));

  const size_t size = 1 << 10;
  auto input_a = std::vector<field_t>(size);
  auto input_b = std::vector<field_t>(size);

  for (size_t i = 0; i < size; ++i) {
    int32_t val_a = rand_uint_32b() % (square_root / 4);
    int32_t val_b = rand_uint_32b() % square_root;
    input_a[i] = field_t::from(val_a);
    input_b[i] = field_t::from(val_b);
  }

  for (auto device : s_registered_devices) {
    ICICLE_CHECK(icicle_set_device(device));

    bool output;

    // Test L2 norm
    {
      uint128_t norm_a_squared = 0;
      uint128_t norm_b_squared = 0;

      for (size_t i = 0; i < size; ++i) {
        int64_t val_a = abs_centered(*(int64_t*)&input_a[i], q);
        int64_t val_b = abs_centered(*(int64_t*)&input_b[i], q);
        norm_a_squared += static_cast<uint128_t>(val_a) * static_cast<uint128_t>(val_a);
        norm_b_squared += static_cast<uint128_t>(val_b) * static_cast<uint128_t>(val_b);
      }

      uint64_t passing_scale =
        static_cast<uint64_t>(std::sqrt(static_cast<double>(norm_a_squared) / static_cast<double>(norm_b_squared))) + 1;

      ICICLE_CHECK(norm::check_norm_relative(
        input_a.data(), input_b.data(), size, eNormType::L2, passing_scale, VecOpsConfig{}, &output));
      ASSERT_TRUE(output) << "L2 relative norm check failed with scale " << passing_scale << " on device " << device;

      uint64_t failing_scale = passing_scale - 1;
      ICICLE_CHECK(norm::check_norm_relative(
        input_a.data(), input_b.data(), size, eNormType::L2, failing_scale, VecOpsConfig{}, &output));
      ASSERT_FALSE(output) << "L2 relative norm check should fail with scale " << failing_scale << " on device "
                           << device;
    }

    // Test L-infinity norm
    {
      int64_t max_abs_a = 0;
      int64_t max_abs_b = 0;

      for (size_t i = 0; i < size; ++i) {
        int64_t val_a = abs_centered(*(int64_t*)&input_a[i], q);
        int64_t val_b = abs_centered(*(int64_t*)&input_b[i], q);
        max_abs_a = std::max(max_abs_a, val_a);
        max_abs_b = std::max(max_abs_b, val_b);
      }

      // Calculate scale that should make the check pass
      uint64_t passing_scale =
        static_cast<uint64_t>(static_cast<double>(max_abs_a) / static_cast<double>(max_abs_b)) + 1;

      // Test with scale that should pass
      ICICLE_CHECK(norm::check_norm_relative(
        input_a.data(), input_b.data(), size, eNormType::LInfinity, passing_scale, VecOpsConfig{}, &output));
      ASSERT_TRUE(output) << "L-infinity relative norm check failed with scale " << passing_scale << " on device "
                          << device;

      // Test with scale that should fail
      uint64_t failing_scale = passing_scale - 1;
      ICICLE_CHECK(norm::check_norm_relative(
        input_a.data(), input_b.data(), size, eNormType::LInfinity, failing_scale, VecOpsConfig{}, &output));
      ASSERT_FALSE(output) << "L-infinity relative norm check should fail with scale " << failing_scale << " on device "
                           << device;
    }

    // Test error cases
    {
      field_t* nullptr_field_t = nullptr;
      bool* nullptr_bool = nullptr;

      // Test null input_a
      ASSERT_NE(
        eIcicleError::SUCCESS,
        norm::check_norm_relative(nullptr_field_t, input_b.data(), size, eNormType::L2, 2, VecOpsConfig{}, &output));

      // Test null input_b
      ASSERT_NE(
        eIcicleError::SUCCESS,
        norm::check_norm_relative(input_a.data(), nullptr_field_t, size, eNormType::L2, 2, VecOpsConfig{}, &output));

      // Test null output
      ASSERT_NE(
        eIcicleError::SUCCESS, norm::check_norm_relative(
                                 input_a.data(), input_b.data(), size, eNormType::L2, 2, VecOpsConfig{}, nullptr_bool));

      // Test zero size
      ASSERT_NE(
        eIcicleError::SUCCESS,
        norm::check_norm_relative(input_a.data(), input_b.data(), 0, eNormType::L2, 2, VecOpsConfig{}, &output));

      // Test with values exceeding sqrt(q)
      auto invalid_input = std::vector<field_t>(size);
      invalid_input[0] = field_t::from(square_root + 1);
      ASSERT_NE(
        eIcicleError::SUCCESS,
        norm::check_norm_relative(
          invalid_input.data(), input_b.data(), size, eNormType::L2, 2, VecOpsConfig{}, &output));
      ASSERT_NE(
        eIcicleError::SUCCESS,
        norm::check_norm_relative(
          input_a.data(), invalid_input.data(), size, eNormType::L2, 2, VecOpsConfig{}, &output));
    }
  }
}

TEST_F(RingTestBase, NormBoundedBatch)
{
  static_assert(field_t::TLC == 2, "Norm checking assumes q ~64b");
  constexpr auto q_storage = field_t::get_modulus();
  const int64_t q = *(int64_t*)&q_storage; // Note this is valid since TLC == 2
  ICICLE_ASSERT(q > 0) << "Expecting at least one slack bit to use int64 arithmetic";

  auto square_root = static_cast<uint32_t>(std::sqrt(q));

  const size_t size = 1 << 10;
  const size_t batch_size = 4;

  auto input_a = std::vector<field_t>(size * batch_size);

  for (size_t i = 0; i < size * batch_size; ++i) {
    input_a[i] = field_t::from(rand_uint_32b() % square_root);
  }

  // Test L2 norm
  for (auto device : s_registered_devices) {
    ICICLE_CHECK(icicle_set_device(device));
    bool* output = new bool[batch_size];
    VecOpsConfig cfg = VecOpsConfig{};
    cfg.batch_size = batch_size;

    uint128_t* actual_norm = new uint128_t[batch_size];
    for (size_t i = 0; i < batch_size; ++i) {
      actual_norm[i] = 0;
      for (size_t j = 0; j < size; ++j) {
        int64_t val = abs_centered(*(int64_t*)&input_a[i * size + j], q);
        actual_norm[i] += static_cast<uint64_t>(val) * static_cast<uint64_t>(val);
      }
    }

    uint64_t max_bound = 0;
    uint64_t min_bound = std::numeric_limits<uint64_t>::max();
    for (size_t i = 0; i < batch_size; ++i) {
      max_bound = std::max(max_bound, static_cast<uint64_t>(std::sqrt(actual_norm[i])));
      min_bound = std::min(min_bound, static_cast<uint64_t>(std::sqrt(actual_norm[i])));
    }

    ICICLE_CHECK(norm::check_norm_bound(input_a.data(), size, eNormType::L2, max_bound + 1, cfg, output));

    for (size_t i = 0; i < batch_size; ++i) {
      ASSERT_TRUE(output[i]) << "L2 norm check should pass for batch " << i << " on device " << device;
    }

    ICICLE_CHECK(norm::check_norm_bound(input_a.data(), size, eNormType::L2, min_bound - 1, cfg, output));

    for (size_t i = 0; i < batch_size; ++i) {
      ASSERT_FALSE(output[i]) << "L2 norm check should fail for batch " << i << " on device " << device;
    }

    delete[] output;
  }

  // Test L-infinity norm
  for (auto device : s_registered_devices) {
    ICICLE_CHECK(icicle_set_device(device));
    bool* output = new bool[batch_size];
    VecOpsConfig cfg = VecOpsConfig{};
    cfg.batch_size = batch_size;

    uint64_t* actual_max_abs = new uint64_t[batch_size];
    for (size_t i = 0; i < batch_size; ++i) {
      actual_max_abs[i] = 0;
      for (size_t j = 0; j < size; ++j) {
        int64_t val = abs_centered(*(int64_t*)&input_a[i * size + j], q);
        actual_max_abs[i] = std::max(actual_max_abs[i], static_cast<uint64_t>(val));
      }
    }

    uint64_t max_bound = 0;
    uint64_t min_bound = std::numeric_limits<uint64_t>::max();
    for (size_t i = 0; i < batch_size; ++i) {
      max_bound = std::max(max_bound, actual_max_abs[i]);
      min_bound = std::min(min_bound, actual_max_abs[i]);
    }

    ICICLE_CHECK(norm::check_norm_bound(input_a.data(), size, eNormType::LInfinity, max_bound + 1, cfg, output));

    for (size_t i = 0; i < batch_size; ++i) {
      ASSERT_TRUE(output[i]) << "L-infinity norm check should pass for batch " << i << " on device " << device;
    }

    ICICLE_CHECK(norm::check_norm_bound(input_a.data(), size, eNormType::LInfinity, min_bound - 1, cfg, output));

    for (size_t i = 0; i < batch_size; ++i) {
      ASSERT_FALSE(output[i]) << "L-infinity norm check should fail for batch " << i << " on device " << device;
    }

    delete[] output;
    delete[] actual_max_abs;
  }
}

TEST_F(RingTestBase, NormRelativeBatch)
{
  static_assert(field_t::TLC == 2, "Norm checking assumes q ~64b");
  constexpr auto q_storage = field_t::get_modulus();
  const int64_t q = *(int64_t*)&q_storage; // Note this is valid since TLC == 2
  ICICLE_ASSERT(q > 0) << "Expecting at least one slack bit to use int64 arithmetic";

  auto square_root = static_cast<uint32_t>(std::sqrt(q));

  const size_t size = 1 << 10;
  const size_t batch_size = 4;

  auto input_a = std::vector<field_t>(size * batch_size);
  auto input_b = std::vector<field_t>(size * batch_size);

  for (size_t i = 0; i < size * batch_size; ++i) {
    input_a[i] = field_t::from(rand_uint_32b() % square_root);
    input_b[i] = field_t::from(rand_uint_32b() % square_root);
  }

  // Test L2 norm
  for (auto device : s_registered_devices) {
    ICICLE_CHECK(icicle_set_device(device));
    bool* output = new bool[batch_size];
    VecOpsConfig cfg = VecOpsConfig{};
    cfg.batch_size = batch_size;

    uint128_t* norm_a_squared = new uint128_t[batch_size];
    uint128_t* norm_b_squared = new uint128_t[batch_size];

    for (size_t i = 0; i < batch_size; ++i) {
      norm_a_squared[i] = 0;
      norm_b_squared[i] = 0;
      for (size_t j = 0; j < size; ++j) {
        int64_t val_a = abs_centered(*(int64_t*)&input_a[i * size + j], q);
        int64_t val_b = abs_centered(*(int64_t*)&input_b[i * size + j], q);
        norm_a_squared[i] += static_cast<uint128_t>(val_a) * static_cast<uint128_t>(val_a);
        norm_b_squared[i] += static_cast<uint128_t>(val_b) * static_cast<uint128_t>(val_b);
      }
    }

    uint128_t* passing_scale = new uint128_t[batch_size];

    for (size_t i = 0; i < batch_size; ++i) {
      passing_scale[i] = static_cast<uint128_t>(
                           std::sqrt(static_cast<double>(norm_a_squared[i]) / static_cast<double>(norm_b_squared[i]))) +
                         1;
    }

    uint128_t max_bound = passing_scale[0];
    uint128_t min_bound = passing_scale[0];
    for (size_t i = 1; i < batch_size; ++i) {
      max_bound = std::max(max_bound, passing_scale[i]);
      min_bound = std::min(min_bound, passing_scale[i]);
    }

    ICICLE_CHECK(
      norm::check_norm_relative(input_a.data(), input_b.data(), size, eNormType::L2, max_bound + 1, cfg, output));

    for (size_t i = 0; i < batch_size; ++i) {
      ASSERT_TRUE(output[i]) << "L2 relative norm check should pass for batch " << i << " on device " << device;
    }

    ICICLE_CHECK(
      norm::check_norm_relative(input_a.data(), input_b.data(), size, eNormType::L2, min_bound - 1, cfg, output));

    for (size_t i = 0; i < batch_size; ++i) {
      ASSERT_FALSE(output[i]) << "L2 relative norm check should fail for batch " << i << " on device " << device;
    }
  }

  // Test L-infinity norm
  for (auto device : s_registered_devices) {
    ICICLE_CHECK(icicle_set_device(device));
    bool* output = new bool[batch_size];
    VecOpsConfig cfg = VecOpsConfig{};
    cfg.batch_size = batch_size;

    uint64_t* max_abs_a = new uint64_t[batch_size];
    uint64_t* max_abs_b = new uint64_t[batch_size];

    for (size_t i = 0; i < batch_size; ++i) {
      max_abs_a[i] = 0;
      max_abs_b[i] = 0;
      for (size_t j = 0; j < size; ++j) {
        int64_t val_a = abs_centered(*(int64_t*)&input_a[i * size + j], q);
        int64_t val_b = abs_centered(*(int64_t*)&input_b[i * size + j], q);
        max_abs_a[i] = std::max(max_abs_a[i], static_cast<uint64_t>(val_a));
        max_abs_b[i] = std::max(max_abs_b[i], static_cast<uint64_t>(val_b));
      }
    }

    uint64_t* passing_scale = new uint64_t[batch_size];

    for (size_t i = 0; i < batch_size; ++i) {
      passing_scale[i] = static_cast<uint64_t>(max_abs_a[i]) / static_cast<uint64_t>(max_abs_b[i]) + 1;
    }

    uint64_t max_bound = passing_scale[0];
    uint64_t min_bound = passing_scale[0];
    for (size_t i = 1; i < batch_size; ++i) {
      max_bound = std::max(max_bound, passing_scale[i]);
      min_bound = std::min(min_bound, passing_scale[i]);
    }

    ICICLE_CHECK(norm::check_norm_relative(
      input_a.data(), input_b.data(), size, eNormType::LInfinity, max_bound + 1, cfg, output));

    for (size_t i = 0; i < batch_size; ++i) {
      ASSERT_TRUE(output[i]) << "L-infinity relative norm check should pass for batch " << i << " on device " << device;
    }

    ICICLE_CHECK(norm::check_norm_relative(
      input_a.data(), input_b.data(), size, eNormType::LInfinity, min_bound - 1, cfg, output));

    for (size_t i = 0; i < batch_size; ++i) {
      ASSERT_FALSE(output[i]) << "L-infinity relative norm check should fail for batch " << i << " on device "
                              << device;
    }
  }
}

#ifdef NTT
TEST_F(RingTestBase, NegacyclicNTT)
{
  int size = 1 << 15;
  std::vector<PolyRing> a(size);
  std::vector<PolyRing> b(size);
  PolyRing::rand_host_many(a.data(), size);
  PolyRing::rand_host_many(b.data(), size);

  for (auto device : s_registered_devices) {
    ICICLE_CHECK(icicle_set_device(device));

    // Dummy NTT to initialize NTT domain for this device (first call per device)
    PolyRing dummy;
    ICICLE_CHECK(ntt(&dummy, 1, NTTDir::kForward, NegacyclicNTTConfig{}, &dummy));

    std::vector<PolyRing> res(size);

    std::stringstream timer_label;
    timer_label << "Rq multiplication via NTT [device=" << device << "]";
    START_TIMER(RqMul);

    // Forward NTT: Rq → Tq
    ICICLE_CHECK(ntt(a.data(), size, NTTDir::kForward, NegacyclicNTTConfig{}, a.data()));
    ICICLE_CHECK(ntt(b.data(), size, NTTDir::kForward, NegacyclicNTTConfig{}, b.data()));

    // Pointwise multiplication in NTT domain
    ICICLE_CHECK(vector_mul(a.data(), b.data(), size, VecOpsConfig{}, res.data()));

    // Inverse NTT: Tq → Rq
    ICICLE_CHECK(ntt(res.data(), size, NTTDir::kInverse, NegacyclicNTTConfig{}, res.data()));

    END_TIMER(RqMul, timer_label.str().c_str(), true);

    // Convert a, b back to coefficient domain (in-place inverse NTT)
    ICICLE_CHECK(ntt(a.data(), size, NTTDir::kInverse, NegacyclicNTTConfig{}, a.data()));
    ICICLE_CHECK(ntt(b.data(), size, NTTDir::kInverse, NegacyclicNTTConfig{}, b.data()));

    // Verify correctness
    for (int i = 0; i < size; ++i) {
      PolyRing expected = Rq_mul(a[i], b[i]);
      EXPECT_EQ(0, memcmp(&expected, &res[i], sizeof(PolyRing)));
    }
  }
}
#endif // NTT

TEST_F(RingTestBase, RandomSampling)
{
  size_t size = 1 << 20;
  size_t seed_len = 32;
  std::vector<std::byte> seed(seed_len);
  for (size_t i = 0; i < seed_len; ++i) {
    seed[i] = static_cast<std::byte>(rand_uint_32b());
  }
  std::vector<std::byte> seed_prime(seed);
  seed_prime[0] = static_cast<std::byte>(uint8_t(seed_prime[0]) + 1); // Make sure the seed is different

  std::vector<std::vector<field_t>> a(s_registered_devices.size());
  std::vector<std::vector<field_t>> b(s_registered_devices.size());
  for (size_t device_index = 0; device_index < s_registered_devices.size(); ++device_index) {
    a[device_index] = std::vector<field_t>(size);
    b[device_index] = std::vector<field_t>(size);
  }

  auto test_random_sampling = [&](bool fast_mode) {
    const int N = 15;
    for (int i = 0; i < N; ++i) {
      for (size_t device_index = 0; device_index < s_registered_devices.size(); ++device_index) {
        ICICLE_CHECK(icicle_set_device(s_registered_devices[device_index]));

        // Different seed inconsistency test
        ICICLE_CHECK(random_sampling(size, fast_mode, seed.data(), seed_len, VecOpsConfig{}, a[device_index].data()));
        ICICLE_CHECK(
          random_sampling(size, fast_mode, seed_prime.data(), seed_len, VecOpsConfig{}, b[device_index].data()));
        bool equal = true;
        for (size_t j = 0; j < size; ++j) {
          if (a[device_index][j] != b[device_index][j]) { equal = false; }
        }
        ASSERT_FALSE(equal);

        // Same seed consistency test
        ICICLE_CHECK(random_sampling(size, fast_mode, seed.data(), seed_len, VecOpsConfig{}, a[device_index].data()));
        ICICLE_CHECK(random_sampling(size, fast_mode, seed.data(), seed_len, VecOpsConfig{}, b[device_index].data()));
        for (size_t i = 0; i < size; ++i) {
          ASSERT_EQ(a[device_index][i], b[device_index][i]);
        }
      }
      for (int j = 0; j < size; ++j) {
        for (size_t device_index = 0; device_index < s_registered_devices.size(); ++device_index) {
          ASSERT_EQ(a[device_index][j], b[device_index][j]);
        }
      }
    }
  };
  test_random_sampling(true);
  test_random_sampling(false);
}

TEST_F(RingTestBase, ChallengePolynomialsSampling)
{
  size_t size = 1 << 20;
  size_t seed_len = 32;
  std::vector<std::byte> seed(seed_len);
  for (size_t i = 0; i < seed_len; ++i) {
    seed[i] = static_cast<std::byte>(i);
  }

  std::vector<std::vector<Rq>> outputs(s_registered_devices.size());
  for (size_t device_index = 0; device_index < s_registered_devices.size(); ++device_index) {
    outputs[device_index] = std::vector<Rq>(size);
  }

  const int N = 4;
  for (int i = 0; i < N; ++i) {
    for (size_t device_index = 0; device_index < s_registered_devices.size(); ++device_index) {
      ICICLE_CHECK(icicle_set_device(s_registered_devices[device_index]));
      ICICLE_CHECK(sample_challenge_space_polynomials(seed.data(), seed_len, size, 31, 10, VecOpsConfig{}, outputs[device_index].data()));
    }
  }

  for (size_t device_index = 1; device_index < s_registered_devices.size(); ++device_index) {
    for (size_t i = 0; i < size; ++i) {
      ASSERT_EQ(outputs[device_index][i], outputs[0][i]);
    }
  }
}
