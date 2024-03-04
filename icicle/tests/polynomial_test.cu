
#include <gtest/gtest.h>
#include <iostream>
#include <memory>

#include "curves/curve_config.cuh"
typedef curve_config::scalar_t test_type;

#include "polynomials/polynomials.h"
#include "appUtils/ntt/ntt.cuh"

using FpMicroseconds = std::chrono::duration<float, std::chrono::microseconds::period>;
#define START_TIMER(timer) auto timer##_start = std::chrono::high_resolution_clock::now();
#define END_TIMER(timer, msg, enable)                                                                                  \
  if (enable)                                                                                                          \
    printf(                                                                                                            \
      "%s: %.3f ms\n", msg, FpMicroseconds(std::chrono::high_resolution_clock::now() - timer##_start).count() / 1000);

using namespace polynomials;

typedef Polynomial<test_type> Polynomial_t;

class PolynomialTest : public ::testing::Test
{
public:
  static inline const int MAX_NTT_LOG_SIZE = 24;
  static inline const bool MEASURE = true;

  // SetUpTestSuite/TearDownTestSuite are called once for the entire test suite
  static void SetUpTestSuite()
  {
    // init NTT domain
    auto ntt_config = ntt::DefaultNTTConfig<test_type>();
    const test_type basic_root = test_type::omega(MAX_NTT_LOG_SIZE);
    ntt::InitDomain(basic_root, ntt_config.ctx);
    // initializing polynoimals factory for CUDA backend
    Polynomial_t::initialize(std::make_unique<CUDAPolynomialFactory<>>());
  }

  static void TearDownTestSuite() {}

  void SetUp() override
  {
    // code that executes before each test
  }

  void TearDown() override
  {
    // code that executes before each test
  }

  static Polynomial_t randomize_polynomial(uint32_t size, bool random = true)
  {
    auto coeff = std::make_unique<test_type[]>(size);
    if (random) {
      random_samples(coeff.get(), size);
    } else {
      incremental_values(coeff.get(), size);
    }
    return Polynomial_t::from_coefficients(coeff.get(), size);
  }

  static void random_samples(test_type* res, uint32_t count)
  {
    for (int i = 0; i < count; i++)
      res[i] = test_type::rand_host();
  }

  static void incremental_values(test_type* res, uint32_t count)
  {
    for (int i = 0; i < count; i++) {
      res[i] = i ? res[i - 1] + test_type::one() : test_type::one();
    }
  }

  static void assert_equal(Polynomial_t& lhs, Polynomial_t& rhs)
  {
    const int deg_lhs = lhs.degree();
    const int deg_rhs = rhs.degree();
    ASSERT_EQ(deg_lhs, deg_rhs);

    auto lhs_coeffs = std::make_unique<test_type[]>(deg_lhs);
    auto rhs_coeffs = std::make_unique<test_type[]>(deg_rhs);
    lhs.get_coefficients_on_host(lhs_coeffs.get(), 1, deg_lhs - 1);
    rhs.get_coefficients_on_host(rhs_coeffs.get(), 1, deg_rhs - 1);

    ASSERT_EQ(0, memcmp(lhs_coeffs.get(), rhs_coeffs.get(), deg_lhs * sizeof(test_type)));
  }

  const static inline auto zero = test_type::zero();
  const static inline auto one = test_type::one();
  const static inline auto two = test_type::from(2);
  const static inline auto three = test_type::from(3);
  const static inline auto four = test_type::from(4);
  const static inline auto five = test_type::from(5);
  const static inline auto minus_one = zero - one;
};

TEST_F(PolynomialTest, evalution)
{
  const test_type coeffs[3] = {one, two, three};
  auto f = Polynomial_t::from_coefficients(coeffs, 3);
  test_type x = test_type::rand_host();

  auto f_x = f(x); // evaluation

  auto expected_f_x = one + two * x + three * x * x;

  EXPECT_EQ(f_x, expected_f_x);
}

TEST_F(PolynomialTest, addition)
{
  const int size_0 = 12, size_1 = 17;
  auto f = randomize_polynomial(size_0);
  auto g = randomize_polynomial(size_1);

  test_type x = test_type::rand_host();
  auto f_x = f(x);
  auto g_x = g(x);
  auto fx_plus_gx = f_x + g_x;

  auto s = f + g;
  auto s_x = s(x);

  EXPECT_EQ(fx_plus_gx, s_x);
}

TEST_F(PolynomialTest, cAPI)
{
  const int size = 3;
  auto coeff = std::make_unique<test_type[]>(size);
  random_samples(coeff.get(), size);

  auto f = polynomial_create_from_coefficients(coeff.get(), size);
  auto g = polynomial_create_from_coefficients(coeff.get(), size);
  auto s = polynomial_add(f, g);

  test_type x = test_type::rand_host();
  // TODO Yuval: use C-API for evalute too
  auto f_x = f->evaluate(x);
  auto g_x = g->evaluate(x);
  auto fx_plus_gx = f_x + g_x;
  auto s_x = s->evaluate(x);
  EXPECT_EQ(fx_plus_gx, s_x);

  polynomial_delete(f);
  polynomial_delete(g);
  polynomial_delete(s);
}

TEST_F(PolynomialTest, multiplication)
{
  const int size_0 = 1 << 15, size_1 = 1 << 12;
  auto f = randomize_polynomial(size_0);
  auto g = randomize_polynomial(size_1);

  test_type x = test_type::rand_host();
  auto f_x = f(x);
  auto g_x = g(x);
  auto fx_mul_gx = f_x * g_x;

  START_TIMER(poly_mult_start);
  auto m = f * g;
  END_TIMER(poly_mult_start, "Polynomial multiplication took", MEASURE);

  auto m_x = m(x);

  EXPECT_EQ(fx_mul_gx, m_x);
}

TEST_F(PolynomialTest, monomials)
{
  const test_type coeffs[3] = {one, zero, two}; // 1+2x^2
  auto f = Polynomial_t::from_coefficients(coeffs, 3);
  const auto x = three;
  const auto expected_f_x = one + two * x * x;
  auto f_x = f(x);

  EXPECT_EQ(f_x, expected_f_x);

  f.add_monomial_inplace(three, 1); // add 3x
  const auto expected_addmonmon_f_x = f_x + three * x;
  const auto addmonom_f_x = f(x);

  EXPECT_EQ(addmonom_f_x, expected_addmonmon_f_x);

  f.sub_monomial_inplace(one); // subtract 1. equivalant to 'f-1'
  const auto expected_submonom_f_x = addmonom_f_x - one;
  const auto submonom_f_x = f(x);

  EXPECT_EQ(submonom_f_x, expected_submonom_f_x);
}

TEST_F(PolynomialTest, ReadCoeffsToHost)
{
  const test_type coeffs_f[3] = {zero, one, two}; // x+2x^2
  auto f = Polynomial_t::from_coefficients(coeffs_f, 3);
  const test_type coeffs_g[3] = {one, one, one}; // 1+x+x^2
  auto g = Polynomial_t::from_coefficients(coeffs_g, 3);

  auto h = f + g; // 1+2x+3x^3
  const auto h0 = h.get_coefficient_on_host(0);
  const auto h1 = h.get_coefficient_on_host(1);
  const auto h2 = h.get_coefficient_on_host(2);
  EXPECT_EQ(h0, one);
  EXPECT_EQ(h1, two);
  EXPECT_EQ(h2, three);

  int64_t nof_coeffs = h.get_coefficients_on_host(nullptr); // query #coeffs
  EXPECT_GE(nof_coeffs, 3);                                 // can be larger due to padding to powers of two
  test_type h_coeffs[3] = {0};
  nof_coeffs = h.get_coefficients_on_host(h_coeffs, 0, 2); // read the coefficients
  EXPECT_EQ(nof_coeffs, 3);                                // expecting 3 due to specified indices

  test_type expected_h_coeffs[nof_coeffs] = {one, two, three, zero};
  for (int i = 0; i < nof_coeffs; ++i) {
    EXPECT_EQ(expected_h_coeffs[i], h_coeffs[i]);
  }
}

TEST_F(PolynomialTest, divisionSimple)
{
  const test_type coeffs_a[4] = {five, zero, four, three}; // 3x^3+4x^2+5
  const test_type coeffs_b[3] = {minus_one, zero, one};    // x^2-1
  auto a = Polynomial_t::from_coefficients(coeffs_a, 4);
  auto b = Polynomial_t::from_coefficients(coeffs_b, 3);

  auto [q, r] = a.divide(b);
  test_type q_coeffs[2] = {0}; // 3x+4
  test_type r_coeffs[2] = {0}; // 3x+9
  const auto q_nof_coeffs = q.get_coefficients_on_host(q_coeffs, 0, 1);
  const auto r_nof_coeffs = r.get_coefficients_on_host(r_coeffs, 0, 1);

  ASSERT_EQ(q_nof_coeffs, 2);
  ASSERT_EQ(r_nof_coeffs, 2);
  ASSERT_EQ(q_coeffs[0], test_type::from(4));
  ASSERT_EQ(q_coeffs[1], test_type::from(3));
  ASSERT_EQ(r_coeffs[0], test_type::from(9));
  ASSERT_EQ(r_coeffs[1], test_type::from(3));
}

TEST_F(PolynomialTest, divisionLarge)
{
  const int size_0 = 1 << 12, size_1 = 1 << 2;
  auto a = randomize_polynomial(size_0);
  auto b = randomize_polynomial(size_1);

  START_TIMER(poly_mult_start);
  auto [q, r] = a.divide(b);
  END_TIMER(poly_mult_start, "Polynomial division took", MEASURE);

  test_type x = test_type::rand_host();
  auto a_x = a(x);
  auto b_x = b(x);
  auto q_x = q(x);
  auto r_x = r(x);

  // a(x) = b(x)*q(x)+r(x)
  EXPECT_EQ(a_x, b_x * q_x + r_x);
}

TEST_F(PolynomialTest, divideByVanishingPolynomial)
{
  const test_type coeffs_v[5] = {minus_one, zero, zero, zero, one}; // x^4-1 vanishes on 4th roots of unity
  auto v = Polynomial_t::from_coefficients(coeffs_v, 5);
  auto h = randomize_polynomial(1 << 11, false);
  auto hv = h * v;

  START_TIMER(poly_div_long);
  auto [h_div, R] = hv.divide(v);
  END_TIMER(poly_div_long, "Polynomial division by vanishing (long division) took", MEASURE);
  assert_equal(h_div, h);

  START_TIMER(poly_div_vanishing);
  auto h_div_by_vanishing = hv.divide_by_vanishing_polynomial(4);
  END_TIMER(poly_div_vanishing, "Polynomial division by vanishing (fast) took", MEASURE);
  assert_equal(h_div_by_vanishing, h);
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}