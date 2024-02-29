
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <memory>

#include "curves/curve_config.cuh"
typedef curve_config::scalar_t test_type;

#include "polynomials/polynomials.h"
#include "appUtils/ntt/ntt.cuh"

using FpMicroseconds = std::chrono::duration<float, std::chrono::microseconds::period>;
#define START_TIMER(timer) auto timer##_start = std::chrono::high_resolution_clock::now();
#define END_TIMER(timer, msg, enable)                                                                                  \
  if (enable)                                                                                                          \
    printf("%s: %.0f us\n", msg, FpMicroseconds(std::chrono::high_resolution_clock::now() - timer##_start).count());

using namespace polynomials;

typedef Polynomial<test_type> Polynomial_t;

class PolynomialTest : public ::testing::Test
{
public:
  static inline const int MAX_NTT_LOG_SIZE = 24;
  static inline const bool DEBUG = false; // set true for debug prints
  static inline const bool MEASURE = true;

  // SetUpTestSuite/TearDownTestSuite are called once for the entire test suite
  static void SetUpTestSuite()
  {
    // init NTT domain
    auto ntt_config = ntt::DefaultNTTConfig<test_type>();
    const test_type basic_root = test_type::omega(MAX_NTT_LOG_SIZE);
    ntt::InitDomain(basic_root, ntt_config.ctx);
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
};

TEST_F(PolynomialTest, evalution)
{
  const auto a = test_type::one();
  const auto b = a + a; // 2
  const auto c = b + a; // 3
  const test_type coeffs[3] = {a, b, c};
  auto f = Polynomial_t::from_coefficients(coeffs, 3);
  test_type x = test_type::rand_host();
  auto f_x = f(x);

  auto expected_f_x = a + b * x + c * x * x;

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
  if (DEBUG) {
    std::cout << "x=" << x << "\n";
    std::cout << "f_x=" << f_x << "\n";
    std::cout << "g_x=" << g_x << "\n";
    std::cout << "s_x=" << s_x << "\n";
    std::cout << "f=(deg=" << f.degree() << ")" << f;
    std::cout << "g=(deg=" << g.degree() << ")" << g;
    std::cout << "s=(deg=" << s.degree() << ")" << s;
  }
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
  const int size_0 = 1 << 12, size_1 = 1 << 10;
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
  const auto zero = test_type::zero();
  const auto one = test_type::one();
  const auto two = one + one;
  const auto three = two + one;

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
  const auto zero = test_type::zero();
  const auto one = test_type::one();
  const auto two = one + one;
  const auto three = two + one;

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

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}