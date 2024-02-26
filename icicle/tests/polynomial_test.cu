
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <memory>

#include "curves/curve_config.cuh"
typedef curve_config::scalar_t test_type;

#include "polynomials/polynomials.h"
#include "appUtils/ntt/ntt.cuh"

using namespace polynomials;

class PolynomialTest : public ::testing::Test
{
public:
  static inline const int MAX_NTT_LOG_SIZE = 20;

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

  static void random_samples(test_type* res, uint32_t count)
  {
    for (int i = 0; i < count; i++)
      res[i] = test_type::rand_host();
  }

  static void incremental_values(test_type* res, uint32_t count)
  {
    for (int i = 0; i < count; i++) {
      res[i] = i ? res[i - 1] + test_type::one() : test_type::zero();
    }
  }
};

TEST_F(PolynomialTest, addition)
{
  const int size_0 = 3, size_1 = 2;
  auto coeff_0 = std::make_unique<test_type[]>(size_0);
  auto coeff_1 = std::make_unique<test_type[]>(size_1);
  random_samples(coeff_0.get(), size_0);
  random_samples(coeff_1.get(), size_1);

  auto a = Polynomial<test_type>::from_coefficients(coeff_0.get(), size_0);
  auto b = Polynomial<test_type>::from_coefficients(coeff_1.get(), size_1);
  auto c = a + b;
  auto d = c - a;

  std::cout << "a=(deg=" << a.degree() << ")" << a;
  std::cout << "b=(deg=" << b.degree() << ")" << b;
  std::cout << "c=(deg=" << c.degree() << ")" << c;
  std::cout << "d=(deg=" << d.degree() << ")" << d;
}

TEST_F(PolynomialTest, cAPI)
{
  const int size = 3;
  auto coeff = std::make_unique<test_type[]>(size);
  random_samples(coeff.get(), size);

  auto a = polynomial_create_from_coefficients(coeff.get(), size);
  auto res = polynomial_add(a, a);

  std::cout << "a=(deg=" << a->degree() << ")" << *a;
  std::cout << "res=(deg=" << res->degree() << ")" << *res;

  polynomial_delete(a);
  polynomial_delete(res);
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}