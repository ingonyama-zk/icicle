
#include <iostream>
#include <vector>
#include <memory>

#include "curves/curve_config.cuh"
typedef curve_config::scalar_t test_type;

#include "polynomials/polynomials.h"
#include "appUtils/ntt/ntt.cuh"

void random_samples(test_type* res, uint32_t count)
{
  for (int i = 0; i < count; i++)
    res[i] = test_type::rand_host();
}

void incremental_values(test_type* res, uint32_t count)
{
  for (int i = 0; i < count; i++) {
    res[i] = i ? res[i - 1] + test_type::one() : test_type::zero();
  }
}

int main()
{
  using namespace polynomials;

  // init NTT domain
  auto ntt_config = ntt::DefaultNTTConfig<test_type>();
  const int MAX_NTT_LOG_SIZE = 20;
  const test_type basic_root = test_type::omega(MAX_NTT_LOG_SIZE);
  ntt::InitDomain(basic_root, ntt_config.ctx);

  // const uint32_t test_size = 3;
  auto coeff_0 = std::make_unique<test_type[]>(3);
  auto coeff_1 = std::make_unique<test_type[]>(2);
  random_samples(coeff_0.get(), 3);
  random_samples(coeff_1.get(), 2);

  auto a = Polynomial<test_type>::from_coefficients(coeff_0.get(), 3);
  auto& b = *polynomial_create_from_coefficients(coeff_1.get(), 2); // C API
  {
    auto c = a + b;
    auto d = c - a;

    std::cout << "a=(deg=" << a.degree() << ")" << a;
    std::cout << "b=(deg=" << b.degree() << ")" << b;
    std::cout << "c=(deg=" << c.degree() << ")" << c;
    std::cout << "d=(deg=" << d.degree() << ")" << d;
  }
  std::cout << "deleting b" << std::endl;

  polynomial_delete(&b);

  return 0;
}