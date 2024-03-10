#define CURVE_ID 1
#include "curves/curve_config.cuh"
#include "polynomials/polynomials.cpp"
using namespace curve_config;
using namespace polynomials;

typedef Polynomial<scalar_t> Polynomial_t;

int main(int argc, char** argv)
{
    const static auto one = scalar_t::one();
    const static auto two = scalar_t::from(2);
    const static auto three = scalar_t::from(3);

    // initializing polynoimals factory for CUDA backend
    Polynomial_t::initialize(std::make_unique<CUDAPolynomialFactory<>>());

    const scalar_t coeffs[3] = {one, two, three};
    auto f = Polynomial_t::from_coefficients(coeffs, 3);
    scalar_t x = scalar_t::rand_host();

    auto f_x = f(x); // evaluation
  return 0;
}