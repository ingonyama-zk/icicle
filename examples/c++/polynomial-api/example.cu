#include <iostream>

#include "polynomials/polynomials.h"
#include "polynomials/cuda_backend/polynomial_cuda_backend.cuh"
#include "ntt/ntt.cuh"
#include "poseidon/tree/merkle.cuh"

// using namespace field_config;
using namespace polynomials;
using namespace merkle;

// define the polynomial type
typedef Polynomial<scalar_t> Polynomial_t;

// we'll use the following constants in the examples
const auto zero = scalar_t::zero();
const auto one = scalar_t::one();
const auto two = scalar_t::from(2);
const auto three = scalar_t::from(3);
const auto four = scalar_t::from(4);
const auto five = scalar_t::from(5);
const auto minus_one = zero - one;

void example_evaluate()
{
  std::cout << std::endl << "Example: Polynomial evaluation on random value" << std::endl;
  const scalar_t coeffs[3] = {one, two, three};
  auto f = Polynomial_t::from_coefficients(coeffs, 3);
  std::cout << "f = " << f << std::endl;
  scalar_t x = scalar_t::rand_host();
  std::cout << "x = " << x << std::endl;
  auto fx = f(x);
  std::cout << "f(x) = " << fx << std::endl;
}

void example_from_rou(const int size)
{
  std::cout << std::endl << "Example: Reconstruct polynomial from values at roots of unity" << std::endl;
  const int log_size = (int)ceil(log2(size));
  const int nof_evals = 1 << log_size;
  auto coeff = std::make_unique<scalar_t[]>(size);
  for (int i = 0; i < size; i++)
    coeff[i] = scalar_t::rand_host();
  auto f = Polynomial_t::from_coefficients(coeff.get(), size);
  // rou: root of unity
  auto omega = scalar_t::omega(log_size);
  scalar_t evals[nof_evals] = {scalar_t::zero()};
  auto x = scalar_t::one();
  for (int i = 0; i < nof_evals; ++i) {
    evals[i] = f(x);
    x = x * omega;
  }
  // reconstruct f from evaluations
  auto fr = Polynomial_t::from_rou_evaluations(evals, nof_evals);
  // check for equality f-fr==0
  auto h = f - fr;
  std::cout << "degree of f - fr = " << h.degree() << std::endl;
}

static Polynomial_t randomize_polynomial(uint32_t size)
{
  auto coeff = std::make_unique<scalar_t[]>(size);
  for (int i = 0; i < size; i++)
    coeff[i] = scalar_t::rand_host();
  return Polynomial_t::from_coefficients(coeff.get(), size);
}

static Polynomial_t incremental_values(uint32_t size)
{
  auto coeff = std::make_unique<scalar_t[]>(size);
  for (int i = 0; i < size; i++) {
    coeff[i] = i ? coeff[i - 1] + scalar_t::one() : scalar_t::one();
  }
  return Polynomial_t::from_coefficients(coeff.get(), size);
}

static bool is_equal(Polynomial_t& lhs, Polynomial_t& rhs)
{
  const int deg_lhs = lhs.degree();
  const int deg_rhs = rhs.degree();
  if (deg_lhs != deg_rhs) { return false; }
  auto lhs_coeffs = std::make_unique<scalar_t[]>(deg_lhs);
  auto rhs_coeffs = std::make_unique<scalar_t[]>(deg_rhs);
  lhs.copy_coeffs(lhs_coeffs.get(), 1, deg_lhs - 1);
  rhs.copy_coeffs(rhs_coeffs.get(), 1, deg_rhs - 1);
  return memcmp(lhs_coeffs.get(), rhs_coeffs.get(), deg_lhs * sizeof(scalar_t)) == 0;
}

void example_addition(const int size0, const int size1)
{
  std::cout << std::endl << "Example: Polynomial addition" << std::endl;
  auto f = randomize_polynomial(size0);
  auto g = randomize_polynomial(size1);
  auto x = scalar_t::rand_host();
  auto f_x = f(x);
  auto g_x = g(x);
  auto fx_plus_gx = f_x + g_x;
  auto h = f + g;
  auto h_x = h(x);
  std::cout << "evaluate and add: " << fx_plus_gx << std::endl;
  std::cout << "add and evaluate: " << h_x << std::endl;
}

void example_addition_inplace(const int size0, const int size1)
{
  std::cout << std::endl << "Example: Polynomial inplace addition" << std::endl;
  auto f = randomize_polynomial(size0);
  auto g = randomize_polynomial(size1);

  auto x = scalar_t::rand_host();
  auto f_x = f(x);
  auto g_x = g(x);
  auto fx_plus_gx = f_x + g_x;
  f += g;
  auto s_x = f(x);
  std::cout << "evaluate and add: " << fx_plus_gx << std::endl;
  std::cout << "add and evaluate: " << s_x << std::endl;
}

void example_multiplication(const int log0, const int log1)
{
  std::cout << std::endl << "Example: Polynomial multiplication" << std::endl;
  const int size0 = 1 << log0, size1 = 1 << log1;
  auto f = randomize_polynomial(size0);
  auto g = randomize_polynomial(size1);
  scalar_t x = scalar_t::rand_host();
  auto fx = f(x);
  auto gx = g(x);
  auto fx_mul_gx = fx * gx;
  auto m = f * g;
  auto mx = m(x);
  std::cout << "evaluate and multiply: " << fx_mul_gx << std::endl;
  std::cout << "multiply and evaluate: " << mx << std::endl;
}

void example_multiplicationScalar(const int log0)
{
  std::cout << std::endl << "Example: Scalar by Polynomial multiplication" << std::endl;
  const int size = 1 << log0;
  auto f = randomize_polynomial(size);
  auto s = scalar_t::from(2);
  auto g = s * f;
  auto x = scalar_t::rand_host();
  auto fx = f(x);
  auto fx2 = s * fx;
  auto gx = g(x);
  std::cout << "Compare (2*f)(x) and 2*f(x): " << std::endl;
  std::cout << gx << std::endl;
  std::cout << fx2 << std::endl;
}

void example_monomials()
{
  std::cout << std::endl << "Example: Monomials" << std::endl;
  const scalar_t coeffs[3] = {one, zero, two}; // 1+2x^2
  auto f = Polynomial_t::from_coefficients(coeffs, 3);
  const auto x = three;
  auto fx = f(x);
  f.add_monomial_inplace(three, 1); // add 3x
  const auto expected_addmonmon_f_x = fx + three * x;
  const auto addmonom_f_x = f(x);
  std::cout << "Computed f'(x) = " << addmonom_f_x << std::endl;
  std::cout << "Expected f'(x) = " << expected_addmonmon_f_x << std::endl;
}

void example_ReadCoeffsToHost()
{
  std::cout << std::endl << "Example: Read coefficients to host" << std::endl;
  const scalar_t coeffs_f[3] = {zero, one, two}; // 0+1x+2x^2
  auto f = Polynomial_t::from_coefficients(coeffs_f, 3);
  const scalar_t coeffs_g[3] = {one, one, one}; // 1+x+x^2
  auto g = Polynomial_t::from_coefficients(coeffs_g, 3);
  auto h = f + g; // 1+2x+3x^3
  std::cout << "Get one coefficient of h() at a time: " << std::endl;
  const auto h0 = h.get_coeff(0);
  const auto h1 = h.get_coeff(1);
  const auto h2 = h.get_coeff(2);
  std::cout << "Coefficients of h: " << std::endl;
  std::cout << "0:" << h0 << " expected: " << one << std::endl;
  std::cout << "1:" << h1 << " expected: " << two << std::endl;
  std::cout << "2:" << h2 << " expected: " << three << std::endl;
  std::cout << "Get all coefficients of h() at a time: " << std::endl;

  scalar_t h_coeffs[3] = {0};
  // fetch the coefficients for a given range
  auto nof_coeffs = h.copy_coeffs(h_coeffs, 0, 2);
  scalar_t expected_h_coeffs[nof_coeffs] = {one, two, three};
  for (int i = 0; i < nof_coeffs; ++i) {
    std::cout << i << ":" << h_coeffs[i] << " expected: " << expected_h_coeffs[i] << std::endl;
  }
}

void example_divisionSmall()
{
  std::cout << std::endl << "Example: Polynomial division (small)" << std::endl;
  const scalar_t coeffs_a[4] = {five, zero, four, three}; // 3x^3+4x^2+5
  const scalar_t coeffs_b[3] = {minus_one, zero, one};    // x^2-1
  auto a = Polynomial_t::from_coefficients(coeffs_a, 4);
  auto b = Polynomial_t::from_coefficients(coeffs_b, 3);
  auto [q, r] = a.divide(b);
  scalar_t q_coeffs[2] = {0}; // 3x+4
  scalar_t r_coeffs[2] = {0}; // 3x+9
  const auto q_nof_coeffs = q.copy_coeffs(q_coeffs, 0, 1);
  const auto r_nof_coeffs = r.copy_coeffs(r_coeffs, 0, 1);
  std::cout << "Quotient: 0:" << q_coeffs[0] << " expected: " << scalar_t::from(4) << std::endl;
  std::cout << "Quotient: 1:" << q_coeffs[1] << " expected: " << scalar_t::from(3) << std::endl;
  std::cout << "Reminder: 0:" << r_coeffs[0] << " expected: " << scalar_t::from(9) << std::endl;
  std::cout << "Reminder: 1:" << r_coeffs[1] << " expected: " << scalar_t::from(3) << std::endl;
}

void example_divisionLarge(const int log0, const int log1)
{
  std::cout << std::endl << "Example: Polynomial division (large)" << std::endl;
  const int size0 = 1 << log0, size1 = 1 << log1;
  auto a = randomize_polynomial(size0);
  auto b = randomize_polynomial(size1);
  auto [q, r] = a.divide(b);
  scalar_t x = scalar_t::rand_host();
  auto ax = a(x);
  auto bx = b(x);
  auto qx = q(x);
  auto rx = r(x);
  // check if a(x) == b(x)*q(x)+r(x)
  std::cout << "a(x) == b(x)*q(x)+r(x)" << std::endl;
  std::cout << "lhs = " << ax << std::endl;
  std::cout << "rhs = " << bx * qx + rx << std::endl;
}

void example_divideByVanishingPolynomial()
{
  std::cout << std::endl << "Example: Polynomial division by vanishing polynomial" << std::endl;
  const scalar_t coeffs_v[5] = {minus_one, zero, zero, zero, one}; // x^4-1 vanishes on 4th roots of unity
  auto v = Polynomial_t::from_coefficients(coeffs_v, 5);
  auto h = incremental_values(1 << 11);
  auto hv = h * v;
  auto [h_div, R] = hv.divide(v);
  std::cout << "h_div == h: " << is_equal(h_div, h) << std::endl;
  auto h_div_by_vanishing = hv.divide_by_vanishing_polynomial(4);
  std::cout << "h_div_by_vanishing == h: " << is_equal(h_div_by_vanishing, h) << std::endl;
}

void example_clone(const int log0)
{
  std::cout << std::endl << "Example: clone polynomial" << std::endl;
  const int size = 1 << log0;
  auto f = randomize_polynomial(size);
  const auto x = scalar_t::rand_host();
  const auto fx = f(x);
  Polynomial_t g;
  g = f.clone();
  g += f;
  auto h = g.clone();
  std::cout << "g(x) = " << g(x) << " expected: " << two * fx << std::endl;
  std::cout << "h(x) = " << h(x) << " expected: " << g(x) << std::endl;
}

void example_EvenOdd() {
  std::cout << std::endl << "Example: Split into even and odd powers " << std::endl;
  const scalar_t coeffs[4] = {one, two, three, four}; // 1+2x+3x^2+4x^3
  auto f = Polynomial_t::from_coefficients(coeffs, 4);
  auto f_even = f.even();
  auto f_odd = f.odd();

  scalar_t even_coeffs[2] = {0};
  scalar_t odd_coeffs[2] = {0};
  const auto even_nof_coeffs = f_even.copy_coeffs(even_coeffs, 0, 1);
  const auto odd_nof_coeffs = f_odd.copy_coeffs(odd_coeffs, 0, 1);
  std::cout << "Even: 0:" << even_coeffs[0] << " expected: " << one << std::endl;
  std::cout << "Even: 1:" << even_coeffs[1] << " expected: " << three << std::endl;
  std::cout << "Odd: 0:" << odd_coeffs[0] << " expected: " << two << std::endl;
  std::cout << "Odd: 1:" << odd_coeffs[1] << " expected: " << four << std::endl;

}

void example_Slice() {
  std::cout << std::endl << "Example: Slice polynomial " << std::endl;
  const scalar_t coeffs[4] = {one, two, three, four}; // 1+2x+3x^2+4x^3
  auto f = Polynomial_t::from_coefficients(coeffs, 4);
  auto f_slice = f.slice(0, 3, 2); // 1+4x
  scalar_t slice_coeffs[2] = {0};
  const auto slice_nof_coeffs = f_slice.copy_coeffs(slice_coeffs, 0, 1);
  std::cout << "Slice: 0:" << slice_coeffs[0] << " expected: " << one << std::endl;
  std::cout << "Slice: 1:" << slice_coeffs[1] << " expected: " << four << std::endl;
} 


template <typename S>
  size_t my_get_digests_len(uint32_t height, uint32_t arity)
  {
    size_t digests_len = 0;
    size_t row_length = 1;
    for (int i = 1; i < height; i++) {
      digests_len += row_length;
      row_length *= arity;
    }

    return digests_len;
  }

void example_DeviceMemoryView() {
  const int log_size = 6;
  const int size = 1 << log_size;
  auto f = randomize_polynomial(size);
  auto [d_coeff, N, device_id] = f.get_coefficients_view();
  // std::cout << "Device id: " << device_id << std::endl;
  // std::cout << "D: " <<   d_coeff.isValid() << std::endl;
  // commit coefficients to Merkle tree
  device_context::DeviceContext ctx = device_context::get_default_device_context();
  PoseidonConstants<scalar_t> constants;
  init_optimized_poseidon_constants<scalar_t>(2, ctx, &constants);
  uint32_t tree_height = log_size + 1;
  int keep_rows = 0; // keep all rows
  size_t digests_len = log_size - 1;
  scalar_t* digests = static_cast<scalar_t*>(malloc(sizeof(scalar_t) * digests_len));
  TreeBuilderConfig config = default_merkle_config();
  config.keep_rows = keep_rows;
  config.are_inputs_on_device = true;
  build_merkle_tree<scalar_t, (2+1)>(d_coeff.get(), digests, tree_height, constants, config);
  std::cout << "Merkle tree root: " << digests[0] << std::endl;
  free(digests);
}

int main(int argc, char** argv)
{
  // Initialize NTT. TODO: can we hide this in the library?
  static const int MAX_NTT_LOG_SIZE = 24;
  auto ntt_config = ntt::default_ntt_config<scalar_t>();
  const scalar_t basic_root = scalar_t::omega(MAX_NTT_LOG_SIZE);
  ntt::init_domain(basic_root, ntt_config.ctx);

  // Virtual factory design pattern: initializing polynomimals factory for CUDA backend
  Polynomial_t::initialize(std::make_unique<CUDAPolynomialFactory<>>());

  example_evaluate();
  example_clone(10);
  example_from_rou(100);
  example_addition(12, 17);
  example_addition_inplace(2, 2);
  example_multiplication(15, 12);
  example_multiplicationScalar(15);
  example_monomials();
  example_ReadCoeffsToHost();
  example_divisionSmall();
  example_divisionLarge(12, 2);
  example_divideByVanishingPolynomial();
  example_EvenOdd();
  example_Slice();
  example_DeviceMemoryView();

  return 0;
}