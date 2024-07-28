#include <iostream>
#include <cassert>

#include "icicle/api/bn254.h"
#include "icicle/polynomials/polynomials.h"

#include "examples_utils.h"

using namespace icicle;
using namespace bn254; // typedef scalar_t as bn254-scalar type

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

static std::unique_ptr<scalar_t[]> generate_pows(scalar_t tau, uint32_t size){
    auto vec = std::make_unique<scalar_t[]>(size);
    vec[0] = scalar_t::one();
    for (size_t i = 1; i < size; ++i) {
      vec[i] = vec[i-1] * tau;
  }
  return std::move(vec);
}

static std::unique_ptr<affine_t[]> generate_SRS(uint32_t size) {
  auto secret_scalar = scalar_t::rand_host();
  auto gen = projective_t::generator();
  auto pows_of_tau = generate_pows(secret_scalar,size);
  auto SRS = std::make_unique<affine_t[]>(size);
  for (size_t i = 0; i < size; ++i) {
      SRS[i] = projective_t::to_affine(pows_of_tau[i] * gen);
  }
  return std::move(SRS);
}


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

void example_multiplication_scalar(const int log0)
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

void example_read_coeffs_to_host()
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

void example_division_small()
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

void example_division_large(const int log0, const int log1)
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

void example_divide_by_vanishing_polynomial()
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

void example_even_odd()
{
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

void example_slice()
{
  std::cout << std::endl << "Example: Slice polynomial " << std::endl;
  const scalar_t coeffs[4] = {one, two, three, four}; // 1+2x+3x^2+4x^3
  auto f = Polynomial_t::from_coefficients(coeffs, 4);
  auto f_slice = f.slice(0 /*=offset*/, 3 /*= stride*/, 2 /*/= size*/); // 1+4x
  scalar_t slice_coeffs[2] = {0};
  const auto slice_nof_coeffs = f_slice.copy_coeffs(slice_coeffs, 0, 1);
  std::cout << "Slice: 0:" << slice_coeffs[0] << " expected: " << one << std::endl;
  std::cout << "Slice: 1:" << slice_coeffs[1] << " expected: " << four << std::endl;
}

void example_device_memory_view()
{
  const int log_size = 6;
  const int size = 1 << log_size;
  auto f = randomize_polynomial(size);
  auto [d_coeffs, N] = f.get_coefficients_view();

  // compute coset evaluations
  auto coset_evals = std::make_unique<scalar_t[]>(size);
  auto ntt_config = default_ntt_config<scalar_t>();
  ntt_config.are_inputs_on_device = true; // using the device data directly as a view
  ntt_config.coset_gen = get_root_of_unity<scalar_t>(size * 2);
  ntt(d_coeffs.get(), size, NTTDir::kForward, ntt_config, coset_evals.get());
}


void example_commit_with_device_memory_view()
{
  //declare time vars
  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  std::chrono::milliseconds duration;

  std::cout << std::endl << "Example: a) commit with Polynomial views [(f1+f2)^2 + (f1-f2)^2 ]_1 = [4 (f1^2+ f_2^2)]_1" << std::endl;
  std::cout<< "Example: b) commit with Polynomial views [(f1+f2)^2 - (f1-f2)^2 ]_1 = [4 f1 *f_2]_1" << std::endl;
  int N = 1025;

  //generate group elements string of length N: (1, beta,beta^2....,beta^{N-1}). g
  std::cout << "Setup: Generating mock SRS" << std::endl;
  start = std::chrono::high_resolution_clock::now();
  auto SRS = generate_SRS(2*N);
  //Allocate memory on device (points)
  affine_t* points_d;
  ICICLE_CHECK(icicle_malloc((void**)&points_d, sizeof(affine_t)* 2 * N));
  // copy SRS to device (could have generated on device, but gives an indicator)
  ICICLE_CHECK(icicle_copy(points_d, SRS.get(), sizeof(affine_t)* 2 * N));
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Setup: SRS of length "<< N << " generated and loaded to device. Took: " << duration.count() << " milliseconds" << std::endl;
  
  //goal:
  //test commitment equality [(f1+f2)^2 + (f1-f2)^2 ]_1 = [4 (f1^2+ f_2^2)]_1
  //test commitment equality [(f1+f2)^2 - (f1-f2)^2 ]_1 = [4 f1 *f_2]_1
  //note: using polyapi to gen scalars: already on device. 
  std::cout << "Setup: Generating polys (on device) f1,f2 of log degree " << log2(N-1) << std::endl;
  start = std::chrono::high_resolution_clock::now();
  auto f1 = randomize_polynomial(N);
  auto f2 = randomize_polynomial(N);
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Setup: Gen poly done. Took: " << duration.count() << " milliseconds" << std::endl;
 
  //deg 2N constraints (f1+f2)^2 + (f1-f2)^2 = 2 (f1^2+ f_2^2)
  std::cout << "Computing constraints..start "<< std::endl;
  start = std::chrono::high_resolution_clock::now();
  auto L1 = (f1+f2)*(f1+f2) + (f1-f2)*(f1-f2);
  auto R1 = scalar_t::from(2) * (f1*f1 + f2*f2);
  //deg 2N constraints (f1+f2)^2 - (f1-f2)^2 = 4 f1 *f_2
  auto L2 = (f1+f2)*(f1+f2) - (f1-f2)*(f1-f2);
  auto R2 = scalar_t::from(4) * f1 * f2;
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Computing constraints..done. Took: " << duration.count() << " milliseconds"<< std::endl;
  
  // extract coeff using coeff view
  auto [viewL1, sizeL1] = L1.get_coefficients_view();
  auto [viewL2, sizeL2] = L2.get_coefficients_view(); 
  auto [viewR1, sizeR1] = R1.get_coefficients_view();
  auto [viewR2, sizeR2] = R2.get_coefficients_view();
  
  std::cout << "Computing Commitments with poly view"<< std::endl;
  start = std::chrono::high_resolution_clock::now();
  MSMConfig config = default_msm_config();
  config.are_points_on_device = true;
  config.are_scalars_on_device = true;
 
  //host vars (for result)
  projective_t hL1{}, hL2{}, hR1{}, hR2{};

  //straightforward msm bn254 api: no batching
  msm(viewL1.get(),points_d,N,config,&hL1);
  msm(viewL2.get(),points_d,N,config,&hL2);
  msm(viewR1.get(),points_d,N,config,&hR1);
  msm(viewR2.get(),points_d,N,config,&hR2);

  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Commitments done. Took: " << duration.count() << " milliseconds"<< std::endl;
 
  //sanity checks
  auto affL1 = projective_t::to_affine(hL1);
  auto affR1 = projective_t::to_affine(hR1);

  auto affL2 = projective_t::to_affine(hL2);
  auto affR2 = projective_t::to_affine(hR2);

 //test commitment equality [(f1+f2)^2 + (f1-f2)^2]_1 = [4 (f_1^2+f_2^2]_1
  assert(affL1.x==affR1.x && affL1.y==affR1.y);
  std::cout << "commitment [(f1+f2)^2 + (f1-f2)^2]_1:" << std::endl; 
  std::cout << "[x: " << affL1.x << ", y: " << affL1.y << "]" << std::endl;
  std::cout << "commitment [[2 (f_1^2+f_2^2]_1:" <<std::endl;
  std::cout << "[x: " << affR1.x << ", y: " << affR1.y << "]" << std::endl;

  assert(affL2.x==affR2.x && affL2.y==affR2.y);
  std::cout << "commitment [(f1+f2)^2 - (f1-f2)^2]_1:"<< std::endl;
  std::cout << "[x: " << affL2.x << ", y: " << affL2.y << "]" << std::endl;
  std::cout << "commitment [4 f_1*f_2]_1:"<<std::endl;
  std::cout << "[x: " << affR2.x << ", y: " << affR2.y << "]" << std::endl;
}



int main(int argc, char** argv)
{
   try_load_and_set_backend_device(argc, argv);  
  
  static const int MAX_NTT_LOG_SIZE = 24;  
  const scalar_t basic_root = scalar_t::omega(MAX_NTT_LOG_SIZE);
  ntt_init_domain(basic_root, default_ntt_init_domain_config());  


  START_TIMER(polyapi);

  example_evaluate();
  example_clone(10);
  example_from_rou(100);
  example_addition(12, 17);
  example_addition_inplace(2, 2);
  example_multiplication(15, 12);
  example_multiplication_scalar(15);
  example_monomials();
  example_read_coeffs_to_host();
  example_division_small();
  example_division_large(12, 2);
  example_divide_by_vanishing_polynomial();
  example_even_odd();
  example_slice();
  example_device_memory_view();
  example_commit_with_device_memory_view();

  END_TIMER(polyapi, "polyapi example took");

  return 0;
}