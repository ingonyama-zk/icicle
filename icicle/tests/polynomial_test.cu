#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <vector>
#include <list>
#include <fstream>

#include "polynomials/polynomials.h"
#include "polynomials/polynomials_c_api.h"
#include "polynomials/cuda_backend/polynomial_cuda_backend.cuh"
#include "polynomials/tracing/polynomial_tracing_backend.cuh"
#include "polynomials/tracing/graph_visualizer_visitor.h"
#include "polynomials/tracing/optimizer.h"
#include "polynomials/tracing/interpreter.h"

#include "ntt/ntt.cuh"
#include "gpu-utils/device_context.cuh"

/*******************************************/

using FpMicroseconds = std::chrono::duration<float, std::chrono::microseconds::period>;
#define START_TIMER(timer) auto timer##_start = std::chrono::high_resolution_clock::now();
#define END_TIMER(timer, msg, enable)                                                                                  \
  if (enable)                                                                                                          \
    printf("%s: %.0f us\n", msg, FpMicroseconds(std::chrono::high_resolution_clock::now() - timer##_start).count());

using namespace polynomials;

typedef Polynomial<scalar_t> Polynomial_t;

static uint64_t ceil_to_power_of_two(uint64_t x) { return 1ULL << uint64_t(ceil(log2(x))); }

class PolynomialTest : public ::testing::Test
{
public:
  static inline const int MAX_NTT_LOG_SIZE = 24;
  static inline const bool MEASURE = true;
  static inline const bool TRACING = true;

  static inline std::shared_ptr<CUDAPolynomialFactory<>> m_cuda_factory = nullptr;
  static inline std::shared_ptr<TracingPolynomialFactory<>> m_tracing_factory = nullptr;

  // SetUpTestSuite/TearDownTestSuite are called once for the entire test suite
  static void SetUpTestSuite()
  {
    // init NTT domain
    auto ntt_config = ntt::DefaultNTTConfig<scalar_t>();
    const scalar_t basic_root = scalar_t::omega(MAX_NTT_LOG_SIZE);
    ntt::InitDomain(basic_root, ntt_config.ctx);

    // initializing polynoimals factory for CUDA backend, or tracing backend
    m_cuda_factory = std::make_shared<CUDAPolynomialFactory<>>();
    m_tracing_factory = std::make_shared<TracingPolynomialFactory<>>(m_cuda_factory);
    if (TRACING) {
      enable_cuda_tracing();
    } else {
      enable_cuda_eager();
    }
  }
  static void enable_cuda_tracing() { Polynomial_t::initialize(m_tracing_factory); }
  static void enable_cuda_eager() { Polynomial_t::initialize(m_cuda_factory); }

  static void TearDownTestSuite()
  {
    m_tracing_factory = nullptr;
    m_cuda_factory = nullptr;
  }

  void SetUp() override
  {
    // code that executes before each test
  }

  void TearDown() override
  {
    // code that executes after each test
  }

  static Polynomial_t randomize_polynomial(uint32_t size, bool random = true)
  {
    auto coeff = std::make_unique<scalar_t[]>(size);
    if (random) {
      random_samples(coeff.get(), size);
    } else {
      incremental_values(coeff.get(), size);
    }
    return Polynomial_t::from_coefficients(coeff.get(), size);
  }

  static void random_samples(scalar_t* res, uint32_t count)
  {
    for (int i = 0; i < count; i++)
      res[i] = scalar_t::rand_host();
  }

  static void incremental_values(scalar_t* res, uint32_t count)
  {
    for (int i = 0; i < count; i++) {
      res[i] = i ? res[i - 1] + scalar_t::one() : scalar_t::one();
    }
  }

  template <typename P, typename A>
  static void compute_powers_of_tau(P g, scalar_t tau, A* res, uint32_t count)
  {
    res[0] = P::to_affine(g);
    for (int i = 1; i < count; i++) {
      g = tau * g;
      res[i] = P::to_affine(g);
    }
  }

  static void assert_equal(Polynomial_t& lhs, Polynomial_t& rhs)
  {
    const int deg_lhs = lhs.degree();
    const int deg_rhs = rhs.degree();
    ASSERT_EQ(deg_lhs, deg_rhs);

    auto lhs_coeffs = std::make_unique<scalar_t[]>(deg_lhs);
    auto rhs_coeffs = std::make_unique<scalar_t[]>(deg_rhs);
    lhs.copy_coefficients_to_host(lhs_coeffs.get(), 1, deg_lhs - 1);
    rhs.copy_coefficients_to_host(rhs_coeffs.get(), 1, deg_rhs - 1);

    ASSERT_EQ(0, memcmp(lhs_coeffs.get(), rhs_coeffs.get(), deg_lhs * sizeof(scalar_t)));
  }

  static Polynomial_t vanishing_polynomial(int degree)
  {
    scalar_t coeffs_v[degree + 1] = {0};
    coeffs_v[0] = minus_one; // -1
    coeffs_v[degree] = one;  // +x^n
    auto v = Polynomial_t::from_coefficients(coeffs_v, degree + 1);
    return v;
  }

  static void draw(Polynomial_t& p, bool optimize = false)
  {
    if (optimize) {
      Optimizer optimizer{};
      optimizer.run(p);
    }

    std::ofstream out_file("trace.gv");
    GraphvizVisualizer visualizer{out_file};
    visualizer.run(p);
  }

  static void optimize_trace(Polynomial_t& p)
  {
    Optimizer optimizer{};
    optimizer.run(p);
  }

  static void eval_trace(Polynomial_t& p, bool optimize = false)
  {
    if (optimize) { optimize_trace(p); }

    Interpreter interpreter{m_cuda_factory->create_backend()};
    interpreter.run(p);
  }

  const static inline auto zero = scalar_t::zero();
  const static inline auto one = scalar_t::one();
  const static inline auto two = scalar_t::from(2);
  const static inline auto three = scalar_t::from(3);
  const static inline auto four = scalar_t::from(4);
  const static inline auto five = scalar_t::from(5);
  const static inline auto minus_one = zero - one;
};

TEST_F(PolynomialTest, evaluation)
{
  const scalar_t coeffs[3] = {one, two, three};
  auto f = Polynomial_t::from_coefficients(coeffs, 3);
  scalar_t x = scalar_t::rand_host();

  auto f_x = f(x); // evaluation

  auto expected_f_x = one + two * x + three * x * x;

  EXPECT_EQ(f_x, expected_f_x);
}

TEST_F(PolynomialTest, evaluationOnDomain)
{
  int size = 1 << 5;
  auto f = PolynomialTest::randomize_polynomial(size);

  size *= 2; // evaluating on a larger domain
  auto default_device_context = device_context::get_default_device_context();
  const auto w = ntt::GetRootOfUnity<scalar_t>((int)log2(size), default_device_context);

  // construct domain as rou
  scalar_t x = one;
  auto domain = std::make_unique<scalar_t[]>(size);
  for (int i = 0; i < size; ++i) {
    domain[i] = x;
    x = x * w;
  }

  // evaluate f on the domain (equivalent to NTT)
  auto evaluations = std::make_unique<scalar_t[]>(size);
  f.evaluate_on_domain(domain.get(), size, evaluations.get());

  // construct g from the evaluations of f
  auto g = Polynomial_t::from_rou_evaluations(evaluations.get(), size);

  // check that f==g
  ASSERT_EQ((f - g).degree(), -1);
}

TEST_F(PolynomialTest, fromEvaluations)
{
  const int size = 100;
  const int log_size = (int)ceil(log2(size));
  const int nof_evals = 1 << log_size;
  auto f = randomize_polynomial(size);

  // evaluate f on roots of unity
  scalar_t omega = scalar_t::omega(log_size);
  scalar_t evals[nof_evals] = {0};
  scalar_t x = one;
  for (int i = 0; i < nof_evals; ++i) {
    evals[i] = f(x);
    x = x * omega;
  }

  // construct g from f's evaluations
  auto g = Polynomial_t::from_rou_evaluations(evals, nof_evals);

  // make sure they are equal, that is f-g=0
  auto h = f - g;
  EXPECT_EQ(h.degree(), -1); // degree -1 is the zero polynomial
}

TEST_F(PolynomialTest, fromEvaluationsNotPowerOfTwo)
{
  const int size = 100;
  const int log_size = (int)ceil(log2(size));
  const int nof_evals = size;
  auto f = randomize_polynomial(size);

  // evaluate f on roots of unity
  scalar_t omega = scalar_t::omega(log_size);
  scalar_t evals[nof_evals] = {0};
  scalar_t x = one;
  for (int i = 0; i < nof_evals; ++i) {
    evals[i] = f(x);
    x = x * omega;
  }

  // construct g from f's evaluations
  auto g = Polynomial_t::from_rou_evaluations(evals, nof_evals);

  // since NTT works on a power of two (therefore the extra elements are arbitrary), f!=g but they should evaluate to
  // the same values on the roots of unity due to construction.
  x = one;
  for (int i = 0; i < nof_evals; ++i) {
    EXPECT_EQ(f(x), g(x));
    x = x * omega;
  }
}

TEST_F(PolynomialTest, addition)
{
  const int size_0 = 12, size_1 = 17;
  auto f = randomize_polynomial(size_0);
  auto g = randomize_polynomial(size_1);

  scalar_t x = scalar_t::rand_host();
  auto f_x = f(x);
  auto g_x = g(x);
  auto fx_plus_gx = f_x + g_x;

  auto s = f + g;
  auto s_x = s(x);

  EXPECT_EQ(fx_plus_gx, s_x);
}

TEST_F(PolynomialTest, addition_inplace)
{
  const int size_0 = 2, size_1 = 2;
  auto f = randomize_polynomial(size_0);
  auto g = randomize_polynomial(size_1);

  scalar_t x = scalar_t::rand_host();
  auto f_x = f(x);
  auto g_x = g(x);
  auto fx_plus_gx = f_x + g_x;

  f += g;
  auto h = f + g;

  draw(h);

  auto s_x = f(x);
  auto h_x = h(x);

  EXPECT_EQ(fx_plus_gx, s_x);
  EXPECT_EQ(h_x, s_x + g_x);
}

TEST_F(PolynomialTest, cAPI)
{
  const int size = 3;
  auto coeff = std::make_unique<scalar_t[]>(size);
  random_samples(coeff.get(), size);

  auto f = CONCAT_EXPAND(FIELD, polynomial_create_from_coefficients)(coeff.get(), size);
  auto g = CONCAT_EXPAND(FIELD, polynomial_create_from_coefficients)(coeff.get(), size);
  auto s = CONCAT_EXPAND(FIELD, polynomial_add)(f, g);

  scalar_t x = scalar_t::rand_host();

  auto f_x = CONCAT_EXPAND(FIELD, polynomial_evaluate)(f, x);
  auto g_x = CONCAT_EXPAND(FIELD, polynomial_evaluate)(g, x);
  auto fx_plus_gx = f_x + g_x;
  auto s_x = CONCAT_EXPAND(FIELD, polynomial_evaluate)(s, x);
  EXPECT_EQ(fx_plus_gx, s_x);

  CONCAT_EXPAND(FIELD, polynomial_delete)(f);
  CONCAT_EXPAND(FIELD, polynomial_delete)(g);
  CONCAT_EXPAND(FIELD, polynomial_delete)(s);
}

TEST_F(PolynomialTest, multiplication)
{
  const int size_0 = 1 << 15, size_1 = 1 << 12;
  auto f = randomize_polynomial(size_0);
  auto g = randomize_polynomial(size_1);

  scalar_t x = scalar_t::rand_host();
  auto f_x = f(x);
  auto g_x = g(x);
  auto fx_mul_gx = f_x * g_x;

  START_TIMER(poly_mult_start);
  auto m = f * g;
  END_TIMER(poly_mult_start, "Polynomial multiplication took", MEASURE);

  auto m_x = m(x);

  EXPECT_EQ(fx_mul_gx, m_x);
}

TEST_F(PolynomialTest, multiplicationScalar)
{
  const int size = 1 << 15;
  auto f = randomize_polynomial(size);

  auto g = two * f;
  auto h = f * three;

  scalar_t x = scalar_t::rand_host();
  auto f_x = f(x);
  auto g_x = g(x);
  auto h_x = h(x);

  EXPECT_EQ(g_x, f_x * two);
  EXPECT_EQ(h_x, f_x * three);

  EXPECT_EQ(g.degree(), f.degree());
  EXPECT_EQ(h.degree(), f.degree());
}

TEST_F(PolynomialTest, monomials)
{
  const scalar_t coeffs[3] = {one, zero, two}; // 1+2x^2
  auto f = Polynomial_t::from_coefficients(coeffs, 3);
  const auto x = three;
  const auto expected_f_x = one + two * x * x;
  auto f_x = f(x);

  EXPECT_EQ(f_x, expected_f_x);

  f.add_monomial_inplace(three, 1); // add 3x
  const auto expected_addmonmon_f_x = f_x + three * x;
  const auto addmonom_f_x = f(x);

  EXPECT_EQ(addmonom_f_x, expected_addmonmon_f_x);

  f.sub_monomial_inplace(one); // subtract 1. equivalent to 'f-1'
  const auto expected_submonom_f_x = addmonom_f_x - one;
  const auto submonom_f_x = f(x);

  EXPECT_EQ(submonom_f_x, expected_submonom_f_x);
}

TEST_F(PolynomialTest, ReadCoeffsToHost)
{
  const scalar_t coeffs_f[3] = {zero, one, two}; // x+2x^2
  auto f = Polynomial_t::from_coefficients(coeffs_f, 3);
  const scalar_t coeffs_g[3] = {one, one, one}; // 1+x+x^2
  auto g = Polynomial_t::from_coefficients(coeffs_g, 3);

  auto h = f + g; // 1+2x+3x^3
  const auto h0 = h.copy_coefficient_to_host(0);
  const auto h1 = h.copy_coefficient_to_host(1);
  const auto h2 = h.copy_coefficient_to_host(2);
  EXPECT_EQ(h0, one);
  EXPECT_EQ(h1, two);
  EXPECT_EQ(h2, three);

  int64_t nof_coeffs = h.copy_coefficients_to_host(nullptr); // query #coeffs
  EXPECT_GE(nof_coeffs, 3);                                  // can be larger due to padding to powers of two
  scalar_t h_coeffs[3] = {0};
  nof_coeffs = h.copy_coefficients_to_host(h_coeffs, 0, 2); // read the coefficients
  EXPECT_EQ(nof_coeffs, 3);                                 // expecting 3 due to specified indices

  scalar_t expected_h_coeffs[nof_coeffs] = {one, two, three};
  for (int i = 0; i < nof_coeffs; ++i) {
    EXPECT_EQ(expected_h_coeffs[i], h_coeffs[i]);
  }
}

TEST_F(PolynomialTest, divisionSimple)
{
  const scalar_t coeffs_a[4] = {five, zero, four, three}; // 3x^3+4x^2+5
  const scalar_t coeffs_b[3] = {minus_one, zero, one};    // x^2-1
  auto a = Polynomial_t::from_coefficients(coeffs_a, 4);
  auto b = Polynomial_t::from_coefficients(coeffs_b, 3);

  auto [q, r] = a.divide(b);
  scalar_t q_coeffs[2] = {0}; // 3x+4
  scalar_t r_coeffs[2] = {0}; // 3x+9
  const auto q_nof_coeffs = q.copy_coefficients_to_host(q_coeffs, 0, 1);
  const auto r_nof_coeffs = r.copy_coefficients_to_host(r_coeffs, 0, 1);

  ASSERT_EQ(q_nof_coeffs, 2);
  ASSERT_EQ(r_nof_coeffs, 2);
  ASSERT_EQ(q_coeffs[0], scalar_t::from(4));
  ASSERT_EQ(q_coeffs[1], scalar_t::from(3));
  ASSERT_EQ(r_coeffs[0], scalar_t::from(9));
  ASSERT_EQ(r_coeffs[1], scalar_t::from(3));
}

TEST_F(PolynomialTest, divisionLarge)
{
  const int size_0 = 1 << 12, size_1 = 1 << 2;
  auto a = randomize_polynomial(size_0);
  auto b = randomize_polynomial(size_1);

  START_TIMER(poly_mult_start);
  auto [q, r] = a.divide(b);
  END_TIMER(poly_mult_start, "Polynomial division took", MEASURE);

  scalar_t x = scalar_t::rand_host();
  auto a_x = a(x);
  auto b_x = b(x);
  auto q_x = q(x);
  auto r_x = r(x);

  // a(x) = b(x)*q(x)+r(x)
  EXPECT_EQ(a_x, b_x * q_x + r_x);
}

TEST_F(PolynomialTest, divideByVanishingPolynomial)
{
  const scalar_t coeffs_v[5] = {minus_one, zero, zero, zero, one}; // x^4-1 vanishes on 4th roots of unity
  auto v = Polynomial_t::from_coefficients(coeffs_v, 5);
  auto h = randomize_polynomial(1 << 11, false);
  auto hv = h * v;

  START_TIMER(poly_div_vanishing);
  auto h_div_by_vanishing = hv.divide_by_vanishing_polynomial(4);
  END_TIMER(poly_div_vanishing, "Polynomial division by vanishing (fast) took", MEASURE);
  assert_equal(h_div_by_vanishing, h);

  START_TIMER(poly_div_long);
  auto [h_div, R] = hv.divide(v);
  END_TIMER(poly_div_long, "Polynomial division by vanishing (long division) took", MEASURE);
  assert_equal(h_div, h);
}

TEST_F(PolynomialTest, clone)
{
  const int size = 1 << 10;
  auto f = randomize_polynomial(size);

  const auto x = scalar_t::rand_host();
  const auto f_x = f(x);

  Polynomial_t g;
  g = f.clone(); // operator=(&&)
  g += f;

  auto h = g.clone(); // move constructor

  ASSERT_EQ(g(x), two * f_x);
  ASSERT_EQ(h(x), g(x));
}

TEST_F(PolynomialTest, View)
{
  const int size = 1 << 6;

  auto f = randomize_polynomial(size);
  auto [d_coeff, N, device_id] = f.get_coefficients_view();

  EXPECT_EQ(d_coeff.isValid(), true);
  auto g = f + f;
  // expecting the view to remain valid in that case
  EXPECT_EQ(d_coeff.isValid(), true);

  f += f;
  if (TRACING) {
    // force tracing backend to evalute. Otherwise d_coeff is still valid since f is not computed
    auto x = scalar_t::rand_host();
    f(x);
  }
  // expecting view to be invalidated since f is modified
  EXPECT_EQ(d_coeff.isValid(), false);
}

TEST_F(PolynomialTest, interpolation)
{
  const int size = 1 << 4;
  const int interpolation_size = 1 << 6;

  const auto x = scalar_t::rand_host();

  auto f = randomize_polynomial(size);
  auto [evals, N, device_id] = f.get_rou_evaluations_view(interpolation_size); // interpolate from 16 to 64 evaluations

  auto g = Polynomial_t::from_rou_evaluations(evals.get(), N); // note the evals is a view to f
  const auto fx = f(x);
  ASSERT_EQ(evals.isValid(), false); // invaidated since f(x) transforms f to coefficients

  const auto gx = g(x); // evaluating g which was constructed from interpolation of f
  ASSERT_EQ(fx, gx);
}

TEST_F(PolynomialTest, slicing)
{
  auto body = [](int size) {
    const bool is_odd_size = size % 2 == 1;
    auto f = randomize_polynomial(size);
    const auto x = scalar_t::rand_host();

    // computing e(x) and o(x) directly
    auto expected_even = scalar_t::zero();
    auto expected_odd = scalar_t::zero();
    for (int i = size - 1; i >= 0; --i) {
      if (i % 2 == 0)
        expected_even = expected_even * x + f.copy_coefficient_to_host(i);
      else
        expected_odd = expected_odd * x + f.copy_coefficient_to_host(i);
    }

    auto e = f.even();
    auto o = f.odd();

    ASSERT_EQ(o.degree(), size / 2 - 1);
    ASSERT_EQ(e.degree(), is_odd_size ? size / 2 : size / 2 - 1);

    // compute even and odd polynomials and compute them at x
    ASSERT_EQ(f.even()(x), expected_even);
    ASSERT_EQ(f.odd()(x), expected_odd);
  };

  body(1 << 10);       // test even size
  body((1 << 10) - 1); // test odd size
}

TEST_F(PolynomialTest, tracingInplace)
{
  const int size_0 = 1 << 16, size_1 = 1 << 16;
  auto x = scalar_t::rand_host();
  const bool visualize = false;

  auto computation = [&](bool tracing, bool measure = true) {
    if (tracing) {
      enable_cuda_tracing();
    } else {
      enable_cuda_eager();
    }
    auto f = randomize_polynomial(size_0, false);
    auto g = randomize_polynomial(size_1, false);
    auto h = randomize_polynomial(size_1, false);

    START_TIMER(timer);

    auto t0 = f + g;
    auto t1 = t0 * two;
    auto t2 = t1 - h;
    t1 += h;
    h += g;
    auto res = t1 + t2 + h;

    if (visualize && tracing) {
      // print res trace to file
      draw(res);
    }

    auto eval = res(x);

    END_TIMER(timer, tracing ? "tracing" : "eager", MEASURE && measure);
    return std::make_tuple(std::move(res), res(x));
  };

  // warmup
  computation(false /*=tracing*/, false /*=measure*/);

  auto [p_eager, eager_result] = computation(false /*=tracing*/);
  auto [p_trace, trace_result] = computation(true /*=tracing*/);

  ASSERT_EQ(eager_result, trace_result);
}

TEST_F(PolynomialTest, tracingMemoryReuse)
{
  const int size = 1 << 10;
  const int nof_polys = 5;
  auto x = scalar_t::rand_host();
  const bool visualize = false;

  auto computation = [&](bool tracing, bool measure = true) {
    if (tracing) {
      enable_cuda_tracing();
    } else {
      enable_cuda_eager();
    }
    std::vector<Polynomial_t> polys;
    for (int i = 0; i < nof_polys; ++i) {
      polys.push_back(randomize_polynomial(size, false));
    }
    auto res = randomize_polynomial(size, false);

    START_TIMER(timer);

    for (auto& p : polys) {
      res = res + p;
    }

    if (tracing) {
      if (visualize) draw(res, true /*=optimize*/);
      eval_trace(res, true);
    }
    END_TIMER(timer, tracing ? "tracing" : "eager", MEASURE && measure);

    auto eval = res(x);

    return std::make_tuple(std::move(res), res(x));
  };

  // warmup
  computation(false /*=tracing*/, false /*=measure*/);

  auto [p_eager, eager_result] = computation(false /*=tracing*/);
  auto [p_trace, trace_result] = computation(true /*=tracing*/);

  ASSERT_EQ(eager_result, trace_result);
}

TEST_F(PolynomialTest, MAC)
{
  const int size = 1 << 18;
  const int nof_polys = 1000;
  auto x = scalar_t::rand_host();
  const bool visualize = false;

  auto computation = [&](bool tracing, bool measure = true) {
    if (tracing) {
      enable_cuda_tracing();
    } else {
      enable_cuda_eager();
    }
    std::vector<Polynomial_t> polys;
    for (int i = 0; i < nof_polys; ++i) {
      polys.push_back(randomize_polynomial(size, false));
    }
    auto res = randomize_polynomial(size, false);

    START_TIMER(build);
    for (auto& p : polys) {
      res = res + x * p;
    }
    END_TIMER(build, tracing ? "build-trace" : "eager-execution", MEASURE && measure);
    if (tracing) {
      if (visualize) draw(res, true /*=optimize*/);
      START_TIMER(optimize);
      optimize_trace(res);
      END_TIMER(optimize, "optimize-trace", MEASURE && measure);
      START_TIMER(eval);
      eval_trace(res, true);
      END_TIMER(eval, "eval-optimized-trace", MEASURE && measure);
    }

    auto eval = res(x);

    return std::make_tuple(std::move(res), eval);
  };

  // warmup
  computation(false /*=tracing*/, false /*=measure*/);

  auto [p_eager, eager_result] = computation(false /*=tracing*/);
  auto [p_trace, trace_result] = computation(true /*=tracing*/);

  ASSERT_EQ(eager_result, trace_result);
}

#ifdef CURVE
#include "msm/msm.cuh"
#include "curves/curve_config.cuh"
using curve_config::affine_t;
using curve_config::g2_affine_t;
using curve_config::g2_projective_t;
using curve_config::projective_t;
class dummy_g2_t : public scalar_t
{
public:
  static constexpr __host__ __device__ dummy_g2_t to_affine(const dummy_g2_t& point) { return point; }

  static constexpr __host__ __device__ dummy_g2_t from_affine(const dummy_g2_t& point) { return point; }

  static constexpr __host__ __device__ dummy_g2_t generator() { return dummy_g2_t{scalar_t::one()}; }

  static __host__ __device__ dummy_g2_t zero() { return dummy_g2_t{scalar_t::zero()}; }

  friend __host__ __device__ dummy_g2_t operator*(const scalar_t& xs, const dummy_g2_t& ys)
  {
    return dummy_g2_t{scalar_t::reduce(scalar_t::mul_wide(xs, ys))};
  }

  friend __host__ __device__ dummy_g2_t operator+(const dummy_g2_t& xs, const dummy_g2_t& ys)
  {
    scalar_t rs = {};
    scalar_t::add_limbs<false>(xs.limbs_storage, ys.limbs_storage, rs.limbs_storage);
    return dummy_g2_t{scalar_t::sub_modulus<1>(rs)};
  }
};

// using the MSM C-API directly since msm::MSM() symbol is hidden in icicle lib and I cannot understand why
namespace msm {
  extern "C" cudaError_t CONCAT_EXPAND(CURVE, MSMCuda)(
    const scalar_t* scalars, const affine_t* points, int msm_size, MSMConfig& config, projective_t* out);

  extern "C" cudaError_t CONCAT_EXPAND(CURVE, G2MSMCuda)(
    const scalar_t* scalars, const g2_affine_t* points, int msm_size, MSMConfig& config, g2_projective_t* out);

  cudaError_t _MSM(const scalar_t* scalars, const affine_t* points, int msm_size, MSMConfig& config, projective_t* out)
  {
    return CONCAT_EXPAND(CURVE, MSMCuda)(scalars, points, msm_size, config, out);
  }
  cudaError_t
  _G2MSM(const scalar_t* scalars, const g2_affine_t* points, int msm_size, MSMConfig& config, g2_projective_t* out)
  {
    return CONCAT_EXPAND(CURVE, G2MSMCuda)(scalars, points, msm_size, config, out);
  }

  cudaError_t
  _G2MSM(const scalar_t* scalars, const dummy_g2_t* points, int msm_size, MSMConfig& config, dummy_g2_t* out)
  {
    scalar_t* scalars_host = static_cast<scalar_t*>(malloc(msm_size * sizeof(scalar_t)));
    cudaMemcpyAsync(scalars_host, scalars, msm_size * sizeof(scalar_t), cudaMemcpyDeviceToHost, config.ctx.stream);
    *out = dummy_g2_t::zero();
    for (int i = 0; i < msm_size; i++)
      *out = *out + scalars_host[i] * points[i];
    free(scalars_host);
    return cudaSuccess;
  }

} // namespace msm
#endif // CURVE

// Following examples are randomizing N private numbers and proving that I know N numbers such that their product is
// equal to 'out'.
//
// Circuit:
//
// in0  in1
//  \   /
//   \ /  in2
//   (X)  /
//   t0\ /  in3
//     (X)  /
//     t1\ /
//       (X)
//        .
//        .
//        .
//        |
//       out
//
// simple construction: t0=in0*in1, t1=t0*in2, t2=t1*in3 and so on to simplify the example
template <class S = scalar_t, class G1A = scalar_t, class G1P = scalar_t, class G2A = scalar_t, class G2P = scalar_t>
class Groth16Example
{
  // based on https://www.rareskills.io/post/groth16
public:
  /******** QAP construction *********/

  Groth16Example(int N)
      : nof_inputs(N), nof_outputs(1), nof_intermediates(nof_inputs - 2),
        witness_size(1 + nof_outputs + nof_inputs + nof_intermediates), input_offset(1 + nof_outputs),
        intermediate_offset(input_offset + nof_inputs), nof_constraints(nof_inputs - 1)
  {
    construct_QAP();
  }

  void construct_QAP()
  {
    // (1) construct matrices A,B,C (based on the circuit)
    // allocating such that columns are consecutive in memory for more efficient polynomial construction from
    // consecutive evaluations
    const int nof_cols = witness_size;
    const int nof_rows = ceil_to_power_of_two(nof_constraints);
    std::vector<S> L(nof_cols * nof_rows, S::zero());
    std::vector<S> R(nof_cols * nof_rows, S::zero());
    std::vector<S> O(nof_cols * nof_rows, S::zero());

    S* L_data = L.data();
    S* R_data = R.data();
    S* O_data = O.data();

    // filling the R1CS matrices (where cols are consecutive, not rows)
    for (int row = 0; row < nof_constraints; ++row) {
      const int L_col = row == 0 ? input_offset : intermediate_offset + row - 1;
      *(L_data + L_col * nof_rows + row) = S::one();

      const int R_col = input_offset + row + 1;
      *(R_data + R_col * nof_rows + row) = S::one();

      const int O_col = row == nof_constraints - 1 ? 1 : intermediate_offset + row;
      *(O_data + O_col * nof_rows + row) = S::one();
    }

    // (2) interpolate the columns of L,R,O to build the polynomials
    L_QAP.reserve(nof_cols);
    R_QAP.reserve(nof_cols);
    O_QAP.reserve(nof_cols);
    for (int col = 0; col < nof_cols; ++col) { // #polynomials is equal to witness_size
      L_QAP.push_back(std::move(Polynomial_t::from_rou_evaluations(L_data + col * nof_rows, nof_rows)));
      R_QAP.push_back(std::move(Polynomial_t::from_rou_evaluations(R_data + col * nof_rows, nof_rows)));
      O_QAP.push_back(std::move(Polynomial_t::from_rou_evaluations(O_data + col * nof_rows, nof_rows)));
    }
  }

  std::vector<S> random_witness_inputs()
  {
    std::vector<S> witness(witness_size, S::zero());
    witness[0] = S::one();
    PolynomialTest::random_samples(witness.data() + input_offset, nof_inputs); // randomize inputs

    return witness;
  }

  void compute_witness(std::vector<S>& witness)
  {
    if (witness_size != witness.size()) { throw std::runtime_error("invalid witness size"); }
    // compute intermediate values (based on the circuit above)
    for (int i = 0; i < nof_intermediates; ++i) {
      const auto& left_input = i == 0 ? witness[input_offset] : witness[intermediate_offset + i - 1];
      const auto& right_input = witness[input_offset + i + 1];
      witness[intermediate_offset + i] = left_input * right_input;
    }
    // compute output as last_input * last_intermediate
    witness[1] = witness[input_offset + nof_inputs - 1] * witness[intermediate_offset + nof_intermediates - 1];
  }

  const int nof_inputs;
  const int nof_outputs;
  const int nof_intermediates;
  const int witness_size;
  const int input_offset;
  const int intermediate_offset;
  const int nof_constraints;
  std::vector<Polynomial_t> L_QAP, R_QAP, O_QAP;

#ifdef CURVE
  /******** SETUP *********/
  // https://static.wixstatic.com/media/935a00_cd68860dafbb4ebe8f166de5cc8cc50c~mv2.png
  struct ToxicWaste {
    S alpha;
    S beta;
    S gamma;
    S delta;
    S tau;
    S gamma_inv;
    S delta_inv;

    ToxicWaste()
    {
      alpha = S::rand_host();
      beta = S::rand_host();
      gamma = S::rand_host();
      delta = S::rand_host();
      tau = S::rand_host();
      gamma_inv = S::inverse(gamma);
      delta_inv = S::inverse(delta);
    }
  };

  struct ProvingKey {
    struct _G1 {
      G1A alpha;
      G1A beta;
      G1A delta;
      std::vector<G1A> powers_of_tau;          // {X^i} @[0..n-1]
      std::vector<G1A> private_witness_points; // {(beta_Ui+alpha_Vi+Wi) / delta} @[l+1..m]
      std::vector<G1A> vanishing_poly_points;  // {x^it(x) / delta} @[0..,n-2]
    };
    struct _G2 {
      G2A beta;
      G2A gamma;
      G2A delta;
      std::vector<G2A> powers_of_tau; // {X^i} @[0..n-1]
    };

    _G1 g1;
    _G2 g2;
  };

  struct VerifyingKey {
    struct _G1 {
      G1A alpha;
      std::vector<G1A> public_witness_points; // {(beta_Ui+alpha_Vi+Wi) / delta} @[0..l]
    };
    struct _G2 {
      G2A beta;
      G2A gamma;
      G2A delta;
    };

    _G1 g1;
    _G2 g2;
  };

  void setup()
  {
    // randomize alpha, beta, gamma, delta, tau
    ToxicWaste toxic_waste;

    G1P g1 = G1P::generator();
    G2P g2 = G2P::generator();

    // Note: n,m,l are from the groth16 paper
    const int m = witness_size - 1;
    const int l = nof_outputs; // public part of the witness
    const int n = ceil_to_power_of_two(nof_constraints);

    // compute the proving and verifying keys
    pk.g1.alpha = G1P::to_affine(toxic_waste.alpha * g1);
    vk.g1.alpha = pk.g1.alpha;
    pk.g1.beta = G1P::to_affine(toxic_waste.beta * g1);
    pk.g1.delta = G1P::to_affine(toxic_waste.delta * g1);

    pk.g1.powers_of_tau.resize(n, G1A::zero());
    PolynomialTest::compute_powers_of_tau(g1, toxic_waste.tau, pk.g1.powers_of_tau.data(), n);

    // { (beta*Ui(tau) + alpha*Vi(tau) + Wi) / delta}
    pk.g1.private_witness_points.reserve(m - l);
    vk.g1.public_witness_points.reserve(l + 1);
    for (int i = 0; i <= m; ++i) {
      auto p = toxic_waste.beta * L_QAP[i] + toxic_waste.alpha * R_QAP[i] + O_QAP[i];
      p = p * (i < l + 1 ? toxic_waste.gamma_inv : toxic_waste.delta_inv);
      if (i < l + 1)
        vk.g1.public_witness_points.push_back(G1P::to_affine(p(toxic_waste.tau) * g1));
      else
        pk.g1.private_witness_points.push_back(G1P::to_affine(p(toxic_waste.tau) * g1));
    }

    // {tau^i(t(tau) / delta}
    const int vanishing_poly_deg = n;
    auto t = PolynomialTest::vanishing_polynomial(vanishing_poly_deg);
    pk.g1.vanishing_poly_points.reserve(n - 1);
    auto x = S::one();
    for (int i = 0; i <= n - 2; ++i) {
      pk.g1.vanishing_poly_points.push_back(G1P::to_affine(x * t(toxic_waste.tau) * toxic_waste.delta_inv * g1));
      x = x * toxic_waste.tau;
    }

    pk.g2.beta = G2P::to_affine(toxic_waste.beta * g2);
    vk.g2.beta = pk.g2.beta;
    pk.g2.gamma = G2P::to_affine(toxic_waste.gamma * g2);
    vk.g2.gamma = pk.g2.gamma;
    pk.g2.delta = G2P::to_affine(toxic_waste.delta * g2);
    vk.g2.delta = pk.g2.delta;

    pk.g2.powers_of_tau.resize(n, G2A::zero());
    PolynomialTest::compute_powers_of_tau(g2, toxic_waste.tau, pk.g2.powers_of_tau.data(), n);
  }

  /******** PROVE *********/
  // https://static.wixstatic.com/media/935a00_432ca182820540df8d67b5c3d5d0d3e1~mv2.png
  struct G16proof {
    G1A A;
    G2A B;
    G1A C;
  };

  G16proof prove(const std::vector<S>& witness) const
  {
    G16proof proof = {};
    const auto r = S::rand_host();
    const auto s = S::rand_host();

    // Note: n,m,l are from the groth16 paper
    const int m = witness_size - 1;
    const int l = nof_outputs; // public part of the witness
    const int n = ceil_to_power_of_two(nof_constraints);

    // construct U,V,W from the QAP and witness
    Polynomial_t U = L_QAP[0].clone();
    Polynomial_t V = R_QAP[0].clone();
    Polynomial_t W = O_QAP[0].clone();
    for (int col = 1; col <= m; ++col) {
      U += witness[col] * L_QAP[col];
      V += witness[col] * R_QAP[col];
      W += witness[col] * O_QAP[col];
    }

    // compute h(x) = (U(x)*V(x)-W(x)) / t(x)
    const int vanishing_poly_deg = n;
    Polynomial_t h = (U * V - W).divide_by_vanishing_polynomial(vanishing_poly_deg);

    auto msm_config = msm::DefaultMSMConfig<S>();
    msm_config.are_scalars_on_device = true;

    // compute [A]1
    {
      G1P U_commited;
      auto [U_coeff, N, device_id] = U.get_coefficients_view();
      CHK_STICKY(msm::_MSM(U_coeff.get(), pk.g1.powers_of_tau.data(), n, msm_config, &U_commited));
      proof.A = G1P::to_affine(U_commited + G1P::from_affine(pk.g1.alpha) + r * G1P::from_affine(pk.g1.delta));
    }

    // compute [B]2 and [B]1 (required to compute C)
    G1P B1;
    {
      G2P V_commited_g2;
      auto [V_coeff, N, device_id] = V.get_coefficients_view();
      CHK_STICKY(msm::_G2MSM(V_coeff.get(), pk.g2.powers_of_tau.data(), n, msm_config, &V_commited_g2));
      proof.B = G2P::to_affine(V_commited_g2 + pk.g2.beta + s * G2P::from_affine(pk.g2.delta));

      G1P V_commited_g1;
      CHK_STICKY(msm::_MSM(V_coeff.get(), pk.g1.powers_of_tau.data(), n, msm_config, &V_commited_g1));
      B1 = V_commited_g1 + pk.g1.beta + G1P::from_affine(pk.g1.delta) * s;
    }

    // compute [C]1
    {
      auto [H_coeff, N, device_id] = h.get_coefficients_view();

      G1P HT_commited;
      CHK_STICKY(msm::_MSM(H_coeff.get(), pk.g1.vanishing_poly_points.data(), n - 1, msm_config, &HT_commited));

      G1P private_inputs_commited;
      msm_config.are_scalars_on_device = false;
      CHK_STICKY(msm::_MSM(
        witness.data() + l + 1, pk.g1.private_witness_points.data(), m - l, msm_config, &private_inputs_commited));

      proof.C = G1P::to_affine(
        private_inputs_commited + HT_commited + G1P::from_affine(proof.A) * s + B1 * r -
        r * s * G1P::from_affine(pk.g1.delta));
    }

    return proof;
  }

  bool verify(const G16proof& proof, const std::vector<S>& public_witness) const
  {
    throw std::runtime_error("pairing not implemented");
    return false;
  }

  // Dummy verification function where pairings are changed to scalar multiplications
  // Suitable for verifying correctness with G1 and/or G2 swapped for scalar types
  bool dummy_verify(const G16proof& proof, const std::vector<S>& public_witness) const
  {
    G1P lhs = proof.B * G1P::from_affine(proof.A);
    G1P rhs = G1P::zero();
    for (int i = 0; i <= nof_outputs; ++i)
      rhs = rhs + public_witness.data()[i] * G1P::from_affine(vk.g1.public_witness_points[i]);
    rhs = vk.g2.gamma * rhs + vk.g2.beta * G1P::from_affine(vk.g1.alpha) + vk.g2.delta * G1P::from_affine(proof.C);
    return (rhs == lhs);
  }

  ProvingKey pk;
  VerifyingKey vk;
#endif // CURVE
};

TEST_F(PolynomialTest, QAP)
{
  // (1) construct R1CS and QAP for circuit with N inputs
  const int N = 3000;
  Groth16Example QAP(N);

  // (2) compute witness: randomize inputs and compute other entries [1,out,...N inputs..., ... intermediate values...]
  auto witness = QAP.random_witness_inputs();
  QAP.compute_witness(witness);

  // (3) compute L(x),R(x),O(x) using the witness
  const int nof_cols = QAP.witness_size;

  Polynomial_t Lx = QAP.L_QAP[0].clone();
  Polynomial_t Rx = QAP.R_QAP[0].clone();
  Polynomial_t Ox = QAP.O_QAP[0].clone();
  for (int col = 1; col < nof_cols; ++col) {
    Lx += witness[col] * QAP.L_QAP[col];
    Rx += witness[col] * QAP.R_QAP[col];
    Ox += witness[col] * QAP.O_QAP[col];
  }

  const int nof_constraints =
    QAP.nof_inputs - 1; // multiplying N numbers yields N-1 constraints for this circuit construction
  const int vanishing_poly_deg = ceil_to_power_of_two(nof_constraints);
  // (4) compute h(x) as '(L(x)R(x)-O(x)) / t(x)'
  Polynomial_t h = (Lx * Rx - Ox).divide_by_vanishing_polynomial(vanishing_poly_deg);

  if (!TRACING && N <= 10) { // only draw small graphs
    draw(h);
  }
  // (5) sanity check: vanishing-polynomial divides (LR-O) without remainder
  {
    auto [h_long_div, r] = (Lx * Rx - Ox).divide(vanishing_polynomial(vanishing_poly_deg));
    EXPECT_EQ(r.degree(), -1); // zero polynomial (expecting division without remainder)
    assert_equal(h, h_long_div);
  }

  //  (6) sanity check: verify AB=C at the evaluation points
  auto default_device_context = device_context::get_default_device_context();
  const auto w = ntt::GetRootOfUnity<scalar_t>((int)ceil(log2(nof_constraints)), default_device_context);
  auto x = scalar_t::one();
  for (int i = 0; i < vanishing_poly_deg; ++i) {
    ASSERT_EQ(Lx(x) * Rx(x), Ox(x));
    x = x * w;
  }
}

#ifdef CURVE
TEST_F(PolynomialTest, commitMSM)
{
  const int size = 1 << 6;
  auto f = randomize_polynomial(size);

  auto [d_coeff, N, device_id] = f.get_coefficients_view();
  auto msm_config = msm::DefaultMSMConfig<scalar_t>();
  msm_config.are_scalars_on_device = true;

  auto points = std::make_unique<affine_t[]>(size);
  projective_t result;

  auto tau = scalar_t::rand_host();
  projective_t g = projective_t::rand_host();
  compute_powers_of_tau(g, tau, points.get(), size);

  EXPECT_EQ(d_coeff.isValid(), true);
  CHK_STICKY(msm::_MSM(d_coeff.get(), points.get(), size, msm_config, &result));

  EXPECT_EQ(result, f(tau) * g);

  if (!TRACING) {
    f += f; // this is invalidating the d_coeff integrity-pointer
    EXPECT_EQ(d_coeff.isValid(), false);
  }
}

TEST_F(PolynomialTest, Groth16)
{
  // (1) construct R1CS and QAP for circuit with N inputs
  Groth16Example<scalar_t, affine_t, projective_t, g2_affine_t, g2_projective_t> groth16_example(30 /*=N*/);

  // (2) compute witness: randomize inputs and compute other entries [1,out,...N inputs..., ... intermediate
  // values...]
  auto witness = groth16_example.random_witness_inputs();
  groth16_example.compute_witness(witness);

  groth16_example.setup();
  auto proof = groth16_example.prove(witness);
  // groth16_example.verify(proof); // cannot implement without pairing
}

TEST_F(PolynomialTest, DummyGroth16)
{
  // (1) construct R1CS and QAP for circuit with N inputs
  Groth16Example<scalar_t, affine_t, projective_t, dummy_g2_t, dummy_g2_t> groth16_example(30 /*=N*/);

  // (2) compute witness: randomize inputs and compute other entries [1,out,...N inputs..., ... intermediate
  // values...]
  auto witness = groth16_example.random_witness_inputs();
  groth16_example.compute_witness(witness);

  groth16_example.setup();
  auto proof = groth16_example.prove(witness);
  ASSERT_EQ(groth16_example.dummy_verify(proof, witness), true);
}
#endif // CURVE

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}