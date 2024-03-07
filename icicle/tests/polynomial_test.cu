
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <vector>
#include <list>

#include "curves/curve_config.cuh"
using curve_config::affine_t;
using curve_config::projective_t;
using curve_config::scalar_t;

#include "polynomials/polynomials.h"
#include "appUtils/ntt/ntt.cuh"
#include "appUtils/msm/msm.cuh"

// using the MSM C-API directly since msm::MSM() is hidden and I cannot understand why
namespace msm {
  extern "C" cudaError_t CONCAT_EXPAND(CURVE, MSMCuda)(
    curve_config::scalar_t* scalars,
    curve_config::affine_t* points,
    int msm_size,
    MSMConfig& config,
    curve_config::projective_t* out);

  cudaError_t __MSM__(
    curve_config::scalar_t* scalars,
    curve_config::affine_t* points,
    int msm_size,
    MSMConfig& config,
    curve_config::projective_t* out)
  {
    return CONCAT_EXPAND(CURVE, MSMCuda)(scalars, points, msm_size, config, out);
  }
} // namespace msm

using FpMicroseconds = std::chrono::duration<float, std::chrono::microseconds::period>;
#define START_TIMER(timer) auto timer##_start = std::chrono::high_resolution_clock::now();
#define END_TIMER(timer, msg, enable)                                                                                  \
  if (enable)                                                                                                          \
    printf(                                                                                                            \
      "%s: %.3f ms\n", msg, FpMicroseconds(std::chrono::high_resolution_clock::now() - timer##_start).count() / 1000);

using namespace polynomials;

typedef Polynomial<scalar_t> Polynomial_t;

class PolynomialTest : public ::testing::Test
{
public:
  static inline const int MAX_NTT_LOG_SIZE = 24;
  static inline const bool MEASURE = true;

  // SetUpTestSuite/TearDownTestSuite are called once for the entire test suite
  static void SetUpTestSuite()
  {
    // init NTT domain
    auto ntt_config = ntt::DefaultNTTConfig<scalar_t>();
    const scalar_t basic_root = scalar_t::omega(MAX_NTT_LOG_SIZE);
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

  static void compute_powers_of_tau(projective_t g, scalar_t tau, affine_t* res, uint32_t count)
  {
    // no arithmetic on affine??
    res[0] = projective_t::to_affine(g);
    for (int i = 1; i < count; i++) {
      g = g * tau;
      res[i] = projective_t::to_affine(g);
    }
  }

  static void assert_equal(Polynomial_t& lhs, Polynomial_t& rhs)
  {
    const int deg_lhs = lhs.degree();
    const int deg_rhs = rhs.degree();
    ASSERT_EQ(deg_lhs, deg_rhs);

    auto lhs_coeffs = std::make_unique<scalar_t[]>(deg_lhs);
    auto rhs_coeffs = std::make_unique<scalar_t[]>(deg_rhs);
    lhs.get_coefficients_on_host(lhs_coeffs.get(), 1, deg_lhs - 1);
    rhs.get_coefficients_on_host(rhs_coeffs.get(), 1, deg_rhs - 1);

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

  scalar_t r = scalar_t::rand_host();

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
  auto s_x = f(x);

  EXPECT_EQ(fx_plus_gx, s_x);
}

TEST_F(PolynomialTest, cAPI)
{
  const int size = 3;
  auto coeff = std::make_unique<scalar_t[]>(size);
  random_samples(coeff.get(), size);

  auto f = polynomial_create_from_coefficients(coeff.get(), size);
  auto g = polynomial_create_from_coefficients(coeff.get(), size);
  auto s = polynomial_add(f, g);

  scalar_t x = scalar_t::rand_host();
  // TODO Yuval: use C-API for evaluate too
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
  const auto h0 = h.get_coefficient_on_host(0);
  const auto h1 = h.get_coefficient_on_host(1);
  const auto h2 = h.get_coefficient_on_host(2);
  EXPECT_EQ(h0, one);
  EXPECT_EQ(h1, two);
  EXPECT_EQ(h2, three);

  int64_t nof_coeffs = h.get_coefficients_on_host(nullptr); // query #coeffs
  EXPECT_GE(nof_coeffs, 3);                                 // can be larger due to padding to powers of two
  scalar_t h_coeffs[3] = {0};
  nof_coeffs = h.get_coefficients_on_host(h_coeffs, 0, 2); // read the coefficients
  EXPECT_EQ(nof_coeffs, 3);                                // expecting 3 due to specified indices

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
  const auto q_nof_coeffs = q.get_coefficients_on_host(q_coeffs, 0, 1);
  const auto r_nof_coeffs = r.get_coefficients_on_host(r_coeffs, 0, 1);

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

  START_TIMER(poly_div_long);
  auto [h_div, R] = hv.divide(v);
  END_TIMER(poly_div_long, "Polynomial division by vanishing (long division) took", MEASURE);
  assert_equal(h_div, h);

  START_TIMER(poly_div_vanishing);
  auto h_div_by_vanishing = hv.divide_by_vanishing_polynomial(4);
  END_TIMER(poly_div_vanishing, "Polynomial division by vanishing (fast) took", MEASURE);
  assert_equal(h_div_by_vanishing, h);
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

TEST_F(PolynomialTest, commitMSM)
{
  const int size = 1 << 6;
  auto f = randomize_polynomial(size);

  auto [d_coeff, N, device_id] = f.get_coefficients_on_device();
  auto msm_config = msm::DefaultMSMConfig();
  msm_config.are_scalars_on_device = true;

  auto points = std::make_unique<affine_t[]>(size);
  projective_t result;

  auto tau = scalar_t::rand_host();
  projective_t g = projective_t::rand_host();
  compute_powers_of_tau(g, tau, points.get(), size);

  msm::__MSM__(d_coeff, points.get(), size, msm_config, &result);

  EXPECT_EQ(result, f(tau) * g);
}

// TODO Yuval: move to examples ??
TEST_F(PolynomialTest, QAP)
{
  // this examples is randomizing N private numbers and proving that I know N numbers such that their product is equal
  // to 'out'.
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

  // (1) randomize N numbers and construct the witness as [1,out,...N inputs..., ... intermediate values...]
  const int nof_inputs = 5;
  const int nof_outputs = 1;
  const int nof_intermediates = nof_inputs - 2; // same as #multiplication gates minus last one (which is the output)
  const int witness_size =
    1 + nof_outputs + nof_inputs + nof_intermediates; // witness = [1, out, inputs..., intermediates...]

  const int input_offset = 1 + nof_outputs;
  const int intermediate_offset = input_offset + nof_inputs;

  std::vector<scalar_t> witness(witness_size, scalar_t::zero());
  witness[0] = scalar_t::one();
  random_samples(witness.data() + input_offset, nof_inputs); // randomize inputs
  // compute intermediate values (based on the circuit above)
  for (int i = 0; i < nof_intermediates; ++i) {
    const auto& left_input = i == 0 ? witness[input_offset] : witness[intermediate_offset + i - 1];
    const auto& right_input = witness[input_offset + i + 1];
    witness[intermediate_offset + i] = left_input * right_input;
  }
  // compute output as last_input * last_intermediate
  witness[1] = witness[input_offset + nof_inputs - 1] * witness[intermediate_offset + nof_intermediates - 1];

  // (2) construct matrices A,B,C (based on the circuit)
  const int nof_constraints = nof_inputs - 1;
  // allocating such that columns are consecutive in memory for more efficient polynomial construction from consecutive
  // evaluations
  const int nof_cols = witness_size;
  const int nof_rows = nof_constraints;
  std::vector<scalar_t> L(nof_cols * nof_rows, scalar_t::zero());
  std::vector<scalar_t> R(nof_cols * nof_rows, scalar_t::zero());
  std::vector<scalar_t> O(nof_cols * nof_rows, scalar_t::zero());

  scalar_t* L_data = L.data();
  scalar_t* R_data = R.data();
  scalar_t* O_data = O.data();

  // filling the R1CS matrices (where cols are consecutive, not rows)
  for (int row = 0; row < nof_rows; ++row) {
    const int L_col = row == 0 ? input_offset : intermediate_offset + row - 1;
    *(L_data + L_col * nof_rows + row) = scalar_t::one();

    const int R_col = input_offset + row + 1;
    *(R_data + R_col * nof_rows + row) = scalar_t::one();

    const int O_col = row == nof_rows - 1 ? 1 : intermediate_offset + row;
    *(O_data + O_col * nof_rows + row) = scalar_t::one();
  }

  // (3) interpolate the columns of L,R,O to build the polynomials
  std::vector<Polynomial_t> L_QAP, R_QAP, O_QAP;
  L_QAP.reserve(nof_cols);
  R_QAP.reserve(nof_cols);
  O_QAP.reserve(nof_cols);
  for (int col = 0; col < nof_cols; ++col) { // #polynomials is equal to witness_size
    L_QAP.push_back(std::move(Polynomial_t::from_rou_evaluations(L_data + col * nof_rows, nof_rows)));
    R_QAP.push_back(std::move(Polynomial_t::from_rou_evaluations(R_data + col * nof_rows, nof_rows)));
    O_QAP.push_back(std::move(Polynomial_t::from_rou_evaluations(O_data + col * nof_rows, nof_rows)));
  }

  // (4) using the witness, compute L(x),R(x),O(x)
  Polynomial_t Lx = L_QAP[0].clone();
  Polynomial_t Rx = R_QAP[0].clone();
  Polynomial_t Ox = O_QAP[0].clone();
  std::cout << "Lx.degree()=" << Lx.degree() << std::endl;
  for (int col = 1; col < nof_cols; ++col) {
    Lx += witness[col] * L_QAP[col];
    Rx += witness[col] * R_QAP[col];
    Ox += witness[col] * O_QAP[col];
    std::cout << "Lx[col].degree()=" << L_QAP[col].degree() << std::endl;
  }

  //  (4b) sanity check: verify that it divides with no remainder
  {
    auto v = vanishing_polynomial(nof_constraints - 1 /*=degree*/);
    std::cout << "Lx.degree()=" << Lx.degree() << std::endl;
    std::cout << "Rx.degree()=" << Rx.degree() << std::endl;
    std::cout << "Ox.degree()=" << Ox.degree() << std::endl;
    std::cout << "LxRx.degree()=" << (Lx * Rx).degree() << std::endl;
    std::cout << "LxRx-Ox.degree()=" << (Lx * Rx - Ox).degree() << std::endl;
    std::cout << "v.degree()=" << v.degree() << std::endl;
    auto [q, r] = (Lx * Rx - Ox).divide(v);
    Polynomial_t h = (Lx * Rx - Ox).divide_by_vanishing_polynomial(nof_constraints);
    std::cout << "q.degree()=" << q.degree() << std::endl;
    std::cout << "r.degree()=" << r.degree() << std::endl;
    std::cout << "h.degree()=" << h.degree() << std::endl;
    EXPECT_EQ(r.degree(), -1);              // zero polynomial
    EXPECT_EQ(q.degree(), nof_constraints); // L(X)R(x)-O(x) is degree 2N so expecting N after division
  }

  // (5) compute h(x) as '(L(x)R(x)-O(x)) / t(x)'
  Polynomial_t h = (Lx * Rx - Ox).divide_by_vanishing_polynomial(nof_constraints);

  // (6) compute A,B,C via MSMs
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}