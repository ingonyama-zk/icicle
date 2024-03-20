
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <vector>
#include <list>

#include "curves/curve_config.cuh"
using curve_config::affine_t;
using curve_config::g2_affine_t;
using curve_config::g2_projective_t;
using curve_config::projective_t;
using curve_config::scalar_t;

#include "polynomials/polynomials.h"
#include "appUtils/ntt/ntt.cuh"
#include "appUtils/msm/msm.cuh"
#include "utils/device_context.cuh"

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

} // namespace msm

/*******************************************/

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

  template <typename P, typename A>
  static void compute_powers_of_tau(P g, scalar_t tau, A* res, uint32_t count)
  {
    // no arithmetic on affine??
    res[0] = P::to_affine(g);
    for (int i = 1; i < count; i++) {
      g = g * tau;
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
  const auto h0 = h.copy_coefficient_to_host(0);
  const auto h1 = h.copy_coefficient_to_host(1);
  const auto h2 = h.copy_coefficient_to_host(2);
  EXPECT_EQ(h0, one);
  EXPECT_EQ(h1, two);
  EXPECT_EQ(h2, three);

  int64_t nof_coeffs = h.copy_coefficients_to_host(nullptr); // query #coeffs
  EXPECT_GE(nof_coeffs, 3);                                 // can be larger due to padding to powers of two
  scalar_t h_coeffs[3] = {0};
  nof_coeffs = h.copy_coefficients_to_host(h_coeffs, 0, 2); // read the coefficients
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

  f += f; // this is invalidating the d_coeff integrity-pointer

  EXPECT_EQ(d_coeff.isValid(), false);
}

TEST_F(PolynomialTest, integrityPointerInvalidation)
{
  const int size = 1 << 6;

  auto f = new Polynomial_t(randomize_polynomial(size));
  auto [d_coeff, N, device_id] = f->get_coefficients_view();

  EXPECT_EQ(d_coeff.isValid(), true);

  delete f; // f is destructed so the coefficients should be invalidated
  EXPECT_EQ(d_coeff.isValid(), false);
}

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

  std::vector<scalar_t> random_witness_inputs()
  {
    std::vector<scalar_t> witness(witness_size, scalar_t::zero());
    witness[0] = scalar_t::one();
    PolynomialTest::random_samples(witness.data() + input_offset, nof_inputs); // randomize inputs

    return witness;
  }

  void compute_witness(std::vector<scalar_t>& witness)
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

  /******** SETUP *********/
  // https://static.wixstatic.com/media/935a00_cd68860dafbb4ebe8f166de5cc8cc50c~mv2.png
  struct ToxicWaste {
    scalar_t alpha;
    scalar_t beta;
    scalar_t gamma;
    scalar_t delta;
    scalar_t tau;
    scalar_t gamma_inv;
    scalar_t delta_inv;

    ToxicWaste()
    {
      alpha = scalar_t::rand_host();
      beta = scalar_t::rand_host();
      gamma = scalar_t::rand_host();
      delta = scalar_t::rand_host();
      tau = scalar_t::rand_host();
      gamma_inv = scalar_t::inverse(gamma);
      delta_inv = scalar_t::inverse(delta);
    }
  };

  struct ProvingKey {
    struct G1 {
      affine_t alpha;
      affine_t beta;
      affine_t delta;
      std::vector<affine_t> powers_of_tau;          // {X^i} @[0..n-1]
      std::vector<affine_t> private_witness_points; // {(beta_Ui+alpha_Vi+Wi) / delta} @[l+1..m]
      std::vector<affine_t> vanishing_poly_points;  // {x^it(x) / delta} @[0..,n-2]
    };
    struct G2 {
      g2_affine_t beta;
      g2_affine_t gamma;
      g2_affine_t delta;
      std::vector<g2_affine_t> powers_of_tau; // {X^i} @[0..n-1]
    };

    G1 g1;
    G2 g2;
  };

  struct VerifyingKey {
    // TODO
  };

  void setup()
  {
    // (1) randomize alpha, beta, gamma, delta, tau
    ToxicWaste toxic_waste;
    // (2) randomize generators G1, G2. TODO Yuval what are the generators used by the protocol?
    projective_t G1 = projective_t::rand_host();
    g2_projective_t G2 = g2_projective_t::rand_host();

    // Note: n,m,l are from the groth16 paper
    const int m = witness_size - 1;
    const int l = nof_outputs; // public part of the witness
    const int n = nof_constraints;

    // (3) compute the proving and verifying keys
    pk.g1.alpha = projective_t::to_affine(toxic_waste.alpha * G1);
    pk.g1.beta = projective_t::to_affine(toxic_waste.beta * G1);
    pk.g1.delta = projective_t::to_affine(toxic_waste.delta * G1);

    pk.g1.powers_of_tau.resize(n, affine_t::zero());
    PolynomialTest::compute_powers_of_tau(G1, toxic_waste.tau, pk.g1.powers_of_tau.data(), n);

    // { (beta*Ui(tau) + alpha*Vi(tau) + Wi) / delta}
    pk.g1.private_witness_points.reserve(m - l);
    for (int i = l + 1; i <= m; ++i) {
      auto p = toxic_waste.beta * L_QAP[i] + toxic_waste.alpha * R_QAP[i] + O_QAP[i];
      p = p * toxic_waste.delta_inv;
      pk.g1.private_witness_points.push_back(projective_t::to_affine(p(toxic_waste.tau) * G1));
    }

    // {tau^i(t(tau) / delta}
    const int vanishing_poly_deg = ceil_to_power_of_two(n);
    auto t = PolynomialTest::vanishing_polynomial(vanishing_poly_deg);
    pk.g1.vanishing_poly_points.reserve(n - 1);
    auto x = scalar_t::one();
    for (int i = 0; i <= n - 2; ++i) {
      pk.g1.vanishing_poly_points.push_back(
        projective_t::to_affine(x * t(toxic_waste.tau) * toxic_waste.delta_inv * G1));
      x = x * toxic_waste.tau;
    }

    pk.g2.beta = g2_projective_t::to_affine(toxic_waste.beta * G2);
    pk.g2.gamma = g2_projective_t::to_affine(toxic_waste.gamma * G2);
    pk.g2.delta = g2_projective_t::to_affine(toxic_waste.delta * G2);

    pk.g2.powers_of_tau.resize(n, g2_affine_t::zero());
    PolynomialTest::compute_powers_of_tau(G2, toxic_waste.tau, pk.g2.powers_of_tau.data(), n);
  }

  /******** PROVE *********/
  // https://static.wixstatic.com/media/935a00_432ca182820540df8d67b5c3d5d0d3e1~mv2.png
  struct G16proof {
    affine_t A;
    g2_affine_t B;
    affine_t C;
  };

  // TODO Yuval: both witness and the method should be const but need to fix MSM to take const inputs first
  G16proof prove(std::vector<scalar_t>& witness)
  {
    G16proof proof = {};
    const auto r = scalar_t::rand_host();
    const auto s = scalar_t::rand_host();

    // Note: n,m,l are from the groth16 paper
    const int m = witness_size - 1;
    const int l = nof_outputs; // public part of the witness
    const int n = nof_constraints;

    auto U = L_QAP[0].clone();
    auto V = R_QAP[0].clone();
    for (int i = 1; i <= m; ++i) {
      U += L_QAP[i] * witness[i];
      V += R_QAP[i] * witness[i];
    }

    auto msm_config = msm::DefaultMSMConfig<scalar_t>();
    msm_config.are_scalars_on_device = true;

    // compute [A]1
    {
      projective_t U_commited;
      auto [d_coeff, N, device_id] = U.get_coefficients_view();
      CHK_STICKY(msm::_MSM(d_coeff.get(), pk.g1.powers_of_tau.data(), n, msm_config, &U_commited));
      proof.A = projective_t::to_affine(U_commited + pk.g1.alpha + r * projective_t::from_affine(pk.g1.delta));
    }

    // compute [B]2 and [B]1 (required to compute C)
    projective_t B1;
    {
      g2_projective_t V_commited_g2;
      auto [d_coeff, N, device_id] = V.get_coefficients_view();
      CHK_STICKY(msm::_G2MSM(d_coeff.get(), pk.g2.powers_of_tau.data(), n, msm_config, &V_commited_g2));
      proof.B = g2_projective_t::to_affine(V_commited_g2 + pk.g2.beta + s * g2_projective_t::from_affine(pk.g2.delta));

      projective_t V_commited_g1;
      CHK_STICKY(msm::_MSM(d_coeff.get(), pk.g1.powers_of_tau.data(), n, msm_config, &V_commited_g1));
      B1 = V_commited_g1 + pk.g1.beta + projective_t::from_affine(pk.g1.delta) * s;
    }

    // compute [C]1
    {
      // compute h. TODO Yuval: is this computation still valid even with the shifts (alpha, beta)?
      Polynomial_t U = L_QAP[0].clone();
      Polynomial_t V = R_QAP[0].clone();
      Polynomial_t W = O_QAP[0].clone();
      for (int col = 1; col <= m; ++col) {
        U += witness[col] * L_QAP[col];
        V += witness[col] * R_QAP[col];
        W += witness[col] * O_QAP[col];
      }

      const int vanishing_poly_deg = ceil_to_power_of_two(n);
      Polynomial_t h = (U * V - W).divide_by_vanishing_polynomial(vanishing_poly_deg);
      auto [d_coeff, N, device_id] = h.get_coefficients_view();

      projective_t HT_commited;
      CHK_STICKY(msm::_MSM(d_coeff.get(), pk.g1.vanishing_poly_points.data(), n - 1, msm_config, &HT_commited));

      projective_t private_inputs_commited;
      msm_config.are_scalars_on_device = false;
      CHK_STICKY(msm::_MSM(
        witness.data() + l + 1, pk.g1.private_witness_points.data(), m - l, msm_config, &private_inputs_commited));

      proof.C = projective_t::to_affine(
        private_inputs_commited + HT_commited + projective_t::from_affine(proof.A) * s + B1 * r -
        r * s * projective_t::from_affine(pk.g1.delta));
    }

    return proof;
  }

  bool verify(const G16proof& proof) const
  {
    throw std::runtime_error("pairing not implemented");
    return false;
  }

  // constructor
  const int nof_inputs;
  const int nof_outputs;
  const int nof_intermediates;
  const int witness_size;
  const int input_offset;
  const int intermediate_offset;
  const int nof_constraints;
  std::vector<Polynomial_t> L_QAP, R_QAP, O_QAP;
  ProvingKey pk;
  VerifyingKey vk;
};

TEST_F(PolynomialTest, QAP)
{
  // (1) construct R1CS and QAP for circuit with N inputs
  Groth16Example QAP(300 /*=N*/);

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
  //  (4) sanity check: verify AB=C at the evaluation points
  {
    auto default_device_context = device_context::get_default_device_context();
    const auto w = ntt::GetRootOfUnity<scalar_t>((int)ceil(log2(nof_constraints)), default_device_context);
    auto x = w;
    for (int i = 0; i < nof_constraints; ++i) {
      ASSERT_EQ(Lx(x) * Rx(x), Ox(x));
      x = x * w;
    }
  }

  // (5) compute h(x) as '(L(x)R(x)-O(x)) / t(x)'
  const int vanishing_poly_deg = ceil_to_power_of_two(nof_constraints);
  Polynomial_t h = (Lx * Rx - Ox).divide_by_vanishing_polynomial(vanishing_poly_deg);

  // (6) sanity check: vanishing-polynomial divides (LR-O) without remainder
  {
    auto [h_long_div, r] = (Lx * Rx - Ox).divide(vanishing_polynomial(vanishing_poly_deg));
    EXPECT_EQ(r.degree(), -1); // zero polynomial (expecting division without remainder)
    assert_equal(h, h_long_div);
  }
}

TEST_F(PolynomialTest, Groth16)
{
  // (1) construct R1CS and QAP for circuit with N inputs
  Groth16Example groth16_example(30 /*=N*/);

  // (2) compute witness: randomize inputs and compute other entries [1,out,...N inputs..., ... intermediate values...]
  auto witness = groth16_example.random_witness_inputs();
  groth16_example.compute_witness(witness);

  groth16_example.setup();
  auto proof = groth16_example.prove(witness);
  // groth16_example.verify(proof); // cannot implement without pairing
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}