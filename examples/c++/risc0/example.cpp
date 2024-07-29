
#include <iostream>
#include <memory>
#include <vector>
#include <list>

#include "examples_utils.h"
#include "icicle/polynomials/polynomials.h"
#include "icicle/api/babybear.h"

using namespace babybear;

// define the polynomial type
typedef Polynomial<scalar_t> Polynomial_t;

// RISC-V register type
typedef int64_t rv_t;

// Convert RISC-V registers to Finite Fields
void to_ff(rv_t* rv, scalar_t* s, size_t n)
{
  for (int i = 0; i < n; ++i) {
    s[i] = scalar_t::from(rv[i]);
  }
}

void p_print(Polynomial_t* p, int logn, scalar_t shift, std::string header = "Print Vector")
{
  std::cout << header << std::endl;
  auto n = 1 << logn;
  auto omega = scalar_t::omega(logn);
  auto x = shift;
  for (int i = 0; i < n; ++i) {
    std::cout << i << ": " << (*p)(x) << std::endl;
    x = x * omega;
  }
}

// value to polynomial
Polynomial_t p_value(scalar_t value)
{
  auto p_value = Polynomial_t::from_coefficients(&value, 1);
  return p_value;
}

Polynomial_t p_rotate(Polynomial_t* p, int logn)
{
  // rotate polynomial coefficients right by one position
  auto n = 1 << logn;
  auto evaluations_rou_domain = std::make_unique<scalar_t[]>(n);
  p->evaluate_on_rou_domain(logn, evaluations_rou_domain.get());
  scalar_t tmp = evaluations_rou_domain[n - 1];
  for (int i = n - 1; i > 0; --i) {
    evaluations_rou_domain[i] = evaluations_rou_domain[i - 1];
  }
  evaluations_rou_domain[0] = tmp;
  return Polynomial_t::from_rou_evaluations(evaluations_rou_domain.get(), n);
}

// mix polynomials (c.f. mix polynomial evaluations)
Polynomial_t p_mix(Polynomial_t* in[], size_t nmix, scalar_t mix_parameter)
{
  scalar_t factor = mix_parameter;
  Polynomial_t out = in[0]->clone();
  for (int i = 1; i < nmix; ++i) {
    out += factor * (*in[i]);
    factor = factor * mix_parameter;
  }
  return out;
}

void solve_linear(scalar_t xa, scalar_t ya, scalar_t xb, scalar_t yb, scalar_t* coeffs)
{
  coeffs[1] = (ya - yb) * scalar_t::inverse(xa - xb);
  coeffs[0] = ya - coeffs[1] * xa;
}

std::unique_ptr<scalar_t[]> InterpolateOnLargerDomain(Polynomial_t* p, int n, scalar_t shift = scalar_t::one())
{
  const int deg = p->degree();
  auto input = std::make_unique<scalar_t[]>(n);
  // TBD: check if scalar_t constructor initializes to zero
  for (int i = 0; i < n; ++i) {
    input[i] = scalar_t::zero();
  }
  p->copy_coeffs(input.get(), 0 /*start*/, deg);
  auto ntt_config = default_ntt_config<scalar_t>();
  ntt_config.coset_gen = shift;
  auto evals_h = std::make_unique<scalar_t[]>(n);
  ICICLE_CHECK(ntt(input.get(), n, NTTDir::kForward, ntt_config, evals_h.get()));
  return evals_h;
}

int main(int argc, char** argv)
{
  try_load_and_set_backend_device(argc, argv);

  START_TIMER(risc0_example);

  std::cout << "This is an ICICLE C++ implementation of the STARK by Hand Explainer." << std::endl;
  std::cout << "https://dev.risczero.com/proof-system/stark-by-hand" << std::endl;

  const int logn = 3;
  const int n = 1 << logn;

  std::cout << "Initializing NTT" << std::endl;
  static const int MAX_NTT_LOG_SIZE = 24;
  auto ntt_config = default_ntt_config<scalar_t>();
  const scalar_t basic_root = scalar_t::omega(MAX_NTT_LOG_SIZE);
  ntt_init_domain(basic_root, default_ntt_init_domain_config());

  std::cout << std::endl << "Lesson 1: The Execution Trace" << std::endl;
  // Trace: Data Columns
  rv_t rv_d1_trace[] = {24, 30, 54, 84, 78, 15, 29, 50};
  rv_t rv_d2_trace[] = {30, 54, 84, 138, 2, 77, 21, 36};
  rv_t rv_d3_trace[] = {54, 84, 138, 222, 71, 17, 92, 33};
  auto d1_trace = std::make_unique<scalar_t[]>(n);
  auto d2_trace = std::make_unique<scalar_t[]>(n);
  auto d3_trace = std::make_unique<scalar_t[]>(n);
  to_ff(rv_d1_trace, d1_trace.get(), n);
  to_ff(rv_d2_trace, d2_trace.get(), n);
  to_ff(rv_d3_trace, d3_trace.get(), n);
  // Trace: Control Columns
  // Init steps are flagged in c1_trace
  // Computation steps are flagged in c2_trace
  // Termination step is flagged in c3_trace
  // 0s at the end of each control column correspond to the padding of the trace
  rv_t rv_c1_trace[] = {1, 0, 0, 0, 0, 0, 0, 0};
  rv_t rv_c2_trace[] = {0, 1, 1, 1, 0, 0, 0, 0};
  rv_t rv_c3_trace[] = {0, 0, 0, 1, 0, 0, 0, 0};
  auto c1_trace = std::make_unique<scalar_t[]>(n);
  auto c2_trace = std::make_unique<scalar_t[]>(n);
  auto c3_trace = std::make_unique<scalar_t[]>(n);
  to_ff(rv_c1_trace, c1_trace.get(), n);
  to_ff(rv_c2_trace, c2_trace.get(), n);
  to_ff(rv_c3_trace, c3_trace.get(), n);

  std::cout << "Lesson 2: Rule checks to validate a computation" << std::endl;
  std::cout << "We use rule-checking polynomials." << std::endl;

  std::cout << "Lesson 3: Padding the Trace" << std::endl;
  // The trace is padded to a power of 2 size to allow for efficient NTT operations.
  // we already did this in the initialization of the trace data
  // We will construct a zero-knowledge proof that:
  // this trace represents a program that satisfies these 6 rules:
  //  1) Fibonacci words here
  //  2) d1_trace[0] == 24  (init 1 constraint)
  //  3) d2_trace[0] == 30  (init 2 constraint)
  //  4) d3_trace[3] == 28  (termination constraint)
  //  5) if c2_trace[i] == 1, then d2_trace[i] == d1_trace[i+1]
  //  6) if c2_trace[i] == 1, then d3_trace[i] == d2_trace[i+1}

  std::cout << "Lesson 4: Constructing Trace Polynomials" << std::endl;
  auto p_d1 = Polynomial_t::from_rou_evaluations(d1_trace.get(), n);
  auto p_d2 = Polynomial_t::from_rou_evaluations(d2_trace.get(), n);
  auto p_d3 = Polynomial_t::from_rou_evaluations(d3_trace.get(), n);
  auto p_c1 = Polynomial_t::from_rou_evaluations(c1_trace.get(), n);
  auto p_c2 = Polynomial_t::from_rou_evaluations(c2_trace.get(), n);
  auto p_c3 = Polynomial_t::from_rou_evaluations(c3_trace.get(), n);

  std::cout << "Lesson 5: ZK Commitments of the Trace Data" << std::endl;
  std::cout << "To maintain a zk protocol, the trace polynomials are evaluated over a zk commitment domain"
            << std::endl;
  std::cout << "zk commitment domain is a coset of Reed Solomon domain shifted by a basic root of unity" << std::endl;
  scalar_t xzk = basic_root;
  p_print(&p_d1, logn, xzk, "ZK commitment for d1 polynomial");
  std::cout << "Build Merkle Tree for ZK commitments (outside the scope of this example)" << std::endl;

  std::cout << "Lesson 6: Constraint Polynomials" << std::endl;
  std::cout << "The constraints are used to check the correctness of the trace. In this example, we check 6 rules to "
               "establish the validity of the trace."
            << std::endl;
  auto p_fib_constraint = (p_d3 - p_d2 - p_d1) * (p_c1 + p_c2 + p_c3);
  auto fib_constraint_zkcommitment = InterpolateOnLargerDomain(&p_fib_constraint, 4 * n, xzk);

  auto p_init1_constraint = (p_d1 - p_value(scalar_t::from(24))) * p_c1;
  // sanity checks printing
  p_print(
    &p_init1_constraint, logn + 2, scalar_t::one(), "Reed-Solomon constraint polynomial gives 0s in every 4th row");
  p_print(&p_init1_constraint, logn + 2, xzk, "ZK Commitment constraint polynomial gives no 0s");
  auto p_init2_constraint = (p_d2 - p_value(scalar_t::from(30))) * p_c1;
  auto p_termination_constraint = (p_d3 - p_value(scalar_t::from(222))) * p_c3;
  auto p_recursion_constraint1 = (p_d1 - p_rotate(&p_d2, logn)) * p_c2;
  auto p_recursion_constraint2 = (p_d2 - p_rotate(&p_d3, logn)) * p_c2;

  std::cout << std::endl << "Lesson 7: Mixing Constraint Polynomials" << std::endl;
  Polynomial_t* p_all_constraints[] = {&p_fib_constraint,         &p_init1_constraint,      &p_init2_constraint,
                                       &p_termination_constraint, &p_recursion_constraint1, &p_recursion_constraint2};
  const size_t nmix = sizeof(p_all_constraints) / sizeof(p_all_constraints[0]);
  auto p_mixed_constraints = p_mix(p_all_constraints, nmix, scalar_t::from(5));
  std::cout << "All constraint polynomials are low-degree:" << std::endl;
  for (int i = 0; i < nmix; ++i) {
    std::cout << i << ": " << p_all_constraints[i]->degree() << std::endl;
  }

  std::cout << "Lesson 8: The Core of the RISC Zero STARK" << std::endl;
  std::cout << "Degree of the mixed constraints polynomial: " << p_mixed_constraints.degree() << std::endl;
  auto p_validity = p_mixed_constraints.divide_by_vanishing_polynomial(n);
  std::cout << "Degree of the validity polynomial: " << p_validity.degree() << std::endl;
  std::cout << "The Verifier should provide the Merke commitment for the above" << std::endl;

  std::cout << "Lesson 9: The DEEP Technique" << std::endl;
  std::cout
    << "The DEEP technique improves the security of a single query by sampling outside of the commitment domain."
    << std::endl;
  // In the original STARK protocol, the Verifier tests validity polynomial at a number of test points;
  // the soundness of the protocol depends on the number of tests.
  // The DEEP-ALI technique allows us to achieve a high degree of soundness with a single test.
  // The details of DEEP are described in the following lesson.

  auto DEEP_point = scalar_t::from(93);
  std::cout << "The prover convinces the verifier that V=C/Z at the DEEP_test_point, " << DEEP_point << std::endl;
  const scalar_t coeffs1[2] = {scalar_t::zero() - DEEP_point, scalar_t::one()};
  auto denom_DEEP1 = Polynomial_t::from_coefficients(coeffs1, 2);
  auto [p_d1_DEEP, r] = (p_d1 - p_value(DEEP_point)).divide(denom_DEEP1);
  std::cout << "The DEEP d1 degree is: " << p_d1_DEEP.degree() << std::endl;
  // d2, d3 use recursion constraints and need the point corresponding to the previous state (clock cycle)
  auto omega = scalar_t::omega(logn);
  auto DEEP_prev_point = DEEP_point * scalar_t::inverse(omega);
  auto coeffs2 = std::make_unique<scalar_t[]>(2);
  coeffs2[0] = scalar_t::zero() - DEEP_prev_point;
  coeffs2[1] = scalar_t::one();
  auto denom_DEEP2 = Polynomial_t::from_coefficients(coeffs2.get(), 2);

  auto coeffs_d2bar = std::make_unique<scalar_t[]>(2);
  solve_linear(DEEP_point, p_d2(DEEP_point), DEEP_prev_point, p_d2(DEEP_prev_point), coeffs_d2bar.get());
  auto d2bar = Polynomial_t::from_coefficients(coeffs_d2bar.get(), 2);
  auto [p_d2_DEEP, r2] = (p_d2 - d2bar).divide(denom_DEEP1 * denom_DEEP2);
  std::cout << "The DEEP d2 degree is: " << p_d2_DEEP.degree() << std::endl;

  auto coeffs_d3bar = std::make_unique<scalar_t[]>(2);
  solve_linear(DEEP_point, p_d3(DEEP_point), DEEP_prev_point, p_d3(DEEP_prev_point), coeffs_d3bar.get());
  auto d3bar = Polynomial_t::from_coefficients(coeffs_d3bar.get(), 2);
  auto [p_d3_DEEP, r3] = (p_d3 - d3bar).divide(denom_DEEP1 * denom_DEEP2);
  std::cout << "The DEEP d3 degree is: " << p_d3_DEEP.degree() << std::endl;

  // DEEP c{1,2,3} polynomials
  const scalar_t coeffs_c1bar[1] = {p_c1(DEEP_point)};
  auto c1bar = Polynomial_t::from_coefficients(coeffs_c1bar, 1);
  auto [p_c1_DEEP, r_c1] = (p_c1 - c1bar).divide(denom_DEEP1);
  std::cout << "The DEEP c1 degree is: " << p_c1_DEEP.degree() << std::endl;
  const scalar_t coeffs_c2bar[1] = {p_c2(DEEP_point)};
  auto c2bar = Polynomial_t::from_coefficients(coeffs_c2bar, 1);
  auto [p_c2_DEEP, r_c2] = (p_c2 - c2bar).divide(denom_DEEP1);
  std::cout << "The DEEP c2 degree is: " << p_c2_DEEP.degree() << std::endl;
  const scalar_t coeffs_c3bar[1] = {p_c3(DEEP_point)};
  auto c3bar = Polynomial_t::from_coefficients(coeffs_c3bar, 1);
  auto [p_c3_DEEP, r_c3] = (p_c3 - c3bar).divide(denom_DEEP1);
  std::cout << "The DEEP c3 degree is: " << p_c3_DEEP.degree() << std::endl;
  // DEEP validity polynomial
  const scalar_t coeffs_vbar[1] = {p_validity(DEEP_point)};
  auto vbar = Polynomial_t::from_coefficients(coeffs_vbar, 1);
  auto [v_DEEP, r_v] = (p_validity - vbar).divide(denom_DEEP1);
  std::cout << "The DEEP validity polynomial degree is: " << v_DEEP.degree() << std::endl;
  std::cout << "The Prover sends DEEP polynomials to the Verifier" << std::endl;

  std::cout << "Lesson 10: Mixing (Batching) for FRI" << std::endl;
  std::cout << "The initial FRI polynomial is the mix of the 7 DEEP polynomials." << std::endl;
  Polynomial_t* all_DEEP[] = {&p_d1_DEEP, &p_d2_DEEP, &p_d3_DEEP, &p_c1_DEEP, &p_c2_DEEP, &p_c3_DEEP, &v_DEEP};
  Polynomial_t fri_input = p_mix(all_DEEP, 7, scalar_t::from(99));
  std::cout << "The degree of the mixed DEEP polynomial is: " << fri_input.degree() << std::endl;

  std::cout << "Lesson 11: FRI Protocol (Commit Phase)" << std::endl;
  std::cout << "The prover provides information to convince the verifier that the DEEP polynomials are low-degree."
            << std::endl;
  int nof_rounds = 3;
  Polynomial_t feven[nof_rounds], fodd[nof_rounds], fri[nof_rounds + 1];
  scalar_t rfri[nof_rounds];
  fri[0] = fri_input.clone();
  for (int i = 0; i < nof_rounds; ++i) {
    feven[i] = fri[i].even();
    fodd[i] = fri[i].odd();
    rfri[i] = scalar_t::rand_host();
    fri[i + 1] = feven[i] + rfri[i] * fodd[i];
    std::cout << "The degree of the Round " << i << " polynomial is: " << fri[i + 1].degree() << std::endl;
  }

  std::cout << "Lesson 12: FRI Protocol (Query Phase)" << std::endl;
  // We use Polynomial API to evaluate the FRI polynomials
  // In practice, verifier will use Merkle commitments
  auto xp = scalar_t::rand_host();
  auto xm = scalar_t::zero() - xp;
  scalar_t lhs[nof_rounds], rhs[nof_rounds];
  for (int i = 0; i < nof_rounds; ++i) {
    rhs[i] = (rfri[i] + xp) * fri[i](xp) * scalar_t::inverse(scalar_t::from(2) * xp) +
             (rfri[i] + xm) * fri[i](xm) * scalar_t::inverse(scalar_t::from(2) * xm);
    lhs[i] = fri[i + 1](xp * xp);
    std::cout << "Round " << i << std::endl << "rhs: " << rhs[i] << std::endl << "lhs: " << lhs[i] << std::endl;
  }

  END_TIMER(risc0_example, "risc0 example");
  return 0;
}