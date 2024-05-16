
#include <iostream>
#include <memory>
#include <vector>
#include <list>

#include "polynomials/polynomials.h"
#include "polynomials/cuda_backend/polynomial_cuda_backend.cuh"
#include "ntt/ntt.cuh"
// #include "poseidon/tree/merkle.cuh"

using namespace polynomials;
// using namespace merkle;

// define the polynomial type
typedef Polynomial<scalar_t> Polynomial_t;

// Merkle tree arity
// #define A 2
// #define T (A + 1)

// RISC-V register type
typedef int64_t rv_t;

// Convert RISC-V registers to Finite Fields
void to_ff(rv_t* rv, scalar_t* s, size_t n) {
  for (int i = 0; i < n; ++i) {
    s[i] = scalar_t::from(rv[i]);
  }
}

void print_vector(scalar_t* v, size_t n, std::string header = "Print Vector") {
  std::cout << header << std::endl;
  for (int i = 0; i < n; ++i) {
    std::cout << i << ": " << v[i] << std::endl;
  }
}


Polynomial_t p_value(scalar_t value) {
  auto coeff = std::make_unique<scalar_t[]>(1);
  coeff[0] = value;
  auto p_value = Polynomial_t::from_coefficients(coeff.get() , 1);
  return p_value;
}

// TBD: remove all evaluations and use polynomials
void compute_value_constraint(
  scalar_t* data,
  scalar_t value,
  scalar_t* control,
  scalar_t* constraint, 
  size_t n
  ) 
{
  // data[?] - value returns 0 when the value loaded into the program matches the asserted input/output
  // control returns 0 when we aren't enforcing this rule.
  // The product of these two terms returns 0 at each row of the original trace.
  for (int i = 0; i < n; ++i) {
    constraint[i] = (data[i] - value) * control[i];
  }
  return;
}

void compute_fib_constraint(
  scalar_t *d1, 
  scalar_t *d2,
  scalar_t *d3,
  scalar_t *c1,
  scalar_t *c2,
  scalar_t *c3,
  scalar_t *constraint,
  int n
  ) 
{
  // d3 - d2 - d1 returns 0 when the fibonacci addition relation holds.
  // c1 + c2 + c3 returns 0 when we aren't enforcing this rule.
  // The product of these two terms returns 0 at each row of the original trace.
  for (int i = 0; i < n; ++i) {
    constraint[i] = (d3[i] - d2[i] - d1[i]) * (c1[i] + c2[i] + c3[i]);
  }
  return;
}

void compute_recursion_constraint(
  scalar_t * da, 
  scalar_t * db, 
  scalar_t * c, 
  scalar_t *constraint,
  int n, 
  int dn) 
{
  // constrain data transfers between registers (two columns of the trace)
  // dn = 1 for trace_columns, dn = 4 for trace_zkcommitment and trace_reedsolomonexpansion
  // # We check that the ith term in trace[0] matches the "previous" term in trace[1].
  // # We use "previous" to mean "one computational step before."
  // # i.e., for trace_columns, we're checking trace[0][i] against trace[0][i-1], while
  // # for trace_zkcommitment and trace_reedsolomonexpansion, we're checking trace[0][i] vs. trace[1][i-4]
  // # We express this length-dependent step-size as len(trace[0])//8
  // # We also use "previous" in a cyclical sense, wrapping around to the end of the trace.
  // # Putting this together, the relation we are checking is:
  // #   trace[0][i] - trace[1][i - len(trace[0])//8 % len(trace[0])
  // # trace[4] is the associated indicator column; trace[4][i] = 0 when this rule isn't enforced.
  for (int i = 0; i < n; ++i) {
    constraint[i] = (da[i] - db[i - dn % n]) * c[i];
  }
}

// mix polynomial evaluations
void mix(scalar_t* in[], scalar_t * out, size_t nmix, size_t n, scalar_t mix_parameter) {
  scalar_t * a;
  scalar_t factor = scalar_t::one();
  for (size_t i = 0; i < n; ++i) {
    out[i] = scalar_t::zero();
  }
  for (size_t i = 0; i < nmix; ++i) {
    a = in[i];
    for (size_t j = 0; j < n; ++j) {
      out[j] = out[j] + a[j] * factor;
    }
    factor = factor * mix_parameter;
  }
}

// mix polynomials (c.f. mix polynomial evaluations)
Polynomial_t p_mix(Polynomial_t* in[], size_t nmix, scalar_t mix_parameter) {
  scalar_t factor = mix_parameter;
  Polynomial_t out = in[0]->clone();
  for (int i = 1; i < nmix; ++i) {
    out += factor * (*in[i]);
    factor = factor * mix_parameter;
  }
  return out;
}

void solve_linear(scalar_t xa, scalar_t ya, scalar_t xb, scalar_t yb, scalar_t * coeffs) {
  coeffs[1] = (ya - yb) * scalar_t::inverse(xa - xb);
  coeffs[0] = ya - coeffs[1] * xa;
}

std::unique_ptr<scalar_t[]> initialize_domain(int logn, scalar_t shift = scalar_t::one()) {
    auto omega = scalar_t::omega(logn);  // Compute the nth root of unity
    auto n = (1 << logn);               // Calculate the domain size as 2^logn
    auto domain = std::make_unique<scalar_t[]>(n);  // Allocate memory for the domain

    // scalar_t x = scalar_t::one();  // Start with one
    scalar_t x = shift;
    for (int i = 0; i < n; ++i) {
        domain[i] = x;             // Assign the current value of xx to the domain
        x = x * omega;            // Update xx by multiplying it with omega
    }
    return domain;  // Return the populated domain
}


std::unique_ptr<scalar_t[]> InterpolateRootsOfUnity(Polynomial_t * p, int n, scalar_t shift = scalar_t::one()) {
  const int deg = p->degree();
  auto input = std::make_unique<scalar_t[]>(n);
  // TBD: check if scalar_t constructor initializes to zero
  for (int i = 0; i < n; ++i) {
    input[i] = scalar_t::zero();
  }
  p->copy_coeffs(input.get(), 0/*start*/, deg);
  auto ntt_config = ntt::default_ntt_config<scalar_t>();
  ntt_config.coset_gen = shift;
  auto evals_h = std::make_unique<scalar_t[]>(n);
  auto err = ntt::ntt(input.get(), n, ntt::NTTDir::kForward, ntt_config, evals_h.get());
  // delete[] input;
  return evals_h;
}


int main(int argc, char** argv)
{
  std::cout << "This is an ICICLE C++ implementation of the STARK by Hand Explainer." << std::endl;
  std::cout << "https://dev.risczero.com/proof-system/stark-by-hand" << std::endl;

  const int logn=3;
  const int n = 1 << logn;
  
  std::cout << "Initializing NTT" << std::endl;
  static const int MAX_NTT_LOG_SIZE = 24;
  auto ntt_config = ntt::default_ntt_config<scalar_t>();
  const scalar_t basic_root = scalar_t::omega(MAX_NTT_LOG_SIZE);
  ntt::init_domain(basic_root, ntt_config.ctx);
  std::cout << "Initializing Polynomials" << std::endl;
  // Virtual factory design pattern: initializing polynomimals factory for CUDA backend
  Polynomial_t::initialize(std::make_unique<CUDAPolynomialFactory<>>());

  std::cout << std::endl << "Lesson 1: The Execution Trace" << std::endl; 
  // Trace: Data Columns
  rv_t rv_d1_trace[] = {24, 30, 54,  84, 78, 15, 29, 50};
  rv_t rv_d2_trace[] = {30, 54, 84,  138, 2, 77, 21, 36};
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
  std::cout << "We defined functions to create rule-checking polynomials. Their names start with p_rule_check." << std::endl;


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

  // Interpolate trace polynomials on 4x expanded rou domain (Reed Solomon) 
  // TBD: I don't need to keep their evaluations since I have the polynomials

  auto d1_trace_rs = InterpolateRootsOfUnity(&p_d1, 4*n);
  auto d2_trace_rs = InterpolateRootsOfUnity(&p_d2, 4*n);
  auto d3_trace_rs = InterpolateRootsOfUnity(&p_d3, 4*n);
  auto c1_trace_rs = InterpolateRootsOfUnity(&p_c1, 4*n);
  auto c2_trace_rs = InterpolateRootsOfUnity(&p_c2, 4*n);
  auto c3_trace_rs = InterpolateRootsOfUnity(&p_c3, 4*n);

  std::cout << "Note that every 4th entry matches the original trace data." << std::endl;
  std::cout << "This is a degree 4 Reed Solomon expansion of the original trace." << std::endl;

  std::cout << "Lesson 5: ZK Commitments of the Trace Data" << std::endl;
  std::cout << "To maintain a zk protocol, the trace polynomials are evaluated over a zk commitment domain" << std::endl;
  std::cout << "zk commitment domain is a coset of Reed Solomon domain shifted by a basic root of unity" << std::endl;

  scalar_t xzk = basic_root;
  auto d1_zkcommitment  = InterpolateRootsOfUnity(&p_d1, 4*n, xzk);
  auto d2_zkcommitment  = InterpolateRootsOfUnity(&p_d2, 4*n, xzk);
  auto d3_zkcommitment  = InterpolateRootsOfUnity(&p_d3, 4*n, xzk);
  auto c1_zkcommitment  = InterpolateRootsOfUnity(&p_c1, 4*n, xzk);
  auto c2_zkcommitment  = InterpolateRootsOfUnity(&p_c2, 4*n, xzk);
  auto c3_zkcommitment  = InterpolateRootsOfUnity(&p_c3, 4*n, xzk);
  
  std::cout << "These zk-commitments do not share any evaluation points with the original trace data." << std::endl;
  for (int i = 0; i < 4*n; ++i) {
    std::cout << i << ": " << d1_zkcommitment[i] << std::endl;
  }
    
  std::cout << "Build Merkle Tree (outside the scope of this example)" << std::endl;

  std::cout << "Lesson 6: Constraint Polynomials" << std::endl;
  std::cout << "The constraints are used to check the correctness of the trace. In this example, we check 6 rules to establish the validity of the trace." << std::endl;


  auto p_fib_constraint =  (p_d3 - p_d2 - p_d1) * (p_c1 + p_c2 + p_c3);

  // if I comment this line, I get polynomial_cuda_backend.cu:183 error: clone() from non implemented state
  auto fib_constraint_zkcommitment = InterpolateRootsOfUnity(&p_fib_constraint, 4*n, xzk);  

    
  auto p_init1_constraint = (p_d1 - p_value(scalar_t::from(24))) * p_c1;

  // sanity checks printing
  auto init1_constraint_rs = InterpolateRootsOfUnity(&p_init1_constraint, 4*n);
  print_vector(init1_constraint_rs.get(), 4*n, "Reed-Solomon expansion Init 1 constraint gives 0s in every 4th row");
  auto init1_constraint_zkcommitment = InterpolateRootsOfUnity(&p_init1_constraint, 4*n, xzk);
  print_vector(init1_constraint_zkcommitment.get(), 4*n, "ZK Commitment Init 1 constraint gives no 0s");

  auto p_init2_constraint = (p_d2 - p_value(scalar_t::from(30))) * p_c1;
  auto p_termination_constraint = (p_d3 - p_value(scalar_t::from(222))) * p_c3;

  // TBD: I had issues with recursion constraints. Need to debug. But not now.
  scalar_t recursion_constraint1[n];
  compute_recursion_constraint(d1_trace.get(), d2_trace.get(), c2_trace.get(), recursion_constraint1, n, 1);
  std::cout << "Original Recursion constraint gives 0s" << std::endl;
  print_vector(recursion_constraint1, n);

  scalar_t recursion_constraint1_rs[4*n];
  compute_recursion_constraint(d1_trace_rs.get(), d2_trace_rs.get(), c2_trace_rs.get(), recursion_constraint1_rs, 4*n, 4);
  print_vector(recursion_constraint1_rs, 4*n, "Reed-Solomon expansion Recursion constraint gives 0s in every 4th row");

  scalar_t recursion_constraint1_zkcommitment[4*n];
  compute_recursion_constraint(d1_zkcommitment.get(), d2_zkcommitment.get(), c2_zkcommitment.get(), recursion_constraint1_zkcommitment, 4*n, 4);
  std::cout << "ZK Commitment Recursion constraint gives no 0s" << std::endl;
  print_vector(recursion_constraint1_zkcommitment, 4*n);

  scalar_t recursion_constraint2[n];
  compute_recursion_constraint(d2_trace.get(), d3_trace.get(), c2_trace.get(), recursion_constraint2, n, 1);
  std::cout << "Original Recursion constraint gives 0s" << std::endl;
  print_vector(recursion_constraint2, n);

  scalar_t recursion_constraint2_rs[4*n];
  compute_recursion_constraint(d2_trace_rs.get(), d3_trace_rs.get(), c2_trace_rs.get(), recursion_constraint2_rs, 4*n, 4);
  std::cout << "Reed-Solomon expansion Recursion constraint gives 0s in every 4th row" << std::endl;
  print_vector(recursion_constraint2_rs, 4*n);

  scalar_t recursion_constraint2_zkcommitment[4*n];
  compute_recursion_constraint(d2_zkcommitment.get(), d3_zkcommitment.get(), c2_zkcommitment.get(), recursion_constraint2_zkcommitment, 4*n, 4);
  std::cout << "ZK Commitment Recursion constraint gives no 0s" << std::endl;
  print_vector(recursion_constraint2_zkcommitment, 4*n);

  std::cout << std::endl << "Lesson 7: Mixing Constraint Polynomials" << std::endl;
  // TBD: what's wrong with recursion constraints?
  // scalar_t* all_constraints[] = {fib_constraint, init1_constraint, init2_constraint, termination_constraint, recursion_constraint1, recursion_constraint2};

  Polynomial_t * p_all_constraints[] = {&p_fib_constraint, &p_init1_constraint, &p_init2_constraint, &p_termination_constraint};
  const size_t nmix = sizeof(p_all_constraints) / sizeof(p_all_constraints[0]);

  auto p_mixed_constraints = p_mix(p_all_constraints, nmix, scalar_t::from(5));

  auto mixed_constraint_rs = InterpolateRootsOfUnity(&p_mixed_constraints, 4*n);
  print_vector(mixed_constraint_rs.get(), 4*n, "Mixed constraint gives 0s in every 4th row");

  auto mixed_constraint_zkcommitment = InterpolateRootsOfUnity(&p_mixed_constraints, 4*n, xzk);
  print_vector(mixed_constraint_zkcommitment.get(), 4*n, "Mixed ZK constraint gives no 0s");

  // Issue: recursive constraints have large degree. Is this an issue?
  for( int i = 0; i < nmix; ++i) {
    std::cout << i << ": " << p_all_constraints[i]->degree() << std::endl;
  }

  std::cout << "Lesson 8: The Core of the RISC Zero STARK" << std::endl;
  std::cout << "Degree of the mixed constraints polynomial: " << p_mixed_constraints.degree() << std::endl;  
  auto p_validity = p_mixed_constraints.divide_by_vanishing_polynomial(n);
  std::cout << "Degree of the validity polynomial: " << p_validity.degree() << std::endl;
  std::cout << "The Verifier should provide the Merke commitment for the above" << std::endl;

  std::cout << "Lesson 9: The DEEP Technique" << std::endl;
  std::cout << "The DEEP technique improves the security of a single query by sampling outside of the commitment domain."  << std::endl;
  
  // In the original STARK protocol, the Verifier tests validity polynomial at a number of test points; 
  // the soundness of the protocol depends on the number of tests. 
  // The DEEP-ALI technique allows us to achieve a high degree of soundness with a single test. 
  // The details of DEEP are described in the following lesson.

  auto DEEP_point = scalar_t::from(93);
  std::cout << "The prover convinces the verifier that V=C/Z at the DEEP_test_point, " << DEEP_point << std::endl;
  
  const scalar_t coeffs1[2] = {scalar_t::zero()-DEEP_point, scalar_t::one()};
  auto denom_DEEP1 = Polynomial_t::from_coefficients(coeffs1, 2);
  auto [p_d1_DEEP, r] = (p_d1 - p_value(DEEP_point)).divide(denom_DEEP1);
  
  std::cout << "The DEEP d1 degree is: " << p_d1_DEEP.degree() << std::endl;

  // d2, d3 use recursion constraints and need the point corresponding to the previous state (clock cycle)
  auto omega = scalar_t::omega(logn);
  auto DEEP_prev_point = DEEP_point*scalar_t::inverse(omega); 
  auto coeffs2 = std::make_unique<scalar_t[]>(2);
  coeffs2[0] = scalar_t::zero() - DEEP_prev_point;
  coeffs2[1] = scalar_t::one();
  auto denom_DEEP2 = Polynomial_t::from_coefficients(coeffs2.get(), 2);

  auto coeffs_d2bar = std::make_unique<scalar_t[]>(2);
  solve_linear(DEEP_point, p_d2(DEEP_point), DEEP_prev_point, p_d2(DEEP_prev_point), coeffs_d2bar.get());
  auto d2bar = Polynomial_t::from_coefficients(coeffs_d2bar.get(), 2);
  auto [p_d2_DEEP, r2] = (p_d2 - d2bar).divide(denom_DEEP1*denom_DEEP2);
  std::cout << "The DEEP d2 degree is: " << p_d2_DEEP.degree() << std::endl;

  auto coeffs_d3bar = std::make_unique<scalar_t[]>(2);
  solve_linear(DEEP_point, p_d3(DEEP_point), DEEP_prev_point, p_d3(DEEP_prev_point), coeffs_d3bar.get());
  auto d3bar = Polynomial_t::from_coefficients(coeffs_d3bar.get(), 2);
  auto [p_d3_DEEP, r3] = (p_d3 - d3bar).divide(denom_DEEP1*denom_DEEP2);
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
  std::cout << "The prover provides information to convince the verifier that the DEEP polynomials are low-degree." << std::endl;

  int nof_rounds = 3;
  Polynomial_t feven[nof_rounds], fodd[nof_rounds], fri[nof_rounds+1];
  scalar_t rfri[nof_rounds];

  fri[0] = fri_input.clone();
  for (int i = 0; i < nof_rounds; ++i) {
    feven[i] = fri[i].even();
    fodd[i] = fri[i].odd();
    rfri[i] = scalar_t::rand_host();  
    fri[i+1] = feven[i] + rfri[i]*fodd[i];
    std::cout << "The degree of the Round " << i << " polynomial is: " << fri[i+1].degree() << std::endl;
  }

  std::cout << "Lesson 12: FRI Protocol (Query Phase)" << std::endl;
  // We use Polynomial API to evaluate the FRI polynomials
  // In practice, verifier will use Merkle commitments
  auto xp = scalar_t::rand_host();
  auto xm = scalar_t::zero() - xp;
  scalar_t lhs[nof_rounds], rhs[nof_rounds];
  for (int i = 0; i < nof_rounds; ++i) {
    rhs[i] = (rfri[i]+xp)*fri[i](xp)*scalar_t::inverse(scalar_t::from(2)*xp) + (rfri[i]+xm)*fri[i](xm)*scalar_t::inverse(scalar_t::from(2)*xm);
    lhs[i] = fri[i+1](xp*xp);
    std::cout << "Round " << i << std::endl << "rhs: " << rhs[i] << std::endl << "lhs: " << lhs[i] << std::endl;
  }

  return 0;
}