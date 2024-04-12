
#include <iostream>
#include <memory>
#include <vector>
#include <list>

#define FIELD_ID BN254
// #include "curves/curve_config.cuh"
#include "polynomials/polynomials.h"
#include "polynomials/polynomials_c_api.h"
#include "polynomials/cuda_backend/polynomial_cuda_backend.cuh"


#include "ntt/ntt.cuh"
#include "gpu-utils/device_context.cuh"

// #include "appUtils/ntt/ntt.cuh"
// #include "appUtils/poseidon/poseidon.cu"
// #include "appUtils/tree/merkle.cu"
// using namespace curve_config;
using namespace polynomials;
// using namespace poseidon;
// using namespace merkle;

// define the polynomial type
typedef Polynomial<scalar_t> Polynomial_t;

// Merkle tree arity
#define A 2
#define T (A + 1)

// RISC-V register type
typedef int64_t rv_t;

// We use Finite Fields instead of RISC-V registers

void to_ff(rv_t* rv, scalar_t* s, size_t n) {
  for (int i = 0; i < n; ++i) {
    s[i] = scalar_t::from(rv[i]);
  }
}


void print_vector(scalar_t* v, size_t n) {
  for (int i = 0; i < n; ++i) {
    std::cout << i << ": " << v[i] << std::endl;
  }
}


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

int main(int argc, char** argv)
{

  std::cout << "This is a ICICLE C++ implementation of the STARK by Hand Explainer." << std::endl;
  std::cout << "https://dev.risczero.com/proof-system/stark-by-hand" << std::endl;

  const int logn=3;
  const int n = 1 << logn;

  std::cout << std::endl << "1. Initialize ICICLE" << std::endl;
  std::cout << "NTT" << std::endl;
  static const int MAX_NTT_LOG_SIZE = 24;
  auto ntt_config = ntt::DefaultNTTConfig<scalar_t>();
  const scalar_t basic_root = scalar_t::omega(MAX_NTT_LOG_SIZE);
  ntt::InitDomain(basic_root, ntt_config.ctx);
  std::cout << "Polynomials" << std::endl;
  // Virtual factory design pattern: initializing polynomimals factory for CUDA backend
  Polynomial_t::initialize(std::make_unique<CUDAPolynomialFactory<>>());
  // Initialize Poseidon
  // std::cout << "Poseidon" << std::endl;
  device_context::DeviceContext ctx = device_context::get_default_device_context();
  // PoseidonConstants<scalar_t> constants;
  // init_optimized_poseidon_constants<scalar_t>(A, ctx, &constants);


  std::cout << std::endl << "2. Generate execution trace data" << std::endl; 
  // Trace: Data Columns
  rv_t rv_d1_trace[] = {24, 30, 54,  84, 78, 15, 29, 50};
  rv_t rv_d2_trace[] = {30, 54, 84,  138, 2, 77, 21, 36};
  rv_t rv_d3_trace[] = {54, 84, 138, 222, 71, 17, 92, 33};

  scalar_t d1_trace[n], d2_trace[n], d3_trace[n];

  to_ff(rv_d1_trace, d1_trace, n);
  to_ff(rv_d2_trace, d2_trace, n);
  to_ff(rv_d3_trace, d3_trace, n);

  std::cout << "d1 trace" << std::endl;
  for (int i = 0; i < n; ++i) {
    std::cout << i << ": " << d1_trace[i] << std::endl;
  }



  // Trace: Control Columns
  // Init steps are flagged in c1_trace
  // Computation steps are flagged in c2_trace
  // Termination step is flagged in c3_trace
  // 0s at the end of each control column correspond to the padding of the trace

  rv_t rv_c1_trace[] = {1, 0, 0, 0, 0, 0, 0, 0};
  rv_t rv_c2_trace[] = {0, 1, 1, 1, 0, 0, 0, 0};
  rv_t rv_c3_trace[] = {0, 0, 0, 1, 0, 0, 0, 0};

  scalar_t c1_trace[n], c2_trace[n], c3_trace[n];
  to_ff(rv_c1_trace, c1_trace, n);
  to_ff(rv_c2_trace, c2_trace, n);
  to_ff(rv_c3_trace, c3_trace, n);

  std::cout << "c1 trace" << std::endl;
  for (int i = 0; i < n; ++i) {
    std::cout << i << ": " << c1_trace[i] << std::endl;
  } 

  // We will construct a zero-knowledge proof that:
  // this trace represents a program that satisfies these 6 rules:
  //  1) Fibonacci words here
  //  2) d1_trace[0] == 24  (init 1 constraint)
  //  3) d2_trace[0] == 30  (init 2 constraint)
  //  4) d3_trace[3] == 28  (termination constraint)
  //  5) if c2_trace[i] == 1, then d2_trace[i] == d1_trace[i+1]
  //  6) if c2_trace[i] == 1, then d3_trace[i] == d2_trace[i+1}

  std::cout << "Lesson 4: Constructing Trace Polynomials" << std::endl;
  std::cout << std::endl << "3. Reconstruct polynomial from trace data" << std::endl;
  // d1_coeffs = np.array(intt(d1_trace, prime=_field_size))
  // d2_coeffs = np.array(intt(d2_trace, prime=_field_size))
  // d3_coeffs = np.array(intt(d3_trace, prime=_field_size))
  // c1_coeffs = np.array(intt(c1_trace, prime=_field_size))
  // c2_coeffs = np.array(intt(c2_trace, prime=_field_size))
  // c3_coeffs = np.array(intt(c3_trace, prime=_field_size))

  auto f = Polynomial_t::from_rou_evaluations(d1_trace, n);
  auto d = f.degree();

  auto d1_poly = Polynomial_t::from_rou_evaluations(d1_trace, n);
  auto d2_poly = Polynomial_t::from_rou_evaluations(d2_trace, n);
  auto d3_poly = Polynomial_t::from_rou_evaluations(d3_trace, n);
  auto c1_poly = Polynomial_t::from_rou_evaluations(c1_trace, n);
  auto c2_poly = Polynomial_t::from_rou_evaluations(c2_trace, n);
  auto c3_poly = Polynomial_t::from_rou_evaluations(c3_trace, n);

  // Evaluating Trace Polynomials over rou powers would return the original trace data
  

  auto d1_degree = d1_poly.degree();
  std::cout << "Degree: " << d1_degree << std::endl;
  auto x = scalar_t::one();
  auto omega = scalar_t::omega(logn);
  for (int i = 0; i < n; ++i) {
    std::cout << "i: " << d1_poly(x) << " trace: " << d1_trace[i] << std::endl;
    x = x * omega;
  }

  std::cout << std::endl << "4. Generate Reed-Solomon traces" << std::endl;
  // Evaluating Trace Polynomials over the "expanded domain" gives a "trace block."

  scalar_t d1_trace_rs[4*n], d2_trace_rs[4*n], d3_trace_rs[4*n];
  scalar_t c1_trace_rs[4*n], c2_trace_rs[4*n], c3_trace_rs[4*n];

  auto omega_rs = scalar_t::omega(2+logn);
  auto x_rs = scalar_t::one();

  for (int i = 0; i < 4*n; ++i) {
    d1_trace_rs[i] = d1_poly(x_rs);
    d2_trace_rs[i] = d2_poly(x_rs);
    d3_trace_rs[i] = d3_poly(x_rs);
    c1_trace_rs[i] = c1_poly(x_rs);
    c2_trace_rs[i] = c2_poly(x_rs);
    c3_trace_rs[i] = c3_poly(x_rs);
    x_rs = x_rs * omega_rs;
  }

  for(int i = 0; i < 4*n; ++i) {
    std::cout << i << ": " << d1_trace_rs[i] << std::endl;
  }
  std::cout << "Note that every 4th entry matches the original trace data." << std::endl;
  std::cout << "This is a degree 4 Reed Solomon expansion of the original trace." << std::endl;

  std::cout << "Lesson 5: ZK Commitments of the Trace Data" << std::endl;
  std::cout << "To maintain a zero-knowledge protocol, the trace polynomials are evaluated over a zk commitment domain" << std::endl;
  std::cout << std::endl << "5. Reconstruct polynomial for the codeword" << std::endl;

  scalar_t d1_zkcommitment[4*n], d2_zkcommitment[4*n], d3_zkcommitment[4*n];
  scalar_t c1_zkcommitment[4*n], c2_zkcommitment[4*n], c3_zkcommitment[4*n];

  
  std::cout << std::endl << "6. Commit to the codeword polynomial" << std::endl;
  std::cout << "Evaluate with a shift " << std::endl;
  scalar_t xzk = basic_root;
  
  for (int i = 0; i < 4*n; ++i) {
    d1_zkcommitment[i] = d1_poly(xzk);
    d2_zkcommitment[i] = d2_poly(xzk);
    d3_zkcommitment[i] = d3_poly(xzk);
    c1_zkcommitment[i] = c1_poly(xzk);
    c2_zkcommitment[i] = c2_poly(xzk);
    c3_zkcommitment[i] = c3_poly(xzk);
    xzk = xzk * omega_rs;
  }

  std::cout << "These zk-commitment blocks do not share any evaluation points with the original trace data." << std::endl;
  for (int i = 0; i < 4*n; ++i) {
    std::cout << i << ": " << d1_zkcommitment[i] << std::endl;
  }
    
  std::cout << "Build Merkle Tree (TBD)" << std::endl;
  std::cout << "Lesson 6: Constraint Polynomials" << std::endl;
  std::cout << "The constraints are used to check the correctness of the trace. In this example, we check 6 rules to establish the validity of the trace." << std::endl;
  // Applying rule checks to trace blocks makes constraint blocks.
  // A constraint block has 0s in every 4th row -- these 0s indicate the passing of the various rulechecks.

  // Applying rule checks to zk-commitment trace blocks makes zk-commitment constraint blocks.
  // Similarly, applying rule checks to trace polynomials makes constraint polynomials.
  // In code, this happens in terms of trace blocks.


  scalar_t fib_constraint[n];
  compute_fib_constraint(d1_trace, d2_trace, d3_trace, c1_trace, c2_trace, c3_trace, fib_constraint, n);
  std::cout <<  "Applied to the original trace data, the constraint yields all 0s: " << std::endl;
  print_vector(fib_constraint,n);
  scalar_t fib_constraint_rs[4*n];
  compute_fib_constraint(d1_trace_rs, d2_trace_rs, d3_trace_rs, c1_trace_rs, c2_trace_rs, c3_trace_rs, fib_constraint_rs, 4*n);
  std::cout <<  "Applied to the Reed-Solomon expanded trace blocks, the constraint yields 0s in every 4th row: " << std::endl;
  print_vector(fib_constraint_rs,4*n);
  scalar_t fib_constraint_zkcommitment[4*n];
  compute_fib_constraint(d1_zkcommitment, d2_zkcommitment, d3_zkcommitment, c1_zkcommitment, c2_zkcommitment, c3_zkcommitment, fib_constraint_zkcommitment, 4*n);
  std::cout <<  "Applied to zk-commitment domain, no 0s" << std::endl;
  print_vector(fib_constraint_zkcommitment,4*n);
    
  // init1_constraint_columns = init1_constraint(trace_data)
  // init1_constraint_reedsolomonexpansion = init1_constraint(trace_reedsolomonexpansion)
  // init1_constraint_zkcommitment = init1_constraint(trace_zkcommitment)

  scalar_t init1_constraint[n];
  compute_value_constraint(d1_trace, scalar_t::from(24), c1_trace, init1_constraint, n);
  std::cout << "Original Init 1 constraint gives 0s" << std::endl;
  print_vector(init1_constraint, n);

  scalar_t init1_constraint_rs[4*n];
  compute_value_constraint(d1_trace_rs, scalar_t::from(24), c1_trace_rs, init1_constraint_rs, 4*n);
  std::cout << "Reed-Solomon expansion Init 1 constraint gives 0s in every 4th row" << std::endl;
  print_vector(init1_constraint_rs, 4*n);

  scalar_t init1_constraint_zkcommitment[4*n];
  compute_value_constraint(d1_zkcommitment, scalar_t::from(24), c1_zkcommitment, init1_constraint_zkcommitment, 4*n);
  std::cout << "ZK Commitment Init 1 constraint gives no 0s" << std::endl;
  print_vector(init1_constraint_zkcommitment, 4*n);

  scalar_t init2_constraint[n];
  compute_value_constraint(d2_trace, scalar_t::from(30), c1_trace, init2_constraint, n);
  std::cout << "Original Init 2 constraint gives 0s" << std::endl;
  print_vector(init2_constraint, n);

  scalar_t init2_constraint_rs[4*n];
  compute_value_constraint(d2_trace_rs, scalar_t::from(30), c1_trace_rs, init2_constraint_rs, 4*n);
  std::cout << "Reed-Solomon expansion Init 2 constraint gives 0s in every 4th row" << std::endl;
  print_vector(init2_constraint_rs, 4*n);

  scalar_t init2_constraint_zkcommitment[4*n];
  compute_value_constraint(d2_zkcommitment, scalar_t::from(30), c1_zkcommitment, init2_constraint_zkcommitment, 4*n);
  std::cout << "ZK Commitment Init 2 constraint gives no 0s" << std::endl;
  print_vector(init2_constraint_zkcommitment, 4*n);

  scalar_t termination_constraint[n];
  compute_value_constraint(d3_trace, scalar_t::from(222), c3_trace, termination_constraint, n);
  std::cout << "Original Termination constraint gives 0s" << std::endl;
  print_vector(termination_constraint, n);

  scalar_t termination_constraint_rs[4*n];
  compute_value_constraint(d3_trace_rs, scalar_t::from(222), c3_trace_rs, termination_constraint_rs, 4*n);
  std::cout << "Reed-Solomon expansion Termination constraint gives 0s in every 4th row" << std::endl;
  print_vector(termination_constraint_rs, 4*n);

  scalar_t termination_constraint_zkcommitment[4*n];
  compute_value_constraint(d3_zkcommitment, scalar_t::from(222), c3_zkcommitment, termination_constraint_zkcommitment, 4*n);
  std::cout << "ZK Commitment Termination constraint gives no 0s" << std::endl;
  print_vector(termination_constraint_zkcommitment, 4*n);

  scalar_t recursion_constraint1[n];
  compute_recursion_constraint(d1_trace, d2_trace, c2_trace, recursion_constraint1, n, 1);
  std::cout << "Original Recursion constraint gives 0s" << std::endl;
  print_vector(recursion_constraint1, n);

  scalar_t recursion_constraint1_rs[4*n];
  compute_recursion_constraint(d1_trace_rs, d2_trace_rs, c2_trace_rs, recursion_constraint1_rs, 4*n, 4);
  std::cout << "Reed-Solomon expansion Recursion constraint gives 0s in every 4th row" << std::endl;
  print_vector(recursion_constraint1_rs, 4*n);

  scalar_t recursion_constraint1_zkcommitment[4*n];
  compute_recursion_constraint(d1_zkcommitment, d2_zkcommitment, c2_zkcommitment, recursion_constraint1_zkcommitment, 4*n, 4);
  std::cout << "ZK Commitment Recursion constraint gives no 0s" << std::endl;
  print_vector(recursion_constraint1_zkcommitment, 4*n);

  scalar_t recursion_constraint2[n];
  compute_recursion_constraint(d2_trace, d3_trace, c2_trace, recursion_constraint2, n, 1);
  std::cout << "Original Recursion constraint gives 0s" << std::endl;
  print_vector(recursion_constraint2, n);

  scalar_t recursion_constraint2_rs[4*n];
  compute_recursion_constraint(d2_trace_rs, d3_trace_rs, c2_trace_rs, recursion_constraint2_rs, 4*n, 4);
  std::cout << "Reed-Solomon expansion Recursion constraint gives 0s in every 4th row" << std::endl;
  print_vector(recursion_constraint2_rs, 4*n);

  scalar_t recursion_constraint2_zkcommitment[4*n];
  compute_recursion_constraint(d2_zkcommitment, d3_zkcommitment, c2_zkcommitment, recursion_constraint2_zkcommitment, 4*n, 4);
  std::cout << "ZK Commitment Recursion constraint gives no 0s" << std::endl;
  print_vector(recursion_constraint2_zkcommitment, 4*n);

  std::cout << std::endl << "Lesson 7: Mixing Constraint Polynomials" << std::endl;

  // scalar_t* all_constraints[] = {fib_constraint, init1_constraint, init2_constraint, termination_constraint, recursion_constraint1, recursion_constraint2};
  scalar_t* all_constraints[] = {fib_constraint, init1_constraint, init2_constraint, termination_constraint};
  size_t nmix = sizeof(all_constraints) / sizeof(all_constraints[0]);
  scalar_t mixed_constraint[n];
  mix(all_constraints, mixed_constraint, nmix, n, scalar_t::from(5));
  std::cout << "Mixed constraint gives 0s" << std::endl;
  print_vector(mixed_constraint, n);

  

  // scalar_t* all_constraints_rs[] = {fib_constraint_rs, init1_constraint_rs, init2_constraint_rs, termination_constraint_rs, recursion_constraint1_rs, recursion_constraint2_rs};
  scalar_t* all_constraints_rs[] = {fib_constraint_rs, init1_constraint_rs, init2_constraint_rs, termination_constraint_rs};
  scalar_t mixed_constraint_rs[4*n];
  mix(all_constraints_rs, mixed_constraint_rs, nmix, 4*n, scalar_t::from(5));
  std::cout << "Mixed constraint gives 0s in every 4th row" << std::endl;
  print_vector(mixed_constraint_rs, 4*n);

  // scalar_t* all_constraints_zkcommitment[] = {fib_constraint_zkcommitment, init1_constraint_zkcommitment, init2_constraint_zkcommitment, termination_constraint_zkcommitment, recursion_constraint1_zkcommitment, recursion_constraint2_zkcommitment};
  scalar_t* all_constraints_zkcommitment[] = {fib_constraint_zkcommitment, init1_constraint_zkcommitment, init2_constraint_zkcommitment, termination_constraint_zkcommitment};
  scalar_t mixed_constraint_zkcommitment[4*n];
  mix(all_constraints_zkcommitment, mixed_constraint_zkcommitment, nmix, 4*n, scalar_t::from(5));
  std::cout << "Mixed constraint gives no 0s" << std::endl;
  print_vector(mixed_constraint_zkcommitment, 4*n);

  // Issue: recursive constraints have large degree. Is this an issue?
  for( int i = 0; i < nmix; ++i) {
    auto poly_rs = Polynomial_t::from_rou_evaluations(all_constraints_rs[i], 4*n);  
    std::cout << i << ": " << poly_rs.degree() << std::endl;
  }
  // auto poly_rs = Polynomial_t::from_rou_evaluations(fib_constraint_zkcommitment, 4*n);
  // std::cout << "degree of poly_rs = " << poly_rs.degree() << std::endl;


  std::cout << "Lesson 8: The Core of the RISC Zero STARK" << std::endl;

  std::cout << "Reed-Solomon domain" << std::endl;
  auto p_mixed_constraint_rs = Polynomial_t::from_rou_evaluations(mixed_constraint_rs, 4*n);
  std::cout << "Degree of the mixed constraint polynomial: " << p_mixed_constraint_rs.degree() << std::endl;
  auto p_validity_rs = p_mixed_constraint_rs.divide_by_vanishing_polynomial(n);
  std::cout << "Degree of the validity polynomial: " << p_validity_rs.degree() << std::endl;

  std::cout << "ZK Commitment domain" << std::endl;
  // I can't use from_rou_evaluations() since the domain is shifted (coset). I'll wait for corresponding API.
  // auto p_mixed_constraint_zkcommitment = Polynomial_t::from_rou_evaluations(mixed_constraint_zkcommitment, 4*n);
  // std::cout << "Degree of the mixed constraint polynomial: " << p_mixed_constraint_zkcommitment.degree() << std::endl;
  // auto p_validity_zkcommitment = p_mixed_constraint_zkcommitment.divide_by_vanishing_polynomial(n);
  // std::cout << "Degree of the validity polynomial: " << p_validity_zkcommitment.degree() << std::endl;

  std::cout << "Evaluations of Validity Polynomial on zk-commitment domain" << std::endl;
  xzk = basic_root;
  for (int i = 0; i < 4*n; ++i) {
    std::cout << i << ": " << p_validity_rs(xzk) << std::endl;
    xzk = xzk * omega_rs;
  }

  std::cout << "The Virifier should provide the Merke commitment for the above" << std::endl;

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
  auto d1_poly_tmp = d1_poly.clone();
  d1_poly_tmp.sub_monomial_inplace(d1_poly(DEEP_point)) ;
  auto [d1_poly_DEEP, r] = d1_poly_tmp.divide(denom_DEEP1);
  std::cout << "The DEEP d1 degree is: " << d1_poly_DEEP.degree() << std::endl;

  // d2, d3 use recursion constraints and need the point corresponding to the previous state (clock cycle)
  auto DEEP_prev_point = DEEP_point*scalar_t::inverse(omega);
  const scalar_t coeffs2[2] = {scalar_t::zero()-DEEP_prev_point, scalar_t::one()};
  auto denom_DEEP2 = Polynomial_t::from_coefficients(coeffs2, 2);
  
  scalar_t coeffs_d2bar[2];
  solve_linear(DEEP_point, d2_poly(DEEP_point), DEEP_prev_point, d2_poly(DEEP_prev_point), coeffs_d2bar);
  auto d2bar = Polynomial_t::from_coefficients(coeffs_d2bar, 2);
  auto [d2_poly_DEEP, r2] = (d2_poly - d2bar).divide(denom_DEEP1*denom_DEEP2);
  std::cout << "The DEEP d2 degree is: " << d2_poly_DEEP.degree() << std::endl;

  scalar_t coeffs_d3bar[2];
  solve_linear(DEEP_point, d3_poly(DEEP_point), DEEP_prev_point, d3_poly(DEEP_prev_point), coeffs_d3bar);
  auto d3bar = Polynomial_t::from_coefficients(coeffs_d3bar, 2);
  auto [d3_poly_DEEP, r3] = (d3_poly - d3bar).divide(denom_DEEP1*denom_DEEP2);
  std::cout << "The DEEP d3 degree is: " << d3_poly_DEEP.degree() << std::endl;


  // DEEP c{1,2,3} polynomials

  const scalar_t coeffs_c1bar[1] = {c1_poly(DEEP_point)};
  auto c1bar = Polynomial_t::from_coefficients(coeffs_c1bar, 1);
  auto [c1_poly_DEEP, r_c1] = (c1_poly - c1bar).divide(denom_DEEP1);
  std::cout << "The DEEP c1 degree is: " << c1_poly_DEEP.degree() << std::endl;

  const scalar_t coeffs_c2bar[1] = {c2_poly(DEEP_point)};
  auto c2bar = Polynomial_t::from_coefficients(coeffs_c2bar, 1);
  auto [c2_poly_DEEP, r_c2] = (c2_poly - c2bar).divide(denom_DEEP1);
  std::cout << "The DEEP c2 degree is: " << c2_poly_DEEP.degree() << std::endl;

  const scalar_t coeffs_c3bar[1] = {c3_poly(DEEP_point)};
  auto c3bar = Polynomial_t::from_coefficients(coeffs_c3bar, 1);
  auto [c3_poly_DEEP, r_c3] = (c3_poly - c3bar).divide(denom_DEEP1);
  std::cout << "The DEEP c3 degree is: " << c3_poly_DEEP.degree() << std::endl;

  // DEEP validity polynomial
  const scalar_t coeffs_vbar[1] = {p_validity_rs(DEEP_point)};
  auto vbar = Polynomial_t::from_coefficients(coeffs_vbar, 1);
  auto [v_DEEP, r_v] = (p_validity_rs - vbar).divide(denom_DEEP1);
  std::cout << "The DEEP validity polynomial degree is: " << v_DEEP.degree() << std::endl;

  std::cout << "The Prover sends DEEP polynomials to the Verifier" << std::endl;

  std::cout << "Lesson 10: Mixing (Batching) for FRI" << std::endl;
  std::cout << "The initial FRI polynomial is the mix of the 7 DEEP polynomials." << std::endl;

  Polynomial_t* all_DEEP[7];

  all_DEEP[0] = &d1_poly_DEEP;
  all_DEEP[1] = &d2_poly_DEEP;
  all_DEEP[2] = &d3_poly_DEEP;
  all_DEEP[3] = &c1_poly_DEEP;
  all_DEEP[4] = &c2_poly_DEEP;
  all_DEEP[5] = &c3_poly_DEEP;
  all_DEEP[6] = &v_DEEP;

  Polynomial_t fri_input = p_mix(all_DEEP, 7, scalar_t::from(99));
  std::cout << "The degree of the mixed DEEP polynomial is: " << fri_input.degree() << std::endl;

  std::cout << "Lesson 11: FRI Protocol (Commit Phase)" << std::endl;
  std::cout << "The prover provides information to convince the verifier that the DEEP polynomials are low-degree." << std::endl;

  // uint32_t tree_height = (logn + 1) + 1; // extra +1 for larger domain
  // size_t digests_len = get_digests_len<scalar_t>(tree_height, A);
  // // std::cout << "Digests length: " << digests_len << std::endl;
  // scalar_t* digests = new scalar_t[digests_len];
  // TreeBuilderConfig config = default_merkle_config<scalar_t>();
  // build_merkle_tree<scalar_t, T>(commitment, digests, tree_height, constants, config);
  // std::cout << "Root: " << digests[0] << std::endl;

  // std::cout << std::endl << "7. FRI Protocol (Commit Phase)" << std::endl;
  // const int m = 2*n;
  // std::cout << "Split" << std::endl;
  // scalar_t f0_coeffs[m] = {0};
  // scalar_t f0even_coeffs[m/2] = {0};
  // scalar_t f0odd_coeffs[m/2] = {0};
  // auto f0 = f2.clone();

  // auto cc = f0.copy_coefficients_to_host(f0_coeffs, 0, -1);
  // std::cout << "Coefficients: " << cc << std::endl;
  // for (int i = 0; i < m; ++i) {
  //   std::cout << i << ": " << f0_coeffs[i] << std::endl;
  // }
  // std::cout << "Merge" << std::endl;
  // for (int i = 0; i < m/2; ++i) {
  //   f0even_coeffs[i] = f0_coeffs[2*i];
  //   f0odd_coeffs[i] = f0_coeffs[2*i+1];
  //   // std::cout << i << ": even: " << f0even_coeffs[i] << std::endl;
  //   // std::cout << i << ": odd:  " << f0odd_coeffs[i] << std::endl;
  // }
  // auto f0even = Polynomial_t::from_coefficients(f0even_coeffs, m/2);
  // auto f0odd = Polynomial_t::from_coefficients(f0odd_coeffs, m/2);
  // verifier-provided randomness 
  // auto r1 = scalar_t::rand_host();
  // Round 1 polynomial
  // auto f1 = f0even + r1 * f0odd;
  // std::cout << std::endl << "8. FRI Protocol (Query Phase)" << std::endl;
  // std::cout << "Check for consistency" << std::endl;
  // scalar_t xp = scalar_t::rand_host();
  // scalar_t xm = scalar_t::zero() - xp;
  // auto rhs = (r1+xp)*f0(xp)*scalar_t::inverse(scalar_t::from(2)*xp) + (r1+xm)*f0(xm)*scalar_t::inverse(scalar_t::from(2)*xm);
  // auto lhs = f1(xp*xp);
  // std::cout << "rhs: " << rhs << std::endl << "lhs: " << lhs << std::endl;
  // if (lhs != rhs) {
  //   std::cout << "Error: Evaluations are not consistent" << std::endl;
  // } else {
  //   std::cout << "Evaluations are consistent" << std::endl;
  // }
  // auto d1 = f1.degree();
  // auto d0 = f0.degree();
  // std::cout << "Degree: " << d1 << ", degree before: " << d0 << std::endl;
  return 0;
}