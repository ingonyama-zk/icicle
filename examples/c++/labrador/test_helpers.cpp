#include "test_helpers.h"

// print a polynomial
void print_vec(const Zq* vec, size_t len, const std::string& name = "")
{
  if (!name.empty()) { std::cout << name << ": "; }
  std::cout << "[";
  for (size_t i = 0; i < len; ++i) {
    std::cout << vec[i];
    if (i < len - 1) { std::cout << ", "; }
  }
  std::cout << "]" << std::endl;
}

// print a polynomial
void print_poly(const PolyRing& poly, const std::string& name = "") { print_vec(poly.values, PolyRing::d, name); }

/// @brief Compares two vectors of Tq polynomials element-wise
/// @param vec1 First vector to compare
/// @param vec2 Second vector to compare
/// @param size Number of Tq elements to compare
/// @return true if vectors are equal, false otherwise
bool poly_vec_eq(const PolyRing* vec1, const PolyRing* vec2, size_t size)
{
  constexpr size_t d = PolyRing::d;
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < d; j++) {
      if (vec1[i].values[j] != vec2[i].values[j]) { return false; }
    }
  }
  return true;
}

// Generate a random polynomial vector with coefficients bounded by max_value
std::vector<PolyRing> rand_poly_vec(size_t size, int64_t max_value)
{
  std::vector<PolyRing> vec(size);
  for (auto& x : vec) {
    for (size_t i = 0; i < PolyRing::d; ++i) { // randomize each coefficient

      // Uniform in [0, max_value]
      uint64_t val = rand_uint_32b() % (max_value + 1);
      x.values[i] = Zq::from(val);
      // negate with 1/2 probability
      if (rand_uint_32b() % 2 == 0) { x.values[i] = x.values[i].neg(); }
    }
  }
  return vec;
}

// Generate a random EqualityInstance satisfied by the given witness S
std::vector<EqualityInstance> create_rand_eq_inst(size_t n, size_t r, const std::vector<Rq>& S, size_t num_const)
{
  int64_t q = get_q<Zq>();
  // set a and phi completely randomly in Tq
  // view as num_const X r^2 matrix
  const std::vector<Tq> A = rand_poly_vec(num_const*r * r, q);
  // view as num_const X rn matrix
  const std::vector<Tq> Phi = rand_poly_vec(num_const*r * n, q);
  std::vector<Tq> b(num_const);

  // S_hat = NTT(S)
  std::vector<Tq> S_hat(r * n);
  ICICLE_CHECK(ntt(S.data(), r * n, NTTDir::kForward, {}, S_hat.data()));

  std::vector<Tq> S_hat_transposed(n * r);
  ICICLE_CHECK(matrix_transpose<Tq>(S_hat.data(), r, n, {}, S_hat_transposed.data()));

  // G_hat = S@S^t
  std::vector<Tq> G_hat(r * r);
  ICICLE_CHECK(matmul(S_hat.data(), r, n, S_hat_transposed.data(), n, r, {}, G_hat.data()));

  std::vector<Tq> G_A_inner_prod(num_const), phi_S_inner_prod(num_const);
  // G_A_inner_prod = A@G_hat
  ICICLE_CHECK(matmul(A.data(), num_const, r * r, G_hat.data(), r * r, 1, {}, G_A_inner_prod.data()));
  // phi_S_inner_prod = Phi@S
  ICICLE_CHECK(matmul(Phi.data(), num_const, r * n, S_hat.data(), r * n, 1, {}, phi_S_inner_prod.data()));

  // b = -(<G, a> + <S, phi>)
  ICICLE_CHECK(vector_add(G_A_inner_prod.data(), phi_S_inner_prod.data(), num_const, {}, b.data()));
  Zq minus_1 = Zq::from(1).neg();
  ICICLE_CHECK(scalar_mul_vec(&minus_1, reinterpret_cast<Zq*>(b.data()), Rq::d*num_const, {}, reinterpret_cast<Zq*>(b.data())));

  std::vector<EqualityInstance> eq_inst; 

  for(size_t i=0; i<num_const; i++){
    EqualityInstance new_eq_inst{r,n, {&A[i*r*r], &A[i*r*r]+r*r}, {&Phi[i*r*n], &Phi[i*r*n]+r*n}, b[i]};
    eq_inst.push_back(new_eq_inst);
  }
  // Now S is a witness for the equality constraint eq_inst
  return eq_inst;
}

// Generate a random ConstZeroInstance satisfied by the given witness S
std::vector<ConstZeroInstance> create_rand_const_zero_inst(size_t n, size_t r, const std::vector<Rq>& S, size_t num_const)
{
  int64_t q = get_q<Zq>();
  const std::vector<EqualityInstance> eq_inst = create_rand_eq_inst(n, r, S, num_const);
  // set a, phi equal to the random EqualityInstance
  std::vector<ConstZeroInstance> const_zero_inst;
  for(size_t i=0; i<num_const; i++){
    ConstZeroInstance new_cz_inst{r, n, eq_inst[i].a, eq_inst[i].phi, Zq::zero()};

    // For b only set const coeff equal to the one in eq_inst[i].b
    // eq_inst_b_unhat = INTT(eq_inst[i].b)
    Rq eq_inst_b_unhat;
    ICICLE_CHECK(ntt(&eq_inst[i].b, 1, NTTDir::kInverse, {}, &eq_inst_b_unhat));

    // set new_cz_inst.b equal to const coeff of eq_inst_b
    new_cz_inst.b = eq_inst_b_unhat.values[0];
    const_zero_inst.push_back(new_cz_inst);
  }
  return const_zero_inst;
}

bool witness_legit_eq_all_ntt(const EqualityInstance& eq_inst, const std::vector<Tq>& S_hat)
{
  size_t r = eq_inst.r;
  size_t n = eq_inst.n;

  assert(S_hat.size() == r * n);

  std::vector<Tq> S_hat_transposed(n * r);
  ICICLE_CHECK(matrix_transpose<Tq>(S_hat.data(), r, n, {}, S_hat_transposed.data()));

  // G_hat = S@S^t
  std::vector<Tq> G_hat(r * r);
  ICICLE_CHECK(matmul(S_hat.data(), r, n, S_hat_transposed.data(), n, r, {}, G_hat.data()));

  Tq G_A_inner_prod, phi_S_inner_prod, eval_hat;
  // G_A_inner_prod = <G, a>
  ICICLE_CHECK(matmul(G_hat.data(), 1, r * r, eq_inst.a.data(), r * r, 1, {}, &G_A_inner_prod));
  // phi_S_inner_prod = <S, phi>
  ICICLE_CHECK(matmul(S_hat.data(), 1, r * n, eq_inst.phi.data(), r * n, 1, {}, &phi_S_inner_prod));

  // eval_hat = b + (<G, a> + <S, phi>)
  ICICLE_CHECK(vector_add(G_A_inner_prod.values, phi_S_inner_prod.values, Rq::d, {}, eval_hat.values));
  ICICLE_CHECK(vector_add(eval_hat.values, eq_inst.b.values, Rq::d, {}, eval_hat.values));

  // print_poly(eval_hat, "eval_hat");
  for (size_t i = 0; i < Tq::d; i++) {
    if (eval_hat.values[i] != Zq::from(0)) { return false; }
  }
  return true;
}

// Check if the given EqualityInstance is satisfied by the witness S or not
bool witness_legit_eq(const EqualityInstance& eq_inst, const std::vector<Rq>& S)
{
  size_t r = eq_inst.r;
  size_t n = eq_inst.n;

  assert(S.size() == r * n);
  // S_hat = NTT(S)
  std::vector<Tq> S_hat(r * n);
  ICICLE_CHECK(ntt(S.data(), r * n, NTTDir::kForward, {}, S_hat.data()));

  return witness_legit_eq_all_ntt(eq_inst, S_hat);
}

bool witness_legit_const_zero_all_ntt(const ConstZeroInstance& cz_inst, const std::vector<Tq>& S_hat)
{
  size_t r = cz_inst.r;
  size_t n = cz_inst.n;

  assert(S_hat.size() == r * n);

  std::vector<Tq> S_hat_transposed(n * r);
  ICICLE_CHECK(matrix_transpose<Tq>(S_hat.data(), r, n, {}, S_hat_transposed.data()));

  // G_hat = S@S^t
  std::vector<Tq> G_hat(r * r);
  ICICLE_CHECK(matmul(S_hat.data(), r, n, S_hat_transposed.data(), n, r, {}, G_hat.data()));

  Tq G_A_inner_prod, phi_S_inner_prod, eval_hat;
  // G_A_inner_prod = <G, a>
  ICICLE_CHECK(matmul(G_hat.data(), 1, r * r, cz_inst.a.data(), r * r, 1, {}, &G_A_inner_prod));
  // phi_S_inner_prod = <S, phi>
  ICICLE_CHECK(matmul(S_hat.data(), 1, r * n, cz_inst.phi.data(), r * n, 1, {}, &phi_S_inner_prod));

  // eval_hat = (<G, a> + <S, phi>)
  ICICLE_CHECK(vector_add(G_A_inner_prod.values, phi_S_inner_prod.values, Rq::d, {}, eval_hat.values));

  // take INTT for eval_hat
  Rq eval;
  ICICLE_CHECK(ntt(&eval_hat, 1, NTTDir::kInverse, {}, &eval));

  // print_poly(eval, "cz_eval");
  if (eval.values[0] + cz_inst.b == Zq::zero()) {
    return true;
  } else {
    return false;
  }
}

// Check if the given ConstZeroInstance is satisfied by the witness S or not
bool witness_legit_const_zero(const ConstZeroInstance& cz_inst, const std::vector<Rq>& S)
{
  size_t r = cz_inst.r;
  size_t n = cz_inst.n;

  // S_hat = NTT(S)
  std::vector<Tq> S_hat(r * n);
  ICICLE_CHECK(ntt(S.data(), r * n, NTTDir::kForward, {}, S_hat.data()));

  return witness_legit_const_zero_all_ntt(cz_inst, S_hat);
}

bool lab_witness_legit(const LabradorInstance& lab_inst, const std::vector<Rq>& S)
{
  size_t r = lab_inst.param.r;
  size_t n = lab_inst.param.n;
  // S_hat = NTT(S)
  std::vector<Tq> S_hat(r * n);
  ICICLE_CHECK(ntt(S.data(), r * n, NTTDir::kForward, {}, S_hat.data()));

  for (auto& eq_inst : lab_inst.equality_constraints) {
    if (!witness_legit_eq_all_ntt(eq_inst, S_hat)) { return false; }
  }
  for (auto& cz_inst : lab_inst.const_zero_constraints) {
    if (!witness_legit_const_zero_all_ntt(cz_inst, S_hat)) { return false; }
  }
  return true;
}
