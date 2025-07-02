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
      if (rand_uint_32b() % 2 == 0) { x.values[i] = Zq::neg(x.values[i]); }
    }
  }
  return vec;
}

// Generate a random EqualityInstance satisfied by the given witness S
EqualityInstance create_rand_eq_inst(size_t n, size_t r, const std::vector<Rq>& S)
{
  int64_t q = get_q<Zq>();
  // set a and phi completely randomly in Tq
  EqualityInstance eq_inst{r, n, rand_poly_vec(r * r, q), rand_poly_vec(n * r, q), zero()};

  // S_hat = NTT(S)
  std::vector<Tq> S_hat(r * n);
  ICICLE_CHECK(ntt(S.data(), r * n, NTTDir::kForward, {}, S_hat.data()));

  std::vector<Tq> S_hat_transposed(n * r);
  ICICLE_CHECK(matrix_transpose<Tq>(S_hat.data(), r, n, {}, S_hat_transposed.data()));

  // G_hat = S@S^t
  std::vector<Tq> G_hat(r * r);
  ICICLE_CHECK(matmul(S_hat.data(), r, n, S_hat_transposed.data(), n, r, {}, G_hat.data()));

  Tq G_A_inner_prod, phi_S_inner_prod;
  // G_A_inner_prod = <G, a>
  ICICLE_CHECK(matmul(G_hat.data(), 1, r * r, eq_inst.a.data(), r * r, 1, {}, &G_A_inner_prod));
  // phi_S_inner_prod = <S, phi>
  ICICLE_CHECK(matmul(S_hat.data(), 1, r * n, eq_inst.phi.data(), r * n, 1, {}, &phi_S_inner_prod));

  // b = -(<G, a> + <S, phi>)
  ICICLE_CHECK(vector_add(G_A_inner_prod.values, phi_S_inner_prod.values, Rq::d, {}, eq_inst.b.values));
  Zq minus_1 = Zq::neg(Zq::from(1));
  ICICLE_CHECK(scalar_mul_vec(&minus_1, eq_inst.b.values, Rq::d, {}, eq_inst.b.values));
  // Now S is a witness for the equality constraint eq_inst
  return eq_inst;
}

// Generate a random ConstZeroInstance satisfied by the given witness S
ConstZeroInstance create_rand_const_zero_inst(size_t n, size_t r, const std::vector<Rq>& S)
{
  int64_t q = get_q<Zq>();
  EqualityInstance eq_inst = create_rand_eq_inst(n, r, S);
  // set a, phi equal to the random EqualityInstance
  ConstZeroInstance const_zero_inst{r, n, eq_inst.a, eq_inst.phi, Zq::zero()};

  // For b only set const coeff equal to the one in eq_inst.b
  // eq_inst_b = INTT(eq_inst.b)
  Rq eq_inst_b;
  ICICLE_CHECK(ntt(&eq_inst.b, 1, NTTDir::kInverse, {}, &eq_inst_b));

  Rq rand_b = rand_poly_vec(1, q)[0];
  // make const coeff of rand_b equal to that of eq_inst_b

  const_zero_inst.b = eq_inst_b.values[0];
  return const_zero_inst;
}

// Check if the given EqualityInstance is satisfied by the witness S or not
bool witness_legit_eq(const EqualityInstance& eq_inst, const std::vector<Rq>& S)
{
  int64_t q = get_q<Zq>();
  size_t r = eq_inst.r;
  size_t n = eq_inst.n;

  assert(S.size() == r * n);
  // S_hat = NTT(S)
  std::vector<Tq> S_hat(r * n);
  ICICLE_CHECK(ntt(S.data(), r * n, NTTDir::kForward, {}, S_hat.data()));

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

// Check if the given ConstZeroInstance is satisfied by the witness S or not
bool witness_legit_const_zero(const ConstZeroInstance& cz_inst, const std::vector<Rq>& S)
{
  int64_t q = get_q<Zq>();
  size_t r = cz_inst.r;
  size_t n = cz_inst.n;

  // S_hat = NTT(S)
  std::vector<Tq> S_hat(r * n);
  ICICLE_CHECK(ntt(S.data(), r * n, NTTDir::kForward, {}, S_hat.data()));

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

bool lab_witness_legit(const LabradorInstance& lab_inst, const std::vector<Rq>& S)
{
  for (auto& eq_inst : lab_inst.equality_constraints) {
    if (!witness_legit_eq(eq_inst, S)) { return false; }
  }
  for (auto& cz_inst : lab_inst.const_zero_constraints) {
    if (!witness_legit_const_zero(cz_inst, S)) { return false; }
  }
  return true;
}

void test_jl()
{
  size_t n = 100, JL_out = 32;
  std::vector<Rq> p = rand_poly_vec(n, 8);
  // y = Pi*p
  std::vector<Zq> y(JL_out);
  const char* jl_seed = "RAND";
  ICICLE_CHECK(jl_projection(
    reinterpret_cast<const Zq*>(p.data()), n * Rq::d, reinterpret_cast<const std::byte*>(&jl_seed), 4, {}, y.data(),
    JL_out));

  print_vec(y.data(), y.size(), "y = Pi*p");

  // std::vector<Zq> Pi(JL_out * n * Rq::d);
  // // compute the Pi matrix, conjugated in Rq
  // ICICLE_CHECK(get_jl_matrix_rows<Zq>(
  //   reinterpret_cast<const std::byte*>(&jl_seed), 4,
  //   n * Rq::d, // row_size
  //   0,         // row_index
  //   JL_out,    // num_rows
  //   false,     // conjugate
  //   {},        // config
  //   Pi.data()));

  std::vector<Rq> Q(JL_out * n);
  // compute the Pi matrix, conjugated in Rq
  ICICLE_CHECK(get_jl_matrix_rows<Rq>(
    reinterpret_cast<const std::byte*>(&jl_seed), 4,
    n,      // row_size
    0,      // row_index
    JL_out, // num_rows
    true,   // conjugate
    {},     // config
    Q.data()));

  std::vector<Tq> Q_hat(JL_out * n), p_hat(n), z_hat(JL_out);
  ICICLE_CHECK(ntt(Q.data(), JL_out * n, NTTDir::kForward, {}, Q_hat.data()));
  ICICLE_CHECK(ntt(p.data(), n, NTTDir::kForward, {}, p_hat.data()));
  // z_hat = Q_hat * p_hat
  ICICLE_CHECK(matmul(Q_hat.data(), JL_out, n, p_hat.data(), n, 1, {}, z_hat.data()));
  std::vector<Rq> z(JL_out);
  ICICLE_CHECK(ntt(z_hat.data(), JL_out, NTTDir::kInverse, {}, z.data()));

  bool succ = true;
  for (int i = 0; i < JL_out; i++) {
    // const(z) == y
    if (z[i].values[0] != y[i]) { succ = false; }
  }
  if (succ) {
    std::cout << "Success\n";
  } else {
    std::cout << "Fail\n";
  }
}
