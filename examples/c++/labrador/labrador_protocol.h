#include "examples_utils.h"
#include "icicle/runtime.h"
#include "icicle/lattice/labrador.h"

using namespace icicle::labrador;

PolyRing zero()
{
  PolyRing z;
  for (size_t i = 0; i < PolyRing::d; i++) {
    z.values[i] = Zq::zero();
  }
  return z;
}

struct EqualityInstance {
  size_t r;            // Number of witness vectors
  size_t n;            // Dimension of each vector in Tq
  std::vector<Tq> a;   // a[i, j] matrix over Tq (r x r matrix)
  std::vector<Tq> phi; // phi[i,j] vector over Tq (r vectors, each of size n; arranged in row major)
  Tq b;                // Polynomial in Tq

  EqualityInstance(size_t r, size_t n) : r(r), n(n), a(r * r, zero()), phi(r * n, zero()), b(zero()) {}
  EqualityInstance(size_t r, size_t n, const std::vector<Tq>& a, const std::vector<Tq>& phi, Tq b)
      : r(r), n(n), a(a), phi(phi), b(b)
  {
    // check if the sizes of a and phi are correct
    if (a.size() != r * r || phi.size() != r * n) {
      throw std::invalid_argument("EqualityInstance: Incorrect sizes for 'a' or 'phi'");
    }
  }

  // Copy constructor
  EqualityInstance(const EqualityInstance& other) : r(other.r), n(other.n), a(other.a), phi(other.phi), b(other.b) {}
};

struct ConstZeroInstance {
  size_t r;            // Number of witness vectors
  size_t n;            // Dimension of each vector in Tq
  std::vector<Tq> a;   // a[i, j] matrix over Tq (r x r matrix)
  std::vector<Tq> phi; // phi[i,j] vector over Tq (r vectors, each of size n; arranged in row major)
  Zq b;                // Such that \sum_ij a[i,j]<s[i], s[j]> + \sum_i <phi[i], s[i]> + b has 0 const coeff

  ConstZeroInstance(size_t r, size_t n) : r(r), n(n), a(r * r, zero()), phi(r * n, zero()), b(Zq::zero()) {}
  ConstZeroInstance(size_t r, size_t n, const std::vector<Tq>& a, const std::vector<Tq>& phi, Zq b)
      : r(r), n(n), a(a), phi(phi), b(b)
  {
    // check if the sizes of a and phi are correct
    if (a.size() != r * r || phi.size() != r * n) {
      throw std::invalid_argument("EqualityInstance: Incorrect sizes for 'a' or 'phi'");
    }
  }

  // Copy constructor
  ConstZeroInstance(const ConstZeroInstance& other) : r(other.r), n(other.n), a(other.a), phi(other.phi), b(other.b) {}
};

struct LabradorParam {
  // Seed to calculate Ajtai Matrix
  std::vector<std::byte> ajtai_seed;

  // Matrix dimensions for Ajtai commitments
  size_t kappa;  // Ajtai matrix A dimensions: n × kappa
  size_t kappa1; // Matrix B,C dimensions for committing to decomposed vectors
  size_t kappa2; // Matrix D dimensions for committing to h vectors

  // Decomposition bases
  uint32_t base1; // Base for decomposing T
  uint32_t base2; // Base for decomposing g
  uint32_t base3; // Base for decomposing h

  // JL projection parameters
  size_t JL_out = 256; // Output dimension for Johnson-Lindenstrauss projection (typically 256)

  // Norm bounds
  double beta;                 // Witness norm bound
  uint64_t op_norm_bound = 15; // Operator norm bound for challenges

  // Constructor with default values matching the base implementation
  LabradorParam(
    const std::vector<std::byte>& ajtai_seed,
    size_t kappa,
    size_t kappa1,
    size_t kappa2,
    size_t base1,
    size_t base2,
    size_t base3,
    double beta)
      : ajtai_seed(ajtai_seed), kappa(kappa), kappa1(kappa1), kappa2(kappa2), base1(base1), base2(base2), base3(base3),
        beta(beta)
  {
  }
  // Copy constructor
  LabradorParam(const LabradorParam& other)
      : ajtai_seed(other.ajtai_seed), kappa(other.kappa), kappa1(other.kappa1), kappa2(other.kappa2),
        base1(other.base1), base2(other.base2), base3(other.base3), beta(other.beta)
  {
  }

  size_t t_len(size_t r) const
  {
    size_t l1 = icicle::balanced_decomposition::compute_nof_digits<Zq>(base1);
    return l1 * r * kappa;
  }
  size_t g_len(size_t r) const
  {
    size_t l2 = icicle::balanced_decomposition::compute_nof_digits<Zq>(base2);
    size_t r_choose_2 = (r * (r + 1)) / 2;
    return (l2 * r_choose_2);
  }
  size_t h_len(size_t r) const
  {
    size_t l3 = icicle::balanced_decomposition::compute_nof_digits<Zq>(base3);
    size_t r_choose_2 = (r * (r + 1)) / 2;
    return (l3 * r_choose_2);
  }
};

struct LabradorInstance {
  size_t r;                                              // Number of witness vectors
  size_t n;                                              // Dimension of each vector in Tq
  LabradorParam param;                                   // LabradorParam for this instance
  std::vector<EqualityInstance> equality_constraints;    // K EqualityInstances
  std::vector<ConstZeroInstance> const_zero_constraints; // L ConstZeroInstances

  LabradorInstance(size_t r, size_t n, const LabradorParam& param) : r(r), n(n), param(param) {}

  // Copy constructor
  LabradorInstance(const LabradorInstance& other)
      : r(other.r), n(other.n), param(other.param), equality_constraints(other.equality_constraints),
        const_zero_constraints(other.const_zero_constraints)
  {
  }

  // Add an EqualityInstance
  void add_equality_constraint(const EqualityInstance& instance)
  {
    if (instance.r != r || instance.n != n) {
      throw std::invalid_argument("EqualityInstance not compatible with LabradorInstance");
    }
    equality_constraints.push_back(instance);
  }

  // Add a ConstZeroInstance
  void add_const_zero_constraint(const ConstZeroInstance& instance)
  {
    if (instance.r != r || instance.n != n) {
      throw std::invalid_argument("ConstZeroInstance not compatible with LabradorInstance");
    }
    const_zero_constraints.push_back(instance);
  }

  void agg_equality_constraints(const std::vector<Tq>& alpha_hat);
};

struct PartialTranscript {
  // committed by the Prover
  std::vector<Tq> u1;
  size_t JL_i;
  std::vector<Zq> p;
  std::vector<Tq> b_agg;
  std::vector<Tq> u2;

  // hash evaluations
  std::vector<std::byte> seed1;
  std::vector<std::byte> seed2;
  std::vector<std::byte> seed3;
  std::vector<std::byte> seed4;

  // challenges- stored for convenience
  std::vector<Zq> psi;
  std::vector<Zq> omega;
  std::vector<Tq> alpha_hat;
  std::vector<Tq> challenges_hat;

  /// @brief Returns the size of the partial proof (only includes necessary elements)
  size_t proof_size()
  {
    return sizeof(Zq) * (u1.size() * Tq::d + p.size() + b_agg.size() * Tq::d + u2.size() * Tq::d) + sizeof(size_t);
  }
};

/// Encapsulates the problem and witness for the reduced instance
///
/// final_const: is the EqualityInstance prepared in Step 22
///
/// z_hat: is the vector computed in Step 29
///
/// t: vector computed in Step 9 (T_tilde in the code)
///
/// g: vector computed in Step 9 (g_tilde in the code)
///
/// h: vector computed in Step 25 (H_tilde in the code)
struct LabradorBaseCaseProof {
  std::vector<Tq> z_hat;
  std::vector<Rq> t;
  std::vector<Rq> g;
  std::vector<Rq> h;

  // TODO: Only for testing. Need to remove this
  EqualityInstance final_const;

  LabradorBaseCaseProof(size_t r, size_t n) : final_const(r, n), z_hat(), t(), g(), h() {};
  LabradorBaseCaseProof(
    EqualityInstance final_const, std::vector<Tq> z_hat, std::vector<Tq> t, std::vector<Tq> g, std::vector<Tq> h)
      : final_const(final_const), z_hat(z_hat), t(t), g(g), h(h)
  {
  }
};

struct LabradorBaseProver {
  LabradorInstance lab_inst;
  // S consists of r vectors of dim n arranged in row major order
  std::vector<Rq> S;

  LabradorBaseProver(const LabradorInstance& lab_inst, const std::vector<Rq>& S) : lab_inst(lab_inst), S(S)
  {
    if (S.size() != lab_inst.r * lab_inst.n) { throw std::invalid_argument("S must have size r * n"); }
  }

  std::vector<Tq> agg_const_zero_constraints(
    size_t num_aggregation_rounds,
    size_t JL_out,
    const std::vector<Tq>& S_hat,
    const std::vector<Tq>& g_hat,
    const std::vector<Zq>& p,
    const std::vector<Tq>& Q_hat,
    const std::vector<Zq>& psi,
    const std::vector<Zq>& omega);
  std::pair<size_t, std::vector<Zq>> select_valid_jl_proj(std::byte* seed, size_t seed_len) const;
  std::pair<LabradorBaseCaseProof, PartialTranscript> base_case_prover();
};

struct LabradorBaseVerifier {
  LabradorInstance lab_inst;
  PartialTranscript trs;
  LabradorBaseCaseProof base_proof;

  LabradorBaseVerifier(
    const LabradorInstance& lab_inst, const PartialTranscript& trs, const LabradorBaseCaseProof& base_proof)
      : lab_inst(lab_inst), trs(trs), base_proof(base_proof)
  {
  }
  bool _verify_base_proof() const;
  bool verify() const;
};

struct LabradorProver {
  LabradorInstance lab_inst;
  // S consists of r vectors of dim n arranged in row major order
  std::vector<Rq> S;
  const size_t NUM_REC;

  LabradorProver(LabradorInstance& lab_inst, std::vector<Rq>& S, size_t NUM_REC)
      : lab_inst(lab_inst), S(S), NUM_REC(NUM_REC)
  {
  }

  std::vector<Rq> prepare_recursion_witness(
    const PartialTranscript& trs, const LabradorBaseCaseProof& pf, size_t base0, size_t mu, size_t nu);

  std::pair<std::vector<PartialTranscript>, LabradorBaseCaseProof> prove();
};

struct LabradorVerifier {
  LabradorInstance lab_inst;
};

// Helper functions:

template <typename Zq>
int64_t get_q()
{
  constexpr auto q_storage = Zq::get_modulus();
  const int64_t q = *(int64_t*)&q_storage; // Note this is valid since TLC == 2
  return q;
}

eIcicleError scale_diagonal_with_mask(
  const Tq* matrix,  // Input n×n matrix (row-major order)
  Zq scaling_factor, // Factor to scale diagonal by
  size_t n,          // Matrix dimension (n×n)
  const VecOpsConfig& config,
  Tq* output) // Output matrix
{
  // Create scaling mask: diagonal = scaling_factor, off-diagonal = 1
  size_t d = Tq::d;
  std::vector<Zq> mask(n * d * n * d, Zq::from(1));

  // Set diagonal elements to scaling factor
  for (uint64_t i = 0; i < n; i++) {
    std::fill(&mask[i * n * d + i], &mask[i * n * d + i + d], scaling_factor);
  }

  // Use vector_mul to apply the mask
  return vector_mul(
    reinterpret_cast<const Zq*>(matrix), mask.data(), n * d * n * d, config, reinterpret_cast<Zq*>(output));
}

/// extracts the symmetric part of a n X n matrix as a n(n+1)/2 size vector
template <typename T>
std::vector<T> extract_symm_part(T* mat, size_t n)
{
  size_t n_choose_2 = (n * (n + 1)) / 2;
  std::vector<T> v(n_choose_2);
  size_t offset = 0;

  // Create stream for async operations
  icicleStreamHandle stream;
  icicle_create_stream(&stream);

  for (size_t i = 0; i < n; i++) {
    icicle_copy_async(&v[offset], &mat[i * n + i], (n - i) * sizeof(T), stream);
    offset += n - i;
  }

  // Synchronize to ensure all copies complete before returning
  icicle_stream_synchronize(stream);
  icicle_destroy_stream(stream);

  return v;
}

/// reconstructs a symmetric n × n matrix from its packed upper–triangular
/// representation produced by extract_symm_part.
/// The returned matrix is in row-major order and satisfies
///     v == extract_symm_part(M.data(), n)
template <typename T>
std::vector<T> reconstruct_symm_matrix(const std::vector<T>& v, size_t n)
{
  // Make sure the caller supplied the right-sized vector
  assert(v.size() == (n * (n + 1)) / 2 && "Packed vector has wrong length");

  std::vector<T> mat(n * n);

  // Copy the packed upper-triangular part row by row.
  icicleStreamHandle stream;
  icicle_create_stream(&stream);

  size_t offset = 0;
  for (size_t i = 0; i < n; ++i) {
    size_t len = n - i; // -- elements on and right of the diagonal
    icicle_copy_async(
      &mat[i * n + i], // destination start  (row i, col i)
      &v[offset],      // source start
      len * sizeof(T), // bytes to copy
      stream);
    offset += len;
  }

  // Wait for the DMA copies to complete before we mirror the data
  icicle_stream_synchronize(stream);
  icicle_destroy_stream(stream);

  // Mirror the upper-triangular entries to the lower-triangular part
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      mat[j * n + i] = mat[i * n + j];
    }
  }

  return mat;
}

// === Fns for testing ===

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
    for (size_t i = 0; i < PolyRing::d; ++i) {          // randomize each coefficient
      uint64_t val = rand_uint_32b() % (max_value + 1); // uniform in [0, max_value]
      x.values[i] = Zq::from(val);
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
