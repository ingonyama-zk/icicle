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

struct QuadraticConstraint {
  size_t r;            // Number of witness vectors
  size_t n;            // Dimension of each vector in Tq
  std::vector<Tq> a;   // a[i, j] matrix over Tq (r x r matrix)
  std::vector<Tq> phi; // phi[i,j] vector over Tq (r vectors, each of size n; arranged in row major)
  Tq b;                // Polynomial in Tq

  QuadraticConstraint(size_t r, size_t n) : r(r), n(n), a(r * r, zero()), phi(r * n, zero()), b(zero()) {}
  QuadraticConstraint(size_t r, size_t n, std::vector<Tq> a, std::vector<Tq> phi, Tq b)
      : r(r), n(n), a(a), phi(phi), b(b)
  {
    // check if the sizes of a and phi are correct
    if (a.size() != r * r || phi.size() != r * n) {
      throw std::invalid_argument("QuadraticConstraint: Incorrect sizes for 'a' or 'phi'");
    }
  }

  // Copy constructor
  QuadraticConstraint(const QuadraticConstraint& other) : r(other.r), n(other.n), a(other.a), phi(other.phi), b(other.b)
  {
  }
};
using ConstZeroInstance = QuadraticConstraint;
using EqualityInstance = QuadraticConstraint;

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
    if (instance.r == r || instance.n == n) {
      throw std::invalid_argument("EqualityInstance not compatible with LabradorInstance");
    }
    equality_constraints.push_back(instance);
  }

  // Add a ConstZeroInstance
  void add_const_zero_constraint(const ConstZeroInstance& instance)
  {
    if (instance.r == r || instance.n == n) {
      throw std::invalid_argument("ConstZeroInstance not compatible with LabradorInstance");
    }
    const_zero_constraints.push_back(instance);
  }

  std::vector<Tq> agg_const_zero_constraints(
    size_t num_aggregation_rounds,
    size_t JL_out,
    const std::vector<Tq>& S_hat,
    const std::vector<Tq>& g_hat,
    std::vector<Rq>& Q,
    const std::vector<Zq>& psi,
    const std::vector<Zq>& omega);

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
  EqualityInstance final_const;
  std::vector<Tq> z_hat;
  std::vector<Rq> t;
  std::vector<Rq> g;
  std::vector<Rq> h;

  LabradorBaseCaseProof(size_t r, size_t n) : final_const(r, n), z_hat(), t(), g(), h() {};
  LabradorBaseCaseProof(
    EqualityInstance final_const, std::vector<Tq> z_hat, std::vector<Tq> t, std::vector<Tq> g, std::vector<Tq> h)
      : final_const(final_const), z_hat(std::move(z_hat)), t(std::move(t)), g(std::move(g)), h(std::move(h))
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

  std::pair<size_t, std::vector<Zq>> select_valid_jl_proj(std::byte* seed, size_t seed_len) const;
  std::pair<LabradorBaseCaseProof, PartialTranscript> base_case_prover();
};

struct LabradorBaseVerifier {
  LabradorInstance lab_inst;
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