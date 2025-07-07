#pragma once

#include "labrador.h"
#include "utils.h"
#include <cstddef>
#include <stdexcept>
#include <vector>

using namespace icicle::labrador;

/* ======================================================================
 *  Constraint descriptions
 * ====================================================================*/

constexpr uint64_t OP_NORM_BOUND = 15;

/// @brief Struct for storing an equality instance of the form:
/// \sum_ij a[i,j]<s[i], s[j]> + \sum_i <phi[i], s[i]> + b = 0
struct EqualityInstance {
  /// Number of witness vectors
  size_t r;
  /// Dimension of each vector in Tq
  size_t n;
  /// a[i,j]  – r×r  matrix over Tq
  std::vector<Tq> a;
  /// phi[i,j] – r vectors, each length n  (row-major)
  std::vector<Tq> phi;
  /// Polynomial in Tq
  Tq b;

  // constructors

  EqualityInstance(size_t r, size_t n) : r(r), n(n), a(r * r, zero()), phi(r * n, zero()), b(zero()) {}

  EqualityInstance(size_t r, size_t n, const std::vector<Tq>& a, const std::vector<Tq>& phi, const Tq& b)
      : r(r), n(n), a(a), phi(phi), b(b)
  {
    if (a.size() != r * r || phi.size() != r * n)
      throw std::invalid_argument("EqualityInstance: incorrect 'a' or 'phi' size");
  }

  EqualityInstance(const EqualityInstance& o) = default;
};

/// @brief Struct for storing a constant-zero constraints of the form:
/// constant(\sum_ij a[i,j]<s[i], s[j]> + \sum_i <phi[i], s[i]> + b) = 0
struct ConstZeroInstance {
  /// Number of witness vectors
  size_t r;
  /// Dimension of each vector in Tq
  size_t n;
  /// a[i,j] – r×r matrix over Tq
  std::vector<Tq> a;
  /// phi[i,j] – r vectors, each length n (row-major)
  std::vector<Tq> phi;
  /// Constant term b such that the entire expression has zero constant coefficient
  Zq b;

  // constructors

  ConstZeroInstance(size_t r, size_t n) : r(r), n(n), a(r * r, zero()), phi(r * n, zero()), b(Zq::zero()) {}

  ConstZeroInstance(size_t r, size_t n, const std::vector<Tq>& a, const std::vector<Tq>& phi, Zq b)
      : r(r), n(n), a(a), phi(phi), b(b)
  {
    if (a.size() != r * r || phi.size() != r * n)
      throw std::invalid_argument("ConstZeroInstance: incorrect ‘a’ or ‘phi’ size");
  }

  ConstZeroInstance(const ConstZeroInstance& o) = default;
};

/* ======================================================================
 *  Protocol parameters
 * ====================================================================*/

/// @brief struct for storing parameter for the Labrador protocol
struct LabradorParam {
  /// Problem size

  /// Number of witness vectors
  size_t r;
  /// Dimension of each vector in Tq
  size_t n;

  /// Seed for Ajtai matrix generation
  std::vector<std::byte> ajtai_seed;

  /// Matrix dimensions for Ajtai commitments

  /// Ajtai matrix A dimensions: n × kappa
  size_t kappa;
  /// Matrix B,C dimensions for committing to decomposed vectors (t,g)
  size_t kappa1;
  /// Matrix D dimensions for committing to decomposed h vectors
  size_t kappa2;

  /// Store Ajtai matrices
  std::vector<Tq> A, B, C, D;

  /// Decomposition bases

  /// Base for decomposing t
  uint32_t base1;
  /// Base for decomposing g
  uint32_t base2;
  /// Base for decomposing h
  uint32_t base3;

  /// JL projection parameters

  /// Output dimension for Johnson-Lindenstrauss projection (typically 256)
  size_t JL_out = 256;

  /// Norm bounds
  /// Witness norm bound
  double beta;
  /// Operator norm bound for challenges
  uint64_t op_norm_bound = OP_NORM_BOUND;

  /// Number of times aggregation is repeated for constant zero constraints
  size_t num_aggregation_rounds = std::ceil(128.0 / std::log2(get_q<Zq>())); // = 3

  // constructors

  LabradorParam(
    size_t r,
    size_t n,
    const std::vector<std::byte>& ajtai_seed,
    size_t kappa,
    size_t kappa1,
    size_t kappa2,
    uint32_t base1,
    uint32_t base2,
    uint32_t base3,
    double beta)
      : r(r), n(n), ajtai_seed(ajtai_seed), kappa(kappa), kappa1(kappa1), kappa2(kappa2), A(), B(), C(), D(),
        base1(base1), base2(base2), base3(base3), beta(beta)
  {
    std::vector<std::byte> seed_A(ajtai_seed), seed_B(ajtai_seed), seed_C(ajtai_seed), seed_D(ajtai_seed);
    seed_A.push_back(std::byte('0'));
    seed_B.push_back(std::byte('1'));
    seed_C.push_back(std::byte('2'));
    seed_D.push_back(std::byte('3'));

    A.resize(n * kappa);
    B.resize(t_len() * kappa1);
    C.resize(g_len() * kappa1);
    D.resize(h_len() * kappa2);

    VecOpsConfig async_config = default_vec_ops_config();
    async_config.is_async = true;

    ICICLE_CHECK(random_sampling(A.size(), true, seed_A.data(), seed_A.size(), async_config, A.data()));
    ICICLE_CHECK(random_sampling(B.size(), true, seed_B.data(), seed_B.size(), async_config, B.data()));
    ICICLE_CHECK(random_sampling(C.size(), true, seed_C.data(), seed_C.size(), async_config, C.data()));
    ICICLE_CHECK(random_sampling(D.size(), true, seed_D.data(), seed_D.size(), async_config, D.data()));

    ICICLE_CHECK(icicle_device_synchronize());
  }

  LabradorParam(const LabradorParam& o) = default;

  /* helper lengths for base proof vectors --------------------------------*/
  size_t t_len() const
  {
    size_t l1 = icicle::balanced_decomposition::compute_nof_digits<Zq>(base1);
    return l1 * r * kappa;
  }

  size_t g_len() const
  {
    size_t l2 = icicle::balanced_decomposition::compute_nof_digits<Zq>(base2);
    size_t r_choose_2 = (r * (r + 1)) / 2;
    return (l2 * r_choose_2);
  }

  size_t h_len() const
  {
    size_t l3 = icicle::balanced_decomposition::compute_nof_digits<Zq>(base3);
    size_t r_choose_2 = (r * (r + 1)) / 2;
    return (l3 * r_choose_2);
  }
};

/* ======================================================================
 *  Instance to be proved
 * ====================================================================*/

/// An instance of the Labrador problem: consists of multiple equality constraints and constant zero constraints
struct LabradorInstance {
  /// LabradorParam for this instance
  LabradorParam param;
  /// Equality constraints
  std::vector<EqualityInstance> equality_constraints;
  /// Const-zero constraints
  std::vector<ConstZeroInstance> const_zero_constraints;

  // constructors

  LabradorInstance(const LabradorParam& p) : param(p) {}
  LabradorInstance(const LabradorInstance&) = default;

  /* -------- constraint helpers ---------------------------------------- */
  void add_equality_constraint(const EqualityInstance& inst)
  {
    if (inst.r != param.r || inst.n != param.n)
      throw std::invalid_argument("EqualityInstance incompatible with LabradorInstance");
    equality_constraints.push_back(inst);
  }

  void add_const_zero_constraint(const ConstZeroInstance& inst)
  {
    if (inst.r != param.r || inst.n != param.n)
      throw std::invalid_argument("ConstZeroInstance incompatible with LabradorInstance");
    const_zero_constraints.push_back(inst);
  }

  /// @brief Aggregates all equality constraints into a single equality constraint by creating a random linear
  /// combination of the constraints using the random polynomials in alpha_hat
  void agg_equality_constraints(const std::vector<Tq>& alpha_hat);
};

/* ======================================================================
 *  Transcript + base-case proof
 * ====================================================================*/

/// @brief Contains messages sent by the Prover to the Verifier in the base case of the Labrador protocol
struct BaseProverMessages {
  /// Ajtai commitment of (t,g)
  std::vector<Tq> u1;
  /// Nonce used by Prover of JL projection
  size_t JL_i;
  /// JL projection of the witness
  std::vector<Zq> p;
  /// Polynomials created during constant zero constraint aggregation
  std::vector<Tq> b_agg;
  /// Ajtai commitment of h
  std::vector<Tq> u2;

  BaseProverMessages() = default;

  size_t proof_size()
  {
    return sizeof(Zq) * (u1.size() * Tq::d + p.size() + b_agg.size() * Tq::d + u2.size() * Tq::d) + sizeof(size_t);
  }
};

struct PartialTranscript {
  /// Prover messages during the protocol
  BaseProverMessages prover_msg;

  /// hash evaluations
  std::vector<std::byte> seed1, seed2, seed3, seed4;

  /// Challenges- stored for convenience
  std::vector<Zq> psi, omega;
  std::vector<Tq> alpha_hat, challenges_hat;

  PartialTranscript() = default;

  inline size_t proof_size() { return prover_msg.proof_size(); }
};

/// @brief Struct to hold the proof for the base case
///
/// z_hat: is the vector computed in Step 29 of the base_case_prover
///
/// t: vector computed in Step 9 of the base_case_prover (T_tilde in the code)
///
/// g: vector computed in Step 9 of the base_case_prover (g_tilde in the code)
///
/// h: vector computed in Step 25 of the base_case_prover (H_tilde in the code)
///
/// @note constructor doesn't check dimensions
struct LabradorBaseCaseProof {
  std::vector<Tq> z_hat;
  std::vector<Rq> t, g, h;

  LabradorBaseCaseProof() = default;
  LabradorBaseCaseProof(
    const std::vector<Tq>& z_hat, const std::vector<Tq>& t, const std::vector<Tq>& g, const std::vector<Tq>& h)
      : z_hat(z_hat), t(t), g(g), h(h)
  {
  }
  LabradorBaseCaseProof(const LabradorBaseCaseProof& other) : z_hat(other.z_hat), t(other.t), g(other.g), h(other.h) {}
};
