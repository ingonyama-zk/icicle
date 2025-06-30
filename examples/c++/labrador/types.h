#pragma once

#include "icicle/lattice/labrador.h"
#include "utils.h"
#include <cstddef>
#include <stdexcept>
#include <vector>

using namespace icicle::labrador;

/* ======================================================================
 *  Constraint descriptions
 * ====================================================================*/

struct EqualityInstance {
  size_t r;            // Number of witness vectors
  size_t n;            // Dimension of each vector in Tq
  std::vector<Tq> a;   // a[i,j]  – r×r  matrix over Tq
  std::vector<Tq> phi; // phi[i,j] – r vectors, each length n  (row-major)
  Tq b;                // Polynomial in Tq

  EqualityInstance(size_t r, size_t n) : r(r), n(n), a(r * r, zero()), phi(r * n, zero()), b(zero()) {}

  EqualityInstance(size_t r, size_t n, const std::vector<Tq>& a, const std::vector<Tq>& phi, Tq b)
      : r(r), n(n), a(a), phi(phi), b(b)
  {
    if (a.size() != r * r || phi.size() != r * n)
      throw std::invalid_argument("EqualityInstance: incorrect ‘a’ or ‘phi’ size");
  }

  EqualityInstance(const EqualityInstance& o) = default;
};

struct ConstZeroInstance {
  size_t r;            // Number of witness vectors
  size_t n;            // Dimension of each vector in Tq
  std::vector<Tq> a;   // a[i,j]  – r×r  matrix over Tq
  std::vector<Tq> phi; // phi[i,j] – r vectors, each length n  (row-major)
  Zq b;                // Such that \sum_ij a[i,j]<s[i], s[j]> + \sum_i <phi[i], s[i]> + b has 0 const coeff

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

struct LabradorParam {
  // Problem size
  size_t r; // number of witness vectors
  size_t n; // dimension of each vector in Tq

  // Seed for Ajtai matrix generation
  std::vector<std::byte> ajtai_seed;

  // Matrix dimensions for Ajtai commitments
  size_t kappa;  // Ajtai matrix A dimensions: n × kappa
  size_t kappa1; // Matrix B,C dimensions for committing to decomposed vectors (t,g)
  size_t kappa2; // Matrix D dimensions for committing to decomposed h vectors

  // Decomposition bases
  uint32_t base1; // Base for decomposing t
  uint32_t base2; // Base for decomposing g
  uint32_t base3; // Base for decomposing h

  // JL projection parameters
  size_t JL_out = 256; // Output dimension for Johnson-Lindenstrauss projection (typically 256)

  // Norm bounds
  double beta;                 // Witness norm bound
  uint64_t op_norm_bound = 15; // Operator norm bound for challenges

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
      : r(r), n(n), ajtai_seed(ajtai_seed), kappa(kappa), kappa1(kappa1), kappa2(kappa2), base1(base1), base2(base2),
        base3(base3), beta(beta)
  {
  }

  LabradorParam(const LabradorParam& o) = default;

  /* helper lengths for compressed vectors --------------------------------*/
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

struct LabradorInstance {
  LabradorParam param;                                   // LabradorParam for this instance
  std::vector<EqualityInstance> equality_constraints;    // K equality constraints
  std::vector<ConstZeroInstance> const_zero_constraints; // L const-zero constraints

  explicit LabradorInstance(const LabradorParam& p) : param(p) {}
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

  void agg_equality_constraints(const std::vector<Tq>& alpha_hat);
};

/* ======================================================================
 *  Transcript + base-case proof
 * ====================================================================*/

struct BaseProverMessages {
  // committed by the Prover
  std::vector<Tq> u1;
  size_t JL_i;
  std::vector<Zq> p;
  std::vector<Tq> b_agg;
  std::vector<Tq> u2;

  BaseProverMessages() = default;

  size_t proof_size()
  {
    return sizeof(Zq) * (u1.size() * Tq::d + p.size() + b_agg.size() * Tq::d + u2.size() * Tq::d) + sizeof(size_t);
  }
};

struct PartialTranscript {
  BaseProverMessages prover_msg;

  // hash evaluations
  std::vector<std::byte> seed1, seed2, seed3, seed4;

  // Challenges- stored for convenience
  std::vector<Zq> psi, omega;
  std::vector<Tq> alpha_hat, challenges_hat;

  PartialTranscript() = default;

  inline size_t proof_size() { return prover_msg.proof_size(); }
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
  std::vector<Rq> t, g, h;

  // TODO: Only for testing. Need to remove this
  EqualityInstance final_const;

  LabradorBaseCaseProof(size_t r, size_t n) : final_const(r, n), z_hat(), t(), g(), h() {};
  LabradorBaseCaseProof(
    EqualityInstance final_const, std::vector<Tq> z_hat, std::vector<Tq> t, std::vector<Tq> g, std::vector<Tq> h)
      : final_const(final_const), z_hat(z_hat), t(t), g(g), h(h)
  {
  }
};