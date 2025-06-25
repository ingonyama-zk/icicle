#pragma once

#include "icicle/lattice/labrador.h"

using namespace icicle::labrador;

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
