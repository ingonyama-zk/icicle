#include "examples_utils.h"

// ICICLE runtime
#include "icicle/runtime.h"

#include "icicle/lattice/labrador.h" // For Zq, Rq, Tq, and the labrador APIs
#include "icicle/hash/keccak.h"      // For Hash

using namespace icicle::labrador;

struct EqualityInstance {
  size_t r;                         // Number of witness vectors
  size_t n;                         // Dimension of each vector in Rq
  std::vector<std::vector<Tq>> a;   // a[i][j] matrix over Rq (r x r matrix)
  std::vector<std::vector<Tq>> phi; // phi[i] vector over Rq (r vectors, each of size n)
  Tq b;                             // Polynomial in Rq

  EqualityInstance(size_t r, size_t n) : r(r), n(n), a(r, std::vector<Tq>(r)), phi(r, std::vector<Tq>(n)), b() {}
  EqualityInstance(size_t r, size_t n, std::vector<std::vector<Tq>> a, std::vector<std::vector<Tq>> phi, Tq b)
      : r(r), n(n), a(std::move(a)), phi(std::move(phi)), b(std::move(b))
  {
    // check if the sizes of a and phi are correct
    if (a.size() != r || phi.size() != r) {
      throw std::invalid_argument("EqualityInstance: 'a' and 'phi' must have size r");
    }
    for (const auto& row : a) {
      if (row.size() != r) { throw std::invalid_argument("EqualityInstance: each row of 'a' must have size r"); }
    }
    for (const auto& vec : phi) {
      if (vec.size() != n) { throw std::invalid_argument("EqualityInstance: each vector in 'phi' must have size n"); }
    }
  }

  // Copy constructor
  EqualityInstance(const EqualityInstance& other) : r(other.r), n(other.n), a(other.a), phi(other.phi), b(other.b) {}
};

struct ConstZeroInstance {
  size_t r;                         // Number of witness vectors
  size_t n;                         // Dimension of each vector in Rq
  std::vector<std::vector<Tq>> a;   // a[i][j] matrix over Tq (r x r matrix)
  std::vector<std::vector<Tq>> phi; // phi[i] vector over Tq (r vectors, each of size n)
  Tq b;                             // Polynomial in Rq

  ConstZeroInstance(size_t r, size_t n) : r(r), n(n), a(r, std::vector<Tq>(r)), phi(r, std::vector<Tq>(n)), b() {}

  // Copy constructor
  ConstZeroInstance(const ConstZeroInstance& other) : r(other.r), n(other.n), a(other.a), phi(other.phi), b(other.b) {}
};

struct LabradorInstance {
  size_t r;                                              // Number of witness vectors
  size_t n;                                              // Dimension of each vector in Tq
  double beta;                                           // Norm bound
  std::vector<EqualityInstance> equality_constraints;    // K EqualityInstances
  std::vector<ConstZeroInstance> const_zero_constraints; // L ConstZeroInstances

  LabradorInstance(size_t r, size_t n, double beta) : r(r), n(n), beta(beta) {}

  // Copy constructor
  LabradorInstance(const LabradorInstance& other)
      : r(other.r), n(other.n), beta(other.beta), equality_constraints(other.equality_constraints),
        const_zero_constraints(other.const_zero_constraints)
  {
  }

  // Add an EqualityInstance
  void add_equality_constraint(const EqualityInstance& instance) { equality_constraints.push_back(instance); }

  // Add a ConstZeroInstance
  void add_const_zero_constraint(const ConstZeroInstance& instance) { const_zero_constraints.push_back(instance); }
};

struct LabradorRecursionRawInstance;
// Struct containing parameters for Labrador protocol
struct LabradorProtocol {
  // Matrix dimensions for Ajtai commitments
  const size_t kappa;  // Ajtai matrix A dimensions: n × kappa
  const size_t kappa1; // Matrix B,C dimensions for committing to decomposed vectors
  const size_t kappa2; // Matrix D dimensions for committing to h vectors

  // Decomposition bases
  const uint32_t base1; // Base for decomposing T
  const uint32_t base2; // Base for decomposing g
  const uint32_t base3; // Base for decomposing h
  const uint32_t base0; // Base for decomposing z in recursion

  // JL projection parameters
  const size_t JL_out; // Output dimension for Johnson-Lindenstrauss projection (typically 256)

  // Challenge space parameters
  const size_t challenge_num_1; // Number of ±1s (typically 31)
  const size_t challenge_num_2; // Number of ±2s (typically 10)

  // Norm bounds
  const double beta;            // Witness norm bound
  const uint64_t op_norm_bound; // Operator norm bound for challenges

  // Constructor with default values matching the base implementation
  LabradorProtocol()
      : kappa(1 << 4), kappa1(1 << 4), kappa2(1 << 4), base1(1 << 16), base2(1 << 16), base3(1 << 16), base0(1 << 16),
        JL_out(256), challenge_num_1(31), challenge_num_2(10), beta(10.0), op_norm_bound(15)
  {
  }

  // Method declarations
  LabradorRecursionRawInstance base_prover(
    LabradorInstance lab_inst,
    const std::vector<std::byte>& ajtai_seed,
    const std::vector<Rq>& S,
    std::vector<Zq>& proof);

  std::pair<LabradorInstance, std::vector<Rq>> prepare_recursive_problem(
    std::vector<std::byte> ajtai_seed, LabradorRecursionRawInstance raw_inst, size_t mu, size_t nu);
};

/// Encapsulates the problem and witness for the recursion instance
///
/// final_const: is the EqualityInstance prepared in Step 22
///
/// u1: is the commitment prepared in Step 10 of the base_prover
///
/// u2: is the commitment prepared in Step 26 of the base_prover
///
/// challenges_hat: are the polynomial challenges sent by the Verifier in Step 28
///
/// z_hat: is the vector computed in Step 29
///
/// t: vector computed in Step 9 (T_tilde in the code)
///
/// g: vector computed in Step 9 (g_tilde in the code)
///
/// h: vector computed in Step 25 (H_tilde in the code)
struct LabradorRecursionRawInstance {
  EqualityInstance final_const;
  std::vector<Tq> u1;
  std::vector<Tq> u2;
  std::vector<Tq> challenges_hat;
  std::vector<Tq> z_hat;
  std::vector<Rq> t;
  std::vector<Rq> g;
  std::vector<Rq> h;

  LabradorRecursionRawInstance(size_t r, size_t n)
      : final_const(r, n), u1(), u2(), challenges_hat(), z_hat(), t(), g(), h() {};
  LabradorRecursionRawInstance(
    EqualityInstance final_const,
    std::vector<Tq> u1,
    std::vector<Tq> u2,
    std::vector<Tq> challenges_hat,
    std::vector<Tq> z_hat,
    std::vector<Tq> t,
    std::vector<Tq> g,
    std::vector<Tq> h)
      : final_const(final_const), u1(std::move(u1)), u2(std::move(u2)), challenges_hat(std::move(challenges_hat)),
        z_hat(std::move(z_hat)), t(std::move(t)), g(std::move(g)), h(std::move(h))
  {
  }
};

Rq icicle::labrador::conjugate(const Rq& p)
{
  Rq result;
  // Copy constant term (index 0)
  result.coeffs[0] = p.coeffs[0];

  // For remaining coefficients, flip and negate
  for (size_t k = 1; k < Rq::d; k++) {
    // TODO: neg is negate?
    result.coeffs[k] = Zq::neg(p.coeffs[Rq::d - k]);
  }

  return result;
}

template <typename Zq>
int64_t get_q()
{
  constexpr auto q_storage = Zq::get_modulus();
  const int64_t q = *(int64_t*)&q_storage; // Note this is valid since TLC == 2
  return q;
}