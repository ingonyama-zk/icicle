#include "examples_utils.h"

// ICICLE runtime
#include "icicle/runtime.h"

// Labrador ring types
#include "icicle/rings/integer_rings/labrador.h" // Zq, ZqRns, Rq, etc.

// ICICLE APIs
#include "icicle/ntt.h"                    // Number-theoretic transforms
#include "icicle/balanced_decomposition.h" // Balanced decomposition/recomposition
#include "icicle/jl_projection.h"          // Johnson–Lindenstrauss projection
#include "icicle/norm.h"                   // Norm computation and bounds
#include "icicle/vec_ops.h"                // Vector operations (add, mul, etc.)
// TODO(Ash): add more includes as needed, such as Matmul, etc.
// TODO (Ash): Define missing APIs here or in another header that is included here

// Type aliases
using Zq = labrador::scalar_t;
using ZqRns = labrador::scalar_rns_t;
using Rq = labrador::Rq;
using Tq = labrador::Tq;

// Parameters
constexpr size_t beta = 10; // TODO(Ash): set beta according to the protocol

// === TODO(Ash): Consider adding protocol-specific types ===
struct EqualityInstance {
  size_t r;                         // Number of witness vectors
  size_t n;                         // Dimension of each vector in Rq
  std::vector<std::vector<Rq>> a;   // a[i][j] matrix over Rq (r x r matrix)
  std::vector<std::vector<Rq>> phi; // phi[i] vector over Rq (r vectors, each of size n)
  Rq b;                             // Polynomial in Rq

  EqualityInstance(size_t r, size_t n) : r(r), n(n), a(r, std::vector<Rq>(r)), phi(r, std::vector<Rq>(n)), b() {}
};

struct ConstZeroInstance {
  size_t r;                         // Number of witness vectors
  size_t n;                         // Dimension of each vector in Rq
  std::vector<std::vector<Rq>> a;   // a[i][j] matrix over Rq (r x r matrix)
  std::vector<std::vector<Rq>> phi; // phi[i] vector over Rq (r vectors, each of size n)
  Rq b;                             // Polynomial in Rq

  ConstZeroInstance(size_t r, size_t n) : r(r), n(n), a(r, std::vector<Rq>(r)), phi(r, std::vector<Rq>(n)), b() {}
};

struct LabradorInstance {
  size_t r;                                              // Number of witness vectors
  size_t n;                                              // Dimension of each vector in Rq
  double beta;                                           // Norm bound
  std::vector<EqualityInstance> equality_constraints;    // K EqualityInstances
  std::vector<ConstZeroInstance> const_zero_constraints; // L ConstZeroInstances

  LabradorInstance(size_t r, size_t n, double beta) : r(r), n(n), beta(beta) {}

  // Add an EqualityInstance
  void add_equality_constraint(const EqualityInstance& instance) { equality_constraints.push_back(instance); }

  // Add a ConstZeroInstance
  void add_const_zero_constraint(const ConstZeroInstance& instance) { const_zero_constraints.push_back(instance); }
};

eIcicleError nega_cyc_NTT(const std::vector<Rq> input, std::vector<Rq> output) { return eIcicleError::SUCCESS; }

// === TODO(Ash): Implement protocol logic ===

eIcicleError setup(/*TODO params*/)
{
  // TODO Ash: labrador setup
  return eIcicleError::SUCCESS;
}

eIcicleError base_prover(const LabradorInstance LabInst, const std::vector<std::vector<Rq>> S, std::vector<Zq> proof)
{
  // Step 1: Pack the Witnesses into a Matrix S

  const size_t r = LabInst.r; // Number of witness vectors
  const size_t n = LabInst.n; // Dimension of witness vectors

  // Ensure S is of the correct size
  if (S.size() != r) { return eIcicleError::INVALID_ARGUMENT; }
  for (const auto& row : S) {
    if (row.size() != n) { return eIcicleError::INVALID_ARGUMENT; }
  }

  // Step 2: Convert S to the NTT Domain
  std::vector<std::vector<Rq>> S_hat(r, std::vector<Rq>(n));

  for (size_t i = 0; i < r; ++i) {
    // Perform negacyclic NTT on the i-th row
    eIcicleError err = nega_cyc_NTT(S[i], S_hat[i]);
    if (err != eIcicleError::SUCCESS) {
      return err; // Propagate any errors from NTT
    }
  }

  // (S_hat is now ready for use in the subsequent steps)

  return eIcicleError::SUCCESS;
}

eIcicleError verify(/*TODO params*/)
{
  // TODO Ash: labrador verifier
  return eIcicleError::SUCCESS;
}

template <typename Zq>
int64_t get_q()
{
  constexpr auto q_storage = Zq::get_modulus();
  const int64_t q = *(int64_t*)&q_storage; // Note this is valid since TLC == 2
  return q;
}

// === Main driver ===

int main(int argc, char* argv[])
{
  try_load_and_set_backend_device(argc, argv);

  int64_t q = get_q<Zq>();

  // randomize the witness Si with low norm
  // TODO Ash: maybe want to allocate them consecutive in memory
  const size_t n = 1 << 8;
  const size_t r = 1 << 8;
  std::vector<std::vector<Rq>> S(r, std::vector<Rq>(n));

  // TODO eventually we will use icicle_malloc() and icicle_copy() to allocate and copy that is device agnostic and
  // support GPU too. First step can be with host memory and then we can add device support.

  auto randomize_Rq_vec = [](std::vector<Rq>& vec, int64_t max_value) {
    for (auto& x : vec) {
      for (size_t i = 0; i < Rq::d; ++i) {                // randomize each coefficient
        uint64_t val = rand_uint_32b() % (max_value + 1); // uniform in [0, sqrt_q]
        x.coeffs[i] = Zq::from(val);
      }
    }
  };

  // std::cout << "0= " << Zq::from(0) << std::endl
  //           << "1= " << Zq::from(1) << std::endl
  //           << "31= " << Zq::from(31) << std::endl;

  // generate random values in [0, sqrt(q)]. We assume witness is low norm.
  const int64_t sqrt_q = static_cast<int64_t>(std::sqrt(q));
  for (size_t i = 0; i < r; ++i) {
    randomize_Rq_vec(S[i], sqrt_q);
  }

  // === Call the protocol ===
  // ICICLE_CHECK(setup(/* TODO(Ash): add arguments */));
  // ICICLE_CHECK(prove(/* TODO(Ash): add arguments */));
  // ICICLE_CHECK(verify(/* TODO(Ash): add arguments */));

  std::cout << "Hello\n";
  return 0;
}