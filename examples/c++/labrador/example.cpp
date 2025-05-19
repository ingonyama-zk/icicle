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

// === TODO(Ash): Consider addinf protocol-specific types ===
// struct Statement { ... }
// struct Witness  { ... }
// struct Proof    { ... }

// === TODO(Ash): Implement protocol logic ===

eIcicleError setup(/*TODO params*/)
{
  // TODO Ash: labrador setup
  return eIcicleError::SUCCESS;
}

eIcicleError prove(/*TODO params*/)
{
  // TODO(Ash): Implement prover logic
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
  const size_t witness_size = 1 << 10;
  std::vector<Rq> S0(witness_size);
  std::vector<Rq> S1(witness_size);
  std::vector<Rq> S2(witness_size);

  auto randomize_Rq_vec = [](std::vector<Rq>& vec, int64_t max_value) {
    for (auto& x : vec) {
      for (size_t i = 0; i < Rq::d; ++i) {                // randomize each coefficient
        uint64_t val = rand_uint_32b() % (max_value + 1); // uniform in [0, sqrt_q]
        x.coeffs[i] = Zq::from(val);
      }
    }
  };

  // generate random values in [0, sqrt(q)]. We assume witness is low norm.
  const int64_t sqrt_q = static_cast<int64_t>(std::sqrt(q));
  randomize_Rq_vec(S0, sqrt_q);
  randomize_Rq_vec(S1, sqrt_q);
  randomize_Rq_vec(S2, sqrt_q);

  // === Call the protocol ===
  ICICLE_CHECK(setup(/* TODO(Ash): add arguments */));
  ICICLE_CHECK(prove(/* TODO(Ash): add arguments */));
  ICICLE_CHECK(verify(/* TODO(Ash): add arguments */));

  return 0;
}