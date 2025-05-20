#include "examples_utils.h"

// ICICLE runtime
#include "icicle/runtime.h"

#include "icicle/lattice/labrador.h" // For Zq, Rq, Tq, and the labrador APIs

using namespace icicle::labrador;

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

  // TODO need to use icicle_malloc() and icicle_copy() to allocate and copy that is device agnostic and
  // support GPU too. First step can be with hots memory and then we can add device support.

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