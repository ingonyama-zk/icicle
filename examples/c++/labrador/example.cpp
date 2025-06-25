#include "icicle/lattice/labrador.h" // For Zq, Rq, Tq, and the labrador APIs
#include "icicle/hash/keccak.h"      // For Hash
#include "examples_utils.h"
#include "icicle/runtime.h"

#include "types.h"
#include "utils.h"
#include "prover.h"
#include "verifier.h"
#include "shared.h"
#include "test_helpers.h"

using namespace icicle::labrador;

// === Main driver ===

int main(int argc, char* argv[])
{
  ICICLE_LOG_INFO << "Labrador example";
  try_load_and_set_backend_device(argc, argv);

  const int64_t q = get_q<Zq>();

  // TODO use icicle_malloc() instead of std::vector. Consider a DeviceVector<T> that behaves like std::vector

  // randomize the witness Si with low norm
  const size_t n = 1 << 5;
  const size_t r = 1 << 3;
  constexpr size_t d = Rq::d;
  std::vector<Rq> S = rand_poly_vec(r * n, 1);
  EqualityInstance eq_inst = create_rand_eq_inst(n, r, S);
  assert(witness_legit_eq(eq_inst, S));
  ConstZeroInstance const_zero_inst = create_rand_const_zero_inst(n, r, S);
  assert(witness_legit_const_zero(const_zero_inst, S));

  // Use current time (milliseconds since epoch) as a unique Ajtai seed
  auto now = std::chrono::system_clock::now();
  auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
  std::string ajtai_seed_str = std::to_string(millis);
  std::cout << "Ajtai seed = " << ajtai_seed_str << std::endl;
  LabradorParam param{
    {reinterpret_cast<const std::byte*>(ajtai_seed_str.data()),
     reinterpret_cast<const std::byte*>(ajtai_seed_str.data()) + ajtai_seed_str.size()},
    1 << 4,    // kappa
    1 << 4,    // kappa1
    1 << 4,    // kappa2,
    1 << 16,   // base1
    1 << 16,   // base2
    1 << 16,   // base3
    n * r * d, // beta
  };
  LabradorInstance lab_inst{r, n, param};
  lab_inst.add_equality_constraint(eq_inst);
  lab_inst.add_const_zero_constraint(const_zero_inst);

  LabradorBaseProver base_prover{lab_inst, S};
  auto [base_proof, trs] = base_prover.base_case_prover();

  LabradorInstance verif_lab_inst{r, n, param};
  verif_lab_inst.add_equality_constraint(eq_inst);
  verif_lab_inst.add_const_zero_constraint(const_zero_inst);

  LabradorBaseVerifier base_verifier{verif_lab_inst, trs, base_proof};
  bool verification_result = base_verifier._verify_base_proof();

  if (verification_result) {
    std::cout << "Base proof verification passed\n";
  } else {
    std::cout << "Base proof verification failed\n";
  }

  std::cout << "Hello\n";
  return 0;
}