#include "labrador.h"           // For Zq, Rq, Tq, and the APIs
#include "icicle/hash/keccak.h" // For Hash
#include "examples_utils.h"
#include "icicle/runtime.h"

#include "types.h"
#include "utils.h"
#include "prover.h"
#include "verifier.h"
#include "shared.h"
#include "test_helpers.h"
#include "benchmarking.h"

#include <iostream>
#include <chrono>
#include <vector>
#include <tuple>
#include <iomanip>
#include <fstream>

using namespace icicle::labrador;

void prover_verifier_trace()
{
  const int64_t q = get_q<Zq>();

  // randomize the witness Si with low norm
  const size_t n = 1 << 9;
  const size_t r = 1 << 5;
  constexpr size_t d = Rq::d;
  const size_t max_value = 2;
  size_t num_eq_const = 10;
  size_t num_cz_const = 10;

  const std::vector<Rq> S = rand_poly_vec(r * n, max_value);
  auto eq_inst = create_rand_eq_inst(n, r, S, num_eq_const);
  std::cout << "Created Eq constraints\n";
  auto const_zero_inst = create_rand_const_zero_inst(n, r, S, num_cz_const);
  std::cout << "Created Cz constraints\n";

  // Use current time (milliseconds since epoch) as a unique Ajtai seed
  auto now = std::chrono::system_clock::now();
  auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
  std::string ajtai_seed_str = std::to_string(millis);
  std::cout << "Ajtai seed = " << ajtai_seed_str << std::endl;

  double beta = sqrt(max_value * n * r * d);
  uint32_t base0 = calc_base0(r, OP_NORM_BOUND, beta);
  LabradorParam param{
    r,
    n,
    {reinterpret_cast<const std::byte*>(ajtai_seed_str.data()),
     reinterpret_cast<const std::byte*>(ajtai_seed_str.data()) + ajtai_seed_str.size()},
    secure_msis_rank(), // kappa
    secure_msis_rank(), // kappa1
    secure_msis_rank(), // kappa2,
    base0,              // base1
    base0,              // base2
    base0,              // base3
    beta,               // beta
  };
  LabradorInstance lab_inst{param};
  lab_inst.add_equality_constraint(eq_inst);
  lab_inst.add_const_zero_constraint(const_zero_inst);

  std::string oracle_seed = "ORACLE_SEED";

  size_t NUM_REC = 3;
  LabradorProver prover{
    lab_inst, S, reinterpret_cast<const std::byte*>(oracle_seed.data()), oracle_seed.size(), NUM_REC};

  std::cout << "Problem param: n,r = " << n << ", " << r << "\n";
  std::cout << "CONSISTENCY_CHECKS = " << (CONSISTENCY_CHECKS ? "true" : "false") << "\n";
  if (CONSISTENCY_CHECKS) { std::cout << "CONSISTENCY_CHECKS TRUE IMPLIES TIMING ESTIMATES ARE INCORRECT\n"; }
  auto [trs, final_proof] = prover.prove();

  // extract all prover_msg from trs vector into a vector prover_msgs
  std::vector<BaseProverMessages> prover_msgs;
  for (const auto& transcript : trs) {
    prover_msgs.push_back(transcript.prover_msg);
  }
  LabradorVerifier verifier{lab_inst,           prover_msgs,
                            final_proof,        reinterpret_cast<const std::byte*>(oracle_seed.data()),
                            oracle_seed.size(), NUM_REC};

  std::cout << "Verification result: \n";
  if (verifier.verify()) {
    std::cout << "Verification passed. \n";
  } else {
    std::cout << "Verification failed. \n";
  }
}

// === Main driver ===

int main(int argc, char* argv[])
{
  ICICLE_LOG_INFO << "Labrador example";
  try_load_and_set_backend_device(argc, argv);

  // I. Use the following code for examining program trace:
  // prover_verifier_trace();

  // II. To run benchmark uncomment:

  std::vector<std::tuple<size_t, size_t>> arr_nr{{1 << 6, 1 << 3}};
  std::vector<std::tuple<size_t, size_t>> num_constraint{{10, 10}};
  size_t NUM_REP = 1;
  bool SKIP_VERIF = false;
  benchmark_program(arr_nr, num_constraint, NUM_REP, SKIP_VERIF);

  return 0;
}