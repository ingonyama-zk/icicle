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
  const size_t n = 1 << 4;
  const size_t r = 1 << 2;
  constexpr size_t d = Rq::d;
  const std::vector<Rq> S = rand_poly_vec(r * n, 2); // S = 2^12 Zq elements
  EqualityInstance eq_inst = create_rand_eq_inst(n, r, S);
  assert(witness_legit_eq(eq_inst, S));
  ConstZeroInstance const_zero_inst = create_rand_const_zero_inst(n, r, S);
  assert(witness_legit_const_zero(const_zero_inst, S));
  ConstZeroInstance const_zero_inst2 = create_rand_const_zero_inst(n, r, S);
  assert(witness_legit_const_zero(const_zero_inst2, S));

  // Use current time (milliseconds since epoch) as a unique Ajtai seed
  auto now = std::chrono::system_clock::now();
  auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
  std::string ajtai_seed_str = std::to_string(millis);
  std::cout << "Ajtai seed = " << ajtai_seed_str << std::endl;
  LabradorParam param{
    r,
    n,
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
  LabradorInstance lab_inst{param};
  lab_inst.add_equality_constraint(eq_inst);
  lab_inst.add_const_zero_constraint(const_zero_inst);
  lab_inst.add_const_zero_constraint(const_zero_inst2);

  std::string oracle_seed = "ORACLE_SEED";

  // LabradorBaseProver base_prover{
  //   lab_inst, S, reinterpret_cast<const std::byte*>(oracle_seed.data()), oracle_seed.size()};

  // auto [base_proof, trs] = base_prover.base_case_prover();

  // LabradorInstance verif_lab_inst{param};
  // verif_lab_inst.add_equality_constraint(eq_inst);
  // verif_lab_inst.add_const_zero_constraint(const_zero_inst);
  // verif_lab_inst.add_const_zero_constraint(const_zero_inst2);
  // // to break verification:
  // // verif_lab_inst.const_zero_constraints[0].b = verif_lab_inst.const_zero_constraints[0].b + Zq::one();
  // LabradorBaseVerifier base_verifier{
  //   verif_lab_inst, trs.prover_msg, reinterpret_cast<const std::byte*>(oracle_seed.data()), oracle_seed.size()};

  // // // Assert that Verifier trs and Prover trs are equal
  // // auto bytes_eq = [](const std::vector<std::byte>& a, const std::vector<std::byte>& b) { return a == b; };
  // // auto zq_vec_eq = [](const std::vector<Zq>& a, const std::vector<Zq>& b) {
  // //   if (a.size() != b.size()) return false;
  // //   for (size_t i = 0; i < a.size(); ++i)
  // //     if (a[i] != b[i]) return false;
  // //   return true;
  // // };

  // // const auto& trs_P = trs;               // Prover's transcript
  // // const auto& trs_V = base_verifier.trs; // Verifier's transcript

  // // auto report = [&](bool ok, const std::string& name) {
  // //   if (!ok) std::cout << "  • " << name << " mismatch\n";
  // //   return ok;
  // // };

  // // bool ok = true;
  // // ok &= report(bytes_eq(trs_P.seed1, trs_V.seed1), "seed1");
  // // ok &= report(bytes_eq(trs_P.seed2, trs_V.seed2), "seed2");
  // // ok &= report(bytes_eq(trs_P.seed3, trs_V.seed3), "seed3");
  // // ok &= report(bytes_eq(trs_P.seed4, trs_V.seed4), "seed4");

  // // ok &= report(zq_vec_eq(trs_P.psi, trs_V.psi), "psi");
  // // ok &= report(zq_vec_eq(trs_P.omega, trs_V.omega), "omega");

  // // ok &= report(poly_vec_eq(trs_P.alpha_hat.data(), trs_V.alpha_hat.data(), trs_P.alpha_hat.size()), "alpha_hat");

  // // ok &= report(
  // //   poly_vec_eq(trs_P.challenges_hat.data(), trs_V.challenges_hat.data(), trs_P.challenges_hat.size()),
  // //   "challenges_hat");

  // // if (!ok) {
  // //   std::cerr << "\nTranscript mismatch detected above.\n";
  // //   return 1;
  // // }
  // // std::cout << "Transcript check passed ✅\n";

  // bool verification_result = base_verifier.fully_verify(base_proof);

  // if (verification_result) {
  //   std::cout << "Base proof verification passed\n";
  // } else {
  //   std::cout << "Base proof verification failed\n";
  // }

  // std::cout << "Beginning recursion... \n";
  // uint32_t base0 = 1 << 3;
  // size_t mu = 1 << 3, nu = 1 << 3;
  // LabradorInstance rec_inst = prepare_recursion_instance(
  //   param,                                        // prev_param
  //   base_prover.lab_inst.equality_constraints[0], // final_const,
  //   trs, base0, mu, nu);
  // LabradorProver dummy_prover{
  //   lab_inst, S, reinterpret_cast<const std::byte*>(oracle_seed.data()), oracle_seed.size(), 1};
  // std::vector<Rq> rec_S = dummy_prover.prepare_recursion_witness(base_proof, base0, mu, nu);
  // std::cout << "rec_inst.r = " << rec_inst.param.r << std::endl;
  // std::cout << "rec_inst.n = " << rec_inst.param.n << std::endl;
  // std::cout << "Num rec_inst.equality_constraints = " << rec_inst.equality_constraints.size() << std::endl;
  // std::cout << "Num rec_inst.const_zero_constraints = " << rec_inst.const_zero_constraints.size() << std::endl;

  // std::cout << "\tTesting rec-witness validity...";
  // assert(lab_witness_legit(rec_inst, rec_S));
  // std::cout << "VALID\n";

  size_t NUM_REC = 1;
  LabradorProver prover{
    lab_inst, S, reinterpret_cast<const std::byte*>(oracle_seed.data()), oracle_seed.size(), NUM_REC};

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
  std::cout << "Hello\n";
  return 0;
}