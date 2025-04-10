#define FIELD_ID BN254
#include "icicle/program/returning_value_program.h"
#include "icicle/hash/keccak.h"
#include "icicle/sumcheck/sumcheck.h"
#include "examples_utils.h"

using MlePoly = Symbol<scalar_t>;
MlePoly user_def_eq_times_a_b_minus_c_combine(const std::vector<MlePoly>& inputs)
{
  const MlePoly& A = inputs[0];
  const MlePoly& B = inputs[1];
  const MlePoly& C = inputs[2];
  const MlePoly& EQ = inputs[3];
  return EQ * (A * B - C);
}

int main(int argc, char* argv[])
{
  try_load_and_set_backend_device(argc, argv);
  std::cout << "\nIcicle Examples: Sumcheck with EQ * (A * B - C) combine function" << std::endl;

  int log_mle_poly_size = 13;
  int mle_poly_size = 1 << log_mle_poly_size;
  int nof_mle_poly = 4;

  std::cout << "\nGenerating input data" << std::endl;
  // generate inputs
  std::vector<scalar_t*> mle_polynomials(nof_mle_poly);
  for (int poly_i = 0; poly_i < nof_mle_poly; poly_i++) {
    mle_polynomials[poly_i] = new scalar_t[mle_poly_size];
    scalar_t::rand_host_many(mle_polynomials[poly_i], mle_poly_size);
  }

  std::cout << "Calculating sum" << std::endl;
  // calculate the claimed sum
  scalar_t claimed_sum = scalar_t::zero();
  for (int element_i = 0; element_i < mle_poly_size; element_i++) {
    const scalar_t a = mle_polynomials[0][element_i];
    const scalar_t b = mle_polynomials[1][element_i];
    const scalar_t c = mle_polynomials[2][element_i];
    const scalar_t eq = mle_polynomials[3][element_i];
    claimed_sum = claimed_sum + (a * b - c) * eq;
  }

  Hash hasher = create_sha3_256_hash();
  const char* domain_label = "ingonyama";
  const char* poly_label = "poly_label";
  const char* challenge_label = "icicle";
  scalar_t seed = scalar_t::from(18);
  bool little_endian = true;

  // create sumcheck
  auto prover_sumcheck = create_sumcheck<scalar_t>();
  // create the combine function to be the pre-defined function eq*(a*b-c)
  CombineFunction<scalar_t> combine_func_pre_def(EQ_X_AB_MINUS_C);
  const int nof_inputs = 4;
  CombineFunction<scalar_t> combine_func_user_def(user_def_eq_times_a_b_minus_c_combine, nof_inputs);

  CombineFunction<scalar_t> combine_funcs[2] = {combine_func_pre_def, combine_func_user_def};

  for (CombineFunction<scalar_t> combine_func : combine_funcs) {
    // create transcript_config for Fiat-Shamir
    SumcheckTranscriptConfig<scalar_t> prover_transcript_config(
      hasher, domain_label, poly_label, challenge_label, seed, little_endian);

    // create default sumcheck config
    SumcheckConfig sumcheck_config;
    // create empty sumcheck proof object which the prover will assign round polynomials into
    SumcheckProof<scalar_t> sumcheck_proof;

    std::cout << "\nCreating proof" << std::endl;
    // create the proof - Prover side
    prover_sumcheck.get_proof(
      mle_polynomials, mle_poly_size, claimed_sum, combine_func, std::move(prover_transcript_config), sumcheck_config,
      sumcheck_proof);

    // create sumcheck object for the Verifier
    auto verifier_sumcheck = create_sumcheck<scalar_t>();
    // create boolean variable for verification output
    bool verification_pass = false;

    std::cout << "Verifying proof" << std::endl;
    // verify the proof - Verifier side
    // NOTE: the transcript config should be identical for both Prover and Verifier
    SumcheckTranscriptConfig<scalar_t> verifier_transcript_config(
      hasher, domain_label, poly_label, challenge_label, seed, little_endian);
    verifier_sumcheck.verify(sumcheck_proof, claimed_sum, std::move(verifier_transcript_config), verification_pass);
    std::cout << "Verification result: " << (verification_pass ? "true" : "false") << std::endl;
  }
}