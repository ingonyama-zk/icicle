#include <cstdint>
#include <iostream>
#include <random>
#define FIELD_ID BN254
#include "icicle/runtime.h"
#include "icicle/fields/field_config.h"
#include "icicle/utils/log.h"
#include "icicle/program/symbol.h"
#include "icicle/program/program.h"
#include "icicle/program/returning_value_program.h"
#include "icicle/hash/hash.h"
#include "icicle/hash/keccak.h"
#include "icicle/sumcheck/sumcheck.h"
#include "examples_utils.h"
#include "icicle/curves/params/bn254.h"
using namespace bn254;

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

  // set device
  Device dev = {"CPU", 0};
  icicle_set_device(dev);

  // create transcript_config for Fiat-Shamir
  SumcheckTranscriptConfig<scalar_t> transcript_config; // default configuration

  // create sumcheck
  auto prover_sumcheck = create_sumcheck<scalar_t>();
  // create the combine function to be the pre-defined function eq*(a*b-c)
  ReturningValueProgram<scalar_t> combine_func(EQ_X_AB_MINUS_C);
  // create default sumcheck config
  SumcheckConfig sumcheck_config;
  // create empty sumcheck proof object which the prover will asign round polynomials into
  SumcheckProof<scalar_t> sumcheck_proof;

  std::cout << "\nCreating proof" << std::endl;
  // create the proof - Prover side
  prover_sumcheck.get_proof(
    mle_polynomials, mle_poly_size, claimed_sum, combine_func, std::move(transcript_config), sumcheck_config,
    sumcheck_proof);

  // create sumcheck object for the Verifier
  auto verifier_sumcheck = create_sumcheck<scalar_t>();
  // create boolean variable for verification output
  bool verification_pass = false;

  std::cout << "Verifying proof" << std::endl;
  // verify the proof - Verifier side
  // NOTE: the transcript should be identical for both Prover and Verifier
  verifier_sumcheck.verify(sumcheck_proof, claimed_sum, std::move(transcript_config), verification_pass);
  std::cout << "Verification result: " << (verification_pass ? "true" : "false") << std::endl;
}