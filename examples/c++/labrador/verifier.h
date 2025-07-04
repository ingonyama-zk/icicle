#pragma once

#include "labrador.h"

#include "types.h"
#include "utils.h"
#include "shared.h"
#include "oracle.h"
#include "test_helpers.h"

struct LabradorBaseVerifier {
  LabradorInstance lab_inst;
  PartialTranscript trs;
  Oracle oracle;

  LabradorBaseVerifier(
    const LabradorInstance& lab_inst,
    const BaseProverMessages& prover_msg,
    const std::byte* oracle_seed,
    size_t oracle_seed_len)
      : lab_inst(lab_inst), trs(), oracle(create_oracle_seed(oracle_seed, oracle_seed_len, lab_inst))
  {
    trs.prover_msg = prover_msg;
    create_transcript();
  }

  LabradorBaseVerifier(const LabradorInstance& lab_inst, const BaseProverMessages& prover_msg, const Oracle& oracle)
      : lab_inst(lab_inst), trs(), oracle(oracle)
  {
    trs.prover_msg = prover_msg;
    create_transcript();
  }

  bool _verify_base_proof(const LabradorBaseCaseProof& base_proof) const;
  bool part_verify();
  bool fully_verify(const LabradorBaseCaseProof& base_proof);

  void agg_const_zero_constraints(size_t num_aggregation_rounds);

  // internal
  void create_transcript();
};

struct LabradorVerifier {
  LabradorInstance lab_inst;
  Oracle oracle;
  const std::vector<BaseProverMessages> prover_msgs;
  LabradorBaseCaseProof final_proof;
  size_t NUM_REC;

  LabradorVerifier(
    const LabradorInstance& lab_inst,
    const std::vector<BaseProverMessages>& prover_msgs,
    const LabradorBaseCaseProof& final_proof,
    const std::byte* oracle_seed,
    size_t oracle_seed_len,
    size_t NUM_REC)
      : lab_inst(lab_inst), prover_msgs(prover_msgs), final_proof(final_proof),
        oracle(create_oracle_seed(oracle_seed, oracle_seed_len, lab_inst)), NUM_REC(NUM_REC)
  {
    if (prover_msgs.size() != NUM_REC) { throw std::invalid_argument("prover_msgs.size() must equal NUM_REC"); }
  }

  bool verify();
};
