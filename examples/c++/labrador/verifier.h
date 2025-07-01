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

  bool _verify_base_proof(const LabradorBaseCaseProof& base_proof) const;
  bool verify(const LabradorBaseCaseProof& base_proof);

  void agg_const_zero_constraints(size_t num_aggregation_rounds, const std::vector<Tq>& Q_hat);

  // internal
  void create_transcript();
};

// struct LabradorVerifier {
//   LabradorInstance lab_inst;
//   Oracle oracle;
//   std::vector<PartialTranscript> trs;
//   LabradorBaseCaseProof final_proof;
//   size_t NUM_REC;

//   LabradorVerifier(
//     const LabradorInstance& lab_inst,
//     const std::vector<BaseProverMessages>& prover_msg,
//     const LabradorBaseCaseProof& final_proof,
//     const std::byte* oracle_seed,
//     size_t oracle_seed_len,
//     size_t NUM_REC)
//       : lab_inst(lab_inst), final_proof(final_proof),
//         oracle(create_oracle_seed(oracle_seed, oracle_seed_len, lab_inst)), NUM_REC(NUM_REC)
//   {
//   }
// };
