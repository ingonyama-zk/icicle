#pragma once

#include "icicle/lattice/labrador.h"

#include "types.h"
#include "utils.h"
#include "shared.h"
#include "oracle.h"
#include "test_helpers.h"

struct LabradorBaseVerifier {
  LabradorInstance lab_inst;
  PartialTranscript trs;
  LabradorBaseCaseProof base_proof;
  Oracle oracle;

  LabradorBaseVerifier(
    const LabradorInstance& lab_inst,
    const BaseProverMessages& prover_msg,
    const LabradorBaseCaseProof& base_proof,
    const std::byte* oracle_seed,
    size_t oracle_seed_len)
      : lab_inst(lab_inst), trs(), base_proof(base_proof),
        oracle(create_oracle_seed(oracle_seed, oracle_seed_len, lab_inst))
  {
    trs.prover_msg = prover_msg;
    create_transcript();
  }

  bool _verify_base_proof() const;
  bool verify();

  void agg_const_zero_constraints(size_t num_aggregation_rounds, const std::vector<Tq>& Q_hat);

  // internal
  void create_transcript();
};

struct LabradorVerifier {
  LabradorInstance lab_inst;
};
