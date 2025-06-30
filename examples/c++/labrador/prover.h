#pragma once

#include "icicle/lattice/labrador.h"

#include "types.h"
#include "utils.h"
#include "shared.h"
#include "oracle.h"
#include "test_helpers.h"

struct LabradorBaseProver {
  LabradorInstance lab_inst;
  // S consists of r vectors of dim n arranged in row major order
  std::vector<Rq> S;
  Oracle oracle;

  // Constructs a prover with an optional external seed for the random oracle.
  // The oracle is initialised with (oracle_seed || bytes(lab_inst)).
  LabradorBaseProver(
    const LabradorInstance& lab_inst, const std::vector<Rq>& S, const std::byte* oracle_seed, size_t oracle_seed_len)
      : lab_inst(lab_inst), S(S), oracle(create_oracle_seed(oracle_seed, oracle_seed_len, lab_inst))
  {
    if (S.size() != lab_inst.param.r * lab_inst.param.n) { throw std::invalid_argument("S must have size r * n"); }
  }

  std::vector<Tq> agg_const_zero_constraints(
    const std::vector<Tq>& S_hat,
    const std::vector<Tq>& G_hat,
    const std::vector<Zq>& p,
    const std::vector<Tq>& Q_hat,
    const std::vector<Zq>& psi,
    const std::vector<Zq>& omega);

  std::pair<size_t, std::vector<Zq>> select_valid_jl_proj(std::byte* seed, size_t seed_len) const;

  std::pair<LabradorBaseCaseProof, PartialTranscript> base_case_prover();
};

struct LabradorProver {
  LabradorInstance lab_inst;
  // S consists of r vectors of dim n arranged in row major order
  std::vector<Rq> S;
  const size_t NUM_REC;

  LabradorProver(const LabradorInstance& lab_inst, const std::vector<Rq>& S, size_t NUM_REC)
      : lab_inst(lab_inst), S(S), NUM_REC(NUM_REC)
  {
  }

  std::vector<Rq> prepare_recursion_witness(const LabradorBaseCaseProof& pf, uint32_t base0, size_t mu, size_t nu);

  std::pair<std::vector<PartialTranscript>, LabradorBaseCaseProof> prove();
};