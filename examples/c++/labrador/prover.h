#pragma once

#include "icicle/lattice/labrador.h"

#include "types.h"
#include "utils.h"
#include "shared.h"
#include "test_helpers.h"

struct LabradorBaseProver {
  LabradorInstance lab_inst;
  // S consists of r vectors of dim n arranged in row major order
  std::vector<Rq> S;

  LabradorBaseProver(const LabradorInstance& lab_inst, const std::vector<Rq>& S) : lab_inst(lab_inst), S(S)
  {
    if (S.size() != lab_inst.param.r * lab_inst.param.n) { throw std::invalid_argument("S must have size r * n"); }
  }

  std::vector<Tq> agg_const_zero_constraints(
    size_t num_aggregation_rounds,
    size_t JL_out,
    const std::vector<Tq>& S_hat,
    const std::vector<Tq>& g_hat,
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

  std::vector<Rq> prepare_recursion_witness(
    const PartialTranscript& trs, const LabradorBaseCaseProof& pf, uint32_t base0, size_t mu, size_t nu);

  std::pair<std::vector<PartialTranscript>, LabradorBaseCaseProof> prove();
};