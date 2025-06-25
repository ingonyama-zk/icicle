#pragma once

#include "icicle/lattice/labrador.h"

#include "types.h"
#include "utils.h"
#include "shared.h"
#include "test_helpers.h"

struct LabradorBaseVerifier {
  LabradorInstance lab_inst;
  PartialTranscript trs;
  LabradorBaseCaseProof base_proof;

  LabradorBaseVerifier(
    const LabradorInstance& lab_inst, const PartialTranscript& trs, const LabradorBaseCaseProof& base_proof)
      : lab_inst(lab_inst), trs(trs), base_proof(base_proof)
  {
  }
  bool _verify_base_proof() const;
  bool verify() const;
};

struct LabradorVerifier {
  LabradorInstance lab_inst;
};
