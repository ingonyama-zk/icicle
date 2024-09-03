#pragma once
#ifndef BW6_761_SCALAR_PARAMS_H
#define BW6_761_SCALAR_PARAMS_H

#include "icicle/fields/storage.h"
#include "icicle/fields/field.h"
#include "icicle/fields/snark_fields/bls12_377_base.h"

namespace bw6_761 {
  typedef bls12_377::fq_config fp_config;

  /**
   * Scalar field. Is always a prime field.
   */
  typedef Field<fp_config> scalar_t;
} // namespace bw6_761

#endif