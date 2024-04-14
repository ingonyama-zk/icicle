#pragma once
#ifndef BW6_761_SCALAR_PARAMS_H
#define BW6_761_SCALAR_PARAMS_H

#include "fields/storage.cuh"
#include "fields/field.cuh"
#include "fields/snark_fields/bls12_377_base.cuh"

namespace bw6_761 {
  typedef bls12_377::fq_config fp_config;

  /**
   * Scalar field. Is always a prime field.
   */
  typedef Field<fp_config> scalar_t;

#ifdef EXT_FIELD
  /**
   * Extension field of `scalar_t` enabled if `-DEXT_FIELD` env variable is.
   */
  typedef ExtensionField<fp_config> extension_t;
#endif
} // namespace bw6_761

#endif