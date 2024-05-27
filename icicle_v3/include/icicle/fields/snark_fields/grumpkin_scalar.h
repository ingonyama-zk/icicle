#pragma once
#ifndef GRUMPKIN_SCALAR_PARAMS_H
#define GRUMPKIN_SCALAR_PARAMS_H

#include "fields/storage.h"
#include "fields/field.h"
#include "fields/snark_fields/bn254_base.h"

namespace grumpkin {
  typedef bn254::fq_config fp_config;

  /**
   * Scalar field. Is always a prime field.
   */
  typedef Field<fp_config> scalar_t;
} // namespace grumpkin

#endif