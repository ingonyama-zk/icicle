#pragma once
#ifndef GRUMPKIN_FIELDS_PARAMS_H
#define GRUMPKIN_FIELDS_PARAMS_H

#include "fields/bn254/bn254_params.cuh"

namespace grumpkin {
  typedef bn254::fq_config fp_config;
  typedef bn254::fp_config fq_config;
} // namespace grumpkin

#endif
