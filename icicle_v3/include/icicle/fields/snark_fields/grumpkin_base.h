#pragma once
#ifndef GRUMPKIN_BASE_PARAMS_H
#define GRUMPKIN_BASE_PARAMS_H

#include "icicle/fields/storage.h"
#include "icicle/fields/snark_fields/bn254_scalar.h"

namespace grumpkin {
  typedef bn254::fp_config fq_config;
}

#endif