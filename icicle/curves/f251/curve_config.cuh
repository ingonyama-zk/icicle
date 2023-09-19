#pragma once

#include "../../primitives/field.cuh"

#include "params.cuh"

namespace F251 {
  typedef Field<PARAMS_F251::fp_config> scalar_field_t;
  typedef scalar_field_t scalar_t;
}