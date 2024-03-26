#pragma once
#ifndef GRUMPKIN_FIELDS_H
#define GRUMPKIN_FIELDS_H

#include "grumpkin_params.cuh"
#include "fields/bn254/bn254.cuh"
#include "fields/field.cuh"

namespace grumpkin {
  typedef bn254::scalar_t point_field_t;
  typedef bn254::point_field_t scalar_t;
} // namespace grumpkin

#endif