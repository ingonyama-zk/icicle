#pragma once
#ifndef BW6_761_FIELDS_H
#define BW6_761_FIELDS_H

#include "bw6_761_params.cuh"
#include "fields/bls12_377/bls12_377.cuh"
#include "fields/field.cuh"
#if defined(EXT_FIELD)
#include "fields/extension_field.cuh"
#endif

namespace bw6_761 {
  typedef bls12_377::point_field_t scalar_t;
  typedef Field<fq_config> point_field_t;

#if defined(EXT_FIELD)
  typedef ExtensionField<fq_config> g2_point_field_t;
#endif

} // namespace bw6_761

#endif