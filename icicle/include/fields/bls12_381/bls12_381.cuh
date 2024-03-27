#pragma once
#ifndef bls12_381_FIELDS_H
#define bls12_381_FIELDS_H

#include "bls12_381_params.cuh"
#include "fields/field.cuh"
#if defined(EXT_FIELD)
#include "fields/extension_field.cuh"
#endif

namespace bls12_381 {
  typedef Field<fp_config> scalar_t;
  typedef Field<fq_config> point_field_t;

#if defined(EXT_FIELD)
  typedef ExtensionField<fq_config> g2_point_field_t;
#endif

} // namespace bls12_381

#endif