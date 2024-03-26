#pragma once
#ifndef BN254_FIELDS_H
#define BN254_FIELDS_H

#define FIELD bn254

#include "bn254_params.cuh"
#include "fields/field.cuh"
#if defined(EXT_DEFINED)
#include "fields/extension_field.cuh"
#endif

namespace bn254 {
  typedef Field<fp_config> scalar_t;
  typedef Field<fq_config> point_field_t;

#if defined(EXT_DEFINED)
  typedef ExtensionField<fq_config> g2_point_field_t;
#endif

} // namespace bn254

#endif