#pragma once
#ifndef FIELD_CONFIG_H
#define FIELD_CONFIG_H

#include "gpu-utils/sharedmem.cuh"

#include "fields/field.cuh"
#if defined(EXT_FIELD)
#include "fields/extension_field.cuh"
#endif

/**
 * @namespace field_config
 * Namespace with type definitions for finite fields. Here, concrete types are created in accordance
 * with the `-DFIELD` env variable passed during build.
 */
namespace field_config {
  /**
   * Scalar field. Is always a prime field.
   */
  typedef Field<fp_config> scalar_t;
} // namespace field_config

#endif