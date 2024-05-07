#pragma once
#ifndef FIELD_CONFIG_H
#define FIELD_CONFIG_H

#include "id.h"
#include "field.cuh"

/**
 * @namespace field_config
 * Namespace with type definitions for finite fields. Here, concrete types are created in accordance
 * with the `-DFIELD` env variable passed during build.
 */
#if FIELD_ID == BN254
#include "snark_fields/bn254_scalar.cuh"
namespace field_config = bn254;
#elif FIELD_ID == BLS12_381
#include "snark_fields/bls12_381_scalar.cuh"
using bls12_381::fp_config;
namespace field_config = bls12_381;
#elif FIELD_ID == BLS12_377
#include "snark_fields/bls12_377_scalar.cuh"
namespace field_config = bls12_377;
#elif FIELD_ID == BW6_761
#include "snark_fields/bw6_761_scalar.cuh"
namespace field_config = bw6_761;
#elif FIELD_ID == GRUMPKIN
#include "snark_fields/grumpkin_scalar.cuh"
namespace field_config = grumpkin;

#elif FIELD_ID == BABY_BEAR
#include "stark_fields/babybear.cuh"
namespace field_config = babybear;
#elif FIELD_ID == STARK_252
#include "stark_fields/stark252.cuh"
namespace field_config = stark252;
#endif

#endif