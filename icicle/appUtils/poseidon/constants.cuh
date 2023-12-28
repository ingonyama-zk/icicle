#pragma once

#include "../../utils/error_handler.cuh"
#include "constants/constants_11.h"
#include "constants/constants_2.h"
#include "constants/constants_4.h"
#include "constants/constants_8.h"
#include <cassert>
#include <map>

const std::string UNSUPPORTED_ARITY_MESSAGE = "Unsupported arity";

uint32_t partial_rounds_number_from_arity(const uint32_t arity)
{
  switch (arity) {
  case 2:
    return 55;
  case 4:
    return 56;
  case 8:
    return 57;
  case 11:
    return 57;
  default:
    THROW_ICICLE_ERR(IcicleError_t::InvalidArgument, UNSUPPORTED_ARITY_MESSAGE);
  }
};

// TO-DO: change to mapping
const uint32_t FULL_ROUNDS_DEFAULT = 4;

// TO-DO: for now, the constants are only generated in bls12_381
template <typename S>
S* load_constants(const uint32_t arity)
{
  unsigned char* constants;
  switch (arity) {
  case 2:
    constants = constants_2;
    break;
  case 4:
    constants = constants_4;
    break;
  case 8:
    constants = constants_8;
    break;
  case 11:
    constants = constants_11;
    break;
  default:
    THROW_ICICLE_ERR(IcicleError_t::InvalidArgument, UNSUPPORTED_ARITY_MESSAGE);
  }
  return reinterpret_cast<S*>(constants);
}