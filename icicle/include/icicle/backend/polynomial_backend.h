#pragma once

#include "icicle/polynomials/polynomials.h"
#include "icicle/polynomials/polynomial_abstract_factory.h"
#include "icicle/fields/field_config.h"
using namespace field_config;

namespace icicle {

  /*************************** Backend registration ***************************/

  void register_polynomial_factory(
    const std::string& deviceType, std::shared_ptr<AbstractPolynomialFactory<scalar_t>> factory);

#define REGISTER_SCALAR_POLYNOMIAL_FACTORY_BACKEND(DEVICE_TYPE, FACTORY)                                               \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_polynomial_factory) = []() -> bool {                                                       \
      register_polynomial_factory(DEVICE_TYPE, std::make_shared<FACTORY>());                                           \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  // External template instantiation to ensure the template is compiled for specific types.
  extern template class Polynomial<scalar_t>;

} // namespace icicle