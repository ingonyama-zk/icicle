#include "icicle/polynomials/polynomial_abstract_factory.h"
#include "icicle/dispatcher.h"
#include "icicle/fields/field_config.h"
using namespace field_config;

namespace icicle {

  ICICLE_OBJECT_DISPATCHER_INST(PolynomialFactory, polynomial_factory, AbstractPolynomialFactory<scalar_t>)

  template <typename S>
  std::shared_ptr<AbstractPolynomialFactory<S>> get_polynomial_abstract_factory(const S* phantom = nullptr);

  template <>
  std::shared_ptr<AbstractPolynomialFactory<scalar_t>>
  get_polynomial_abstract_factory<scalar_t>(const scalar_t* phantom)
  {
    return PolynomialFactory::get_factory();
  }

} // namespace icicle