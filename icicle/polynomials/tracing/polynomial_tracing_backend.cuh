#pragma once

#include "curves/curve_config.cuh"
#include "polynomials/polynomials.h"

using curve_config::scalar_t;

namespace polynomials {
  template <typename C = scalar_t, typename D = C, typename I = C>
  class TracingPolynomialFactory : public AbstractPolynomialFactory<C, D, I>
  {
  private:
    std::shared_ptr<AbstractPolynomialFactory<C, D, I>> m_base_factory; // decorator pattern

  public:
    TracingPolynomialFactory(std::shared_ptr<AbstractPolynomialFactory<C, D, I>> base_factory);
    ~TracingPolynomialFactory() = default;
    std::shared_ptr<IPolynomialContext<C, D, I>> create_context() override;
    std::shared_ptr<IPolynomialBackend<C, D, I>> create_backend() override;
  };
} // namespace polynomials