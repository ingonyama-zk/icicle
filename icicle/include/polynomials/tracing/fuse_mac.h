#pragma once

#include "fields/field_config.cuh"
#include "polynomials/polynomials.h"
#include "polynomials/tracing/polynomial_tracing_backend.cuh"
#include "polynomials/tracing/pass.h"

namespace polynomials {

  template <typename C = field_config::scalar_t, typename D = C, typename I = C>
  class FuseMac : public Pass<C, D, I>
  {
  private:
    void visit(std::shared_ptr<TracingPolynomialContext<C, D, I>>& context);

  public:
    FuseMac() = default;
    void run(std::shared_ptr<TracingPolynomialContext<C, D, I>>& context) override;
  };

  extern template class FuseMac<>;

} // namespace polynomials