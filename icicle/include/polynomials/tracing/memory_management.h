#pragma once

#include "fields/field_config.cuh"
#include "polynomials/polynomials.h"
#include "polynomials/tracing/polynomial_tracing_backend.cuh"
#include "polynomials/tracing/pass.h"

// TODO Yuval: maybe better do it as part of the Interpreter and not a pass

namespace polynomials {

  template <typename C = field_config::scalar_t, typename D = C, typename I = C>
  class MemoryManagement : public Pass<C, D, I>
  {
  private:
    void visit(std::shared_ptr<TracingPolynomialContext<C, D, I>>& context);
    bool is_compatible(std::shared_ptr<TracingPolynomialContext<C, D, I>>& node, uint64_t min_size = 0);

  public:
    MemoryManagement() = default;
    void run(std::shared_ptr<TracingPolynomialContext<C, D, I>>& context) override;
  };

  extern template class MemoryManagement<>;

} // namespace polynomials