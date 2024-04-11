#pragma once

#include "fields/field_config.cuh"
#include "polynomials/polynomials.h"
#include "polynomials/tracing/polynomial_tracing_backend.cuh"
#include "polynomials/tracing/memory_management.h"

namespace polynomials {

  template <typename C = field_config::scalar_t, typename D = C, typename I = C>
  class Optimizer
  {
  private:
  public:
    Optimizer() = default;
    void run(std::shared_ptr<TracingPolynomialContext<C, D, I>> context)
    {
      MemoryManagement mm{};
      mm.run(context);
    }
    void run(Polynomial<C, D, I>& p)
    {
      auto trace_ctxt = dynamic_cast<TracingPolynomialContext<C, D, I>*>(p.get_context());
      if (!trace_ctxt) {
        std::cerr << "[WARNING] Graph visualizer expecting TracingPolynomialContext. draw skipped.\n";
        return;
      }
      run(trace_ctxt->getptr());
    }
  };
} // namespace polynomials