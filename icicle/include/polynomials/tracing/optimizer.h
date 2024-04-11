#pragma once

#include "fields/field_config.cuh"
#include "polynomials/polynomials.h"
#include "polynomials/tracing/polynomial_tracing_backend.cuh"
#include "polynomials/tracing/pass.h"
#include "polynomials/tracing/memory_management.h"
#include "polynomials/tracing/fuse_mac.h"

#include <list>
#include <memory>

namespace polynomials {

  template <typename C = field_config::scalar_t, typename D = C, typename I = C>
  class Optimizer
  {
  private:
    std::list<std::unique_ptr<Pass<C, D, I>>> m_passes;

  public:
    Optimizer()
    {
      m_passes.push_back(std::move(std::make_unique<FuseMac<C, D, I>>()));
      m_passes.push_back(std::move(std::make_unique<MemoryManagement<C, D, I>>()));
    }
    void run(std::shared_ptr<TracingPolynomialContext<C, D, I>> context)
    {
      for (auto& pass : m_passes) {
        pass->run(context);
      }
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