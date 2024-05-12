#pragma once

#include "../../gpu-utils/device_context.cuh"
#include "../../fields/field_config.cuh"
#include "../polynomials.h"

using device_context::DeviceContext;

namespace polynomials {
  template <typename C = scalar_t, typename D = C, typename I = C>
  class CUDAPolynomialFactory : public AbstractPolynomialFactory<C, D, I>
  {
    std::vector<DeviceContext> m_device_contexts; // device-id --> device context
    std::vector<int> m_device_streams;   // device-id --> device stream. Storing the streams here as workaround
                                                  // since DeviceContext has a reference to a stream.

  public:
    CUDAPolynomialFactory();
    ~CUDAPolynomialFactory();
    std::shared_ptr<IPolynomialContext<C, D, I>> create_context() override;
    std::shared_ptr<IPolynomialBackend<C, D, I>> create_backend() override;
  };
} // namespace polynomials