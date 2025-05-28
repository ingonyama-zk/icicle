#pragma once

#include <functional>
#include "icicle/utils/utils.h"
#include "icicle/device.h"
#include "icicle/hash/skyscraper.h"
#include "icicle/fields/field_config.h"
using namespace field_config;

namespace icicle {

  struct SkyscraperConstantsOptions {
    unsigned int n = 1;
    unsigned int beta = 5;
    unsigned int s = 8;
  };

  using SkyscraperFactoryImpl = std::function<eIcicleError(
    const Device& device,
    unsigned n,
    unsigned beta,
    unsigned s,
    std::shared_ptr<HashBackend>& /*OUT*/)>;

  void register_skyscraper_factory(const std::string& deviceType, SkyscraperFactoryImpl impl);

#define REGISTER_SKYSCRAPER_FACTORY_BACKEND(DEVICE_TYPE, FUNC) \
  namespace { \
    static bool UNIQUE(_reg_skyscraper_factory) = []() -> bool { \
      register_skyscraper_factory(DEVICE_TYPE, FUNC); \
      return true; \
    }(); \
  }

} // namespace icicle 