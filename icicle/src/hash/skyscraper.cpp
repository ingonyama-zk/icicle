#include "icicle/errors.h"
#include "icicle/backend/hash/skyscraper_backend.h"
#include "icicle/dispatcher.h"

namespace icicle {

  // Skyscraper
  ICICLE_DISPATCHER_INST(SkyscraperHasherDispatcher, skyscraper_factory, SkyscraperFactoryImpl);

  Hash create_skyscraper_hash(unsigned n, unsigned beta, unsigned s)
  {
    std::shared_ptr<HashBackend> backend;
    ICICLE_CHECK(SkyscraperHasherDispatcher::execute(n, beta, s, backend));
    Hash skyscraper{backend};
    return skyscraper;
  }

} // namespace icicle 