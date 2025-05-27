#include "icicle/pqc/ml_kem.h"

#include "icicle/errors.h"
#include "icicle/dispatcher.h"
#include "icicle/backend/pqc/ml_kem_backend.h"

namespace icicle {
  namespace pqc {
    namespace ml_kem {

      //
      ICICLE_DISPATCHER_INST(MlKemKeygenDispatcher, ml_kem_keygen, MlKemKeygenImpl);
      ICICLE_DISPATCHER_INST(MlKemEncapsulateDispatcher, ml_kem_encapsulate, MlKemKeygenImpl);
      ICICLE_DISPATCHER_INST(MlKemDecapsulateDispatcher, ml_kem_decapsualte, MlKemDecapsulateImpl);

      // TODO implement the ml kem api with the dispatchers

    } // namespace ml_kem
  } // namespace pqc

} // namespace icicle