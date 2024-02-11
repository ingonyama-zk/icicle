// #include <cuda_runtime.h>
// #include "../../curves/include/types.h"

#ifndef _NTT_H
#define _NTT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum class Ordering { kNN, kNR, kRN, kRR };

  // struct DeviceContext {
  //   cudaStream_t& stream;  /**< Stream to use. Default value: 0. */
  //   size_t device_id; /**< Index of the currently used GPU. Default value: 0. */
  //   cudaMemPool_t mempool; /**< Mempool to use. Default value: 0. */
  // };

// typedef struct NTTConfig {
//   DeviceContext ctx;
//   scalar_t coset_gen;
//   int batch_size;
//   Ordering ordering;
//   bool are_inputs_on_device;
//   bool are_outputs_on_device;
//   bool is_async;
// }

#ifdef __cplusplus
}
#endif

#endif
