#pragma once
#ifndef DEVICE_CONTEXT_H
#define DEVICE_CONTEXT_H

#include <stdlib.h>
// used as dummy device context to keep interface consistent with GPU
namespace device_context {

  constexpr int MAX_DEVICES = 32;

  /**
   * Properties of the device used in icicle functions.
   */
  struct DeviceContext {
    int stream;  /**< Stream to use. Default value: 0. */
    int device_id; /**< Index of the currently used GPU. Default value: 0. */
    int mempool; /**< Mempool to use. Default value: 0. */
  };

  /**
   * Return default device context that corresponds to using the default stream of the first GPU
   */
  // inline DeviceContext get_default_device_context() // TODO: naming convention ?
  // {
  //   static cudaStream_t default_stream = (cudaStream_t)0;
  //   return DeviceContext{
  //     (cudaStream_t&)default_stream, // stream
  //     0,                             // device_id
  //     0,                             // mempool
  //   };
  // }

} // namespace device_context

#endif