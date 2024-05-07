#pragma once
#ifndef DEVICE_CONTEXT_H
#define DEVICE_CONTEXT_H

namespace device_context {

  constexpr std::size_t MAX_DEVICES = 32;

  /**
   * Properties of the device used in icicle functions.
   */
  struct DeviceContext {
    int stream;  /**< Stream to use. Default value: 0. */
    std::size_t device_id; /**< Index of the currently used GPU. Default value: 0. */
    int mempool; /**< Mempool to use. Default value: 0. */
  };

  /**
   * Return default device context that corresponds to using the default stream of the first GPU
   */
  inline DeviceContext get_default_device_context() // TODO: naming convention ?
  {
    static int default_stream = 0;
    return DeviceContext{
      default_stream, // stream
      0,                             // device_id
      0,                             // mempool
    };
  }

} // namespace device_context

#endif