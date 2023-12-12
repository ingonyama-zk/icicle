#include "device_context.cuh"

namespace device_context {

  extern "C" DeviceContext GetDefaultDeviceContext() { return get_default_device_context(); }

} // namespace device_context
