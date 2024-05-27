#include "icicle/runtime.h"
#include "icicle/runtime.h"
#include "icicle/device_api.h"
#include "icicle/errors.h"

using namespace icicle;

extern "C" eIcicleError icicleSetDevice(const Device& device) { return DeviceAPI::setThreadLocalDevice(device); }

extern "C" eIcicleError icicleMalloc(void** ptr, size_t size)
{
  return DeviceAPI::getThreadLocalDeviceAPI()->allocateMemory(ptr, size);
}

extern "C" eIcicleError icicleMallocAsync(void** ptr, size_t size, icicleStreamHandle stream)
{
  return DeviceAPI::getThreadLocalDeviceAPI()->allocateMemoryAsync(ptr, size, stream);
}

extern "C" eIcicleError icicleFree(void* ptr) { return DeviceAPI::getThreadLocalDeviceAPI()->freeMemory(ptr); }

extern "C" eIcicleError icicleFreeAsync(void* ptr, icicleStreamHandle stream)
{
  return DeviceAPI::getThreadLocalDeviceAPI()->freeMemoryAsync(ptr, stream);
}

extern "C" eIcicleError icicleGetAvailableMemory(size_t& total /*OUT*/, size_t& free /*OUT*/)
{
  return DeviceAPI::getThreadLocalDeviceAPI()->getAvailableMemory(total, free);
}

extern "C" eIcicleError icicleCopyToHost(void* dst, const void* src, size_t size)
{
  return DeviceAPI::getThreadLocalDeviceAPI()->copyToHost(dst, src, size);
}

extern "C" eIcicleError icicleCopyToHostAsync(void* dst, const void* src, size_t size, icicleStreamHandle stream)
{
  return DeviceAPI::getThreadLocalDeviceAPI()->copyToHostAsync(dst, src, size, stream);
}

extern "C" eIcicleError icicleCopyToDevice(void* dst, const void* src, size_t size)
{
  return DeviceAPI::getThreadLocalDeviceAPI()->copyToDevice(dst, src, size);
}

extern "C" eIcicleError icicleCopyToDeviceAsync(void* dst, const void* src, size_t size, icicleStreamHandle stream)
{
  return DeviceAPI::getThreadLocalDeviceAPI()->copyToDeviceAsync(dst, src, size, stream);
}

extern "C" eIcicleError icicleStreamSynchronize(icicleStreamHandle stream)
{
  return DeviceAPI::getThreadLocalDeviceAPI()->synchronize(stream);
}

extern "C" eIcicleError icicleDeviceSynchronize() { return DeviceAPI::getThreadLocalDeviceAPI()->synchronize(nullptr); }

extern "C" eIcicleError icicleCreateStream(icicleStreamHandle* stream)
{
  return DeviceAPI::getThreadLocalDeviceAPI()->createStream(stream);
}

extern "C" eIcicleError icicleDestroyStream(icicleStreamHandle stream)
{
  return DeviceAPI::getThreadLocalDeviceAPI()->destroyStream(stream);
}
