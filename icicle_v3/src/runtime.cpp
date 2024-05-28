#include "icicle/runtime.h"
#include "icicle/runtime.h"
#include "icicle/device_api.h"
#include "icicle/errors.h"

using namespace icicle;

extern "C" eIcicleError icicle_set_device(const Device& device) { return DeviceAPI::set_thread_local_device(device); }

extern "C" eIcicleError icicle_malloc(void** ptr, size_t size)
{
  return DeviceAPI::get_thread_local_deviceAPI()->allocate_memory(ptr, size);
}

extern "C" eIcicleError icicle_malloc_async(void** ptr, size_t size, icicleStreamHandle stream)
{
  return DeviceAPI::get_thread_local_deviceAPI()->allocate_memory_async(ptr, size, stream);
}

extern "C" eIcicleError icicle_free(void* ptr) { return DeviceAPI::get_thread_local_deviceAPI()->free_memory(ptr); }

extern "C" eIcicleError icicle_free_async(void* ptr, icicleStreamHandle stream)
{
  return DeviceAPI::get_thread_local_deviceAPI()->free_memory_async(ptr, stream);
}

extern "C" eIcicleError icicle_get_available_memory(size_t& total /*OUT*/, size_t& free /*OUT*/)
{
  return DeviceAPI::get_thread_local_deviceAPI()->get_available_memory(total, free);
}

extern "C" eIcicleError icicle_copy_to_host(void* dst, const void* src, size_t size)
{
  return DeviceAPI::get_thread_local_deviceAPI()->copy_to_host(dst, src, size);
}

extern "C" eIcicleError icicle_copy_to_host_async(void* dst, const void* src, size_t size, icicleStreamHandle stream)
{
  return DeviceAPI::get_thread_local_deviceAPI()->copy_to_host_async(dst, src, size, stream);
}

extern "C" eIcicleError icicle_copy_to_device(void* dst, const void* src, size_t size)
{
  return DeviceAPI::get_thread_local_deviceAPI()->copy_to_device(dst, src, size);
}

extern "C" eIcicleError icicle_copy_to_device_async(void* dst, const void* src, size_t size, icicleStreamHandle stream)
{
  return DeviceAPI::get_thread_local_deviceAPI()->copy_to_device_async(dst, src, size, stream);
}

extern "C" eIcicleError icicle_stream_synchronize(icicleStreamHandle stream)
{
  return DeviceAPI::get_thread_local_deviceAPI()->synchronize(stream);
}

extern "C" eIcicleError icicle_device_synchronize()
{
  return DeviceAPI::get_thread_local_deviceAPI()->synchronize(nullptr);
}

extern "C" eIcicleError icicle_create_stream(icicleStreamHandle* stream)
{
  return DeviceAPI::get_thread_local_deviceAPI()->create_stream(stream);
}

extern "C" eIcicleError icicle_destroy_stream(icicleStreamHandle stream)
{
  return DeviceAPI::get_thread_local_deviceAPI()->destroy_stream(stream);
}
