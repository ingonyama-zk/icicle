#pragma once

#include <stdexcept>
#include <string>
#include <memory>
#include <list>

#include "icicle/device.h"
#include "icicle/errors.h"

namespace icicle {

  /**
   * @brief Typedef for an abstract stream used for asynchronous operations.
   */
  typedef void* icicleStreamHandle;

  /**
   * @brief Abstract class representing the API for device operations.
   */
  class DeviceAPI
  {
  public:
    /**
     * @brief Virtual destructor for DeviceAPI.
     */
    virtual ~DeviceAPI() {}

    /**
     * @brief Set active device for thread
     *
     * @param device The device
     * @return eIcicleError Status of the device set
     */
    virtual eIcicleError set_device(const Device& device) = 0;

    // Memory management

    /**
     * @brief Allocates memory on the specified device.
     *
     * @param ptr Pointer to the allocated memory.
     * @param size Size of the memory to allocate.
     * @return eIcicleError Status of the memory allocation.
     */
    virtual eIcicleError allocate_memory(void** ptr, size_t size) const = 0;

    /**
     * @brief Asynchronously allocates memory on the specified device.
     *
     * @param ptr Pointer to the allocated memory.
     * @param size Size of the memory to allocate.
     * @param stream Stream to use for the asynchronous operation.
     * @return eIcicleError Status of the memory allocation.
     */
    virtual eIcicleError allocate_memory_async(void** ptr, size_t size, icicleStreamHandle stream) const = 0;

    /**
     * @brief Frees memory on the specified device.
     *
     * @param ptr Pointer to the memory to free.
     * @return eIcicleError Status of the memory deallocation.
     */
    virtual eIcicleError free_memory(void* ptr) const = 0;

    /**
     * @brief Asynchronously frees memory on the specified device.
     *
     * @param ptr Pointer to the memory to free.
     * @param stream Stream to use for the asynchronous operation.
     * @return eIcicleError Status of the memory deallocation.
     */
    virtual eIcicleError free_memory_async(void* ptr, icicleStreamHandle stream) const = 0;

    /**
     * @brief Gets the total and available memory on the specified device.
     *
     * @param total Total memory available on the device (output parameter).
     * @param free Available memory on the device (output parameter).
     * @return eIcicleError Status of the memory query.
     */
    virtual eIcicleError get_available_memory(size_t& total /*OUT*/, size_t& free /*OUT*/) const = 0;

    // Data transfer

    /**
     * @brief Copies data from the device to the host.
     *
     * @param dst Destination pointer on the host.
     * @param src Source pointer on the device.
     * @param size Size of the data to copy.
     * @return eIcicleError Status of the data copy.
     */
    virtual eIcicleError copy_to_host(void* dst, const void* src, size_t size) const = 0;

    /**
     * @brief Asynchronously copies data from the device to the host.
     *
     * @param dst Destination pointer on the host.
     * @param src Source pointer on the device.
     * @param size Size of the data to copy.
     * @param stream Stream to use for the asynchronous operation.
     * @return eIcicleError Status of the data copy.
     */
    virtual eIcicleError
    copy_to_host_async(void* dst, const void* src, size_t size, icicleStreamHandle stream) const = 0;

    /**
     * @brief Copies data from the host to the device.
     *
     * @param dst Destination pointer on the device.
     * @param src Source pointer on the host.
     * @param size Size of the data to copy.
     * @return eIcicleError Status of the data copy.
     */
    virtual eIcicleError copy_to_device(void* dst, const void* src, size_t size) const = 0;

    /**
     * @brief Asynchronously copies data from the host to the device.
     *
     * @param dst Destination pointer on the device.
     * @param src Source pointer on the host.
     * @param size Size of the data to copy.
     * @param stream Stream to use for the asynchronous operation.
     * @return eIcicleError Status of the data copy.
     */
    virtual eIcicleError
    copy_to_device_async(void* dst, const void* src, size_t size, icicleStreamHandle stream) const = 0;

    // Synchronization

    /**
     * @brief Synchronizes the specified device or stream.
     *
     * @param stream The stream to synchronize (nullptr for device synchronization).
     * @return eIcicleError Status of the synchronization.
     */
    virtual eIcicleError synchronize(icicleStreamHandle stream = nullptr) const = 0;

    // Stream management

    /**
     * @brief Creates a stream on the specified device.
     *
     * @param stream Pointer to the created stream.
     * @return eIcicleError Status of the stream creation.
     */
    virtual eIcicleError create_stream(icicleStreamHandle* stream) const = 0;

    /**
     * @brief Destroys a stream on the specified device.
     *
     * @param stream The stream to destroy.
     * @return eIcicleError Status of the stream destruction.
     */
    virtual eIcicleError destroy_stream(icicleStreamHandle stream) const = 0;

  private:
    thread_local static inline Device sCurDevice = {nullptr, -1}; // device that is currently active for this thread
    thread_local static inline const DeviceAPI* sCurDeviceAPI;    // API for the currently active device of this thread

  public:
    static eIcicleError set_thread_local_device(const Device& device);
    static const Device& get_thread_local_device();
    static const DeviceAPI* get_thread_local_deviceAPI();
  };

  /**
   * @brief Retrieve a DeviceAPI instance based on the device.
   *
   * @param deviceType The device type to register the DeviceAPI for.
   * @param api An instance of the derived api type to be used as deviceAPI interface.
   */
  extern "C" void register_deviceAPI(const std::string& deviceType, std::shared_ptr<DeviceAPI> api);

  /**
   * @brief Register DeviceAPI instance for a device type.
   *
   * @param device The device to create the DeviceAPI instance for.
   * @return DeviceAPI* Pointer to the created DeviceAPI instance.
   */
  extern "C" DeviceAPI* get_deviceAPI(const Device& device);

  /**
   * @brief Retrieve a list of registered device types.
   *
   * @return A list of registered device types.
   */
  extern "C" std::list<std::string> get_registered_devices();

  /**
   * Device API registration macro.
   * Usage:
   * (1) implement the interface:
   *   Class MyDeviceAPI : public icicle::DeviceAPI {...}
   * (1) register:
   *  REGISTER_DEVICE_API("MyDevice", MyDeviceAPI);
   */

#define REGISTER_DEVICE_API(DEVICE_TYPE, API_CLASS)                                                                    \
  namespace {                                                                                                          \
    static bool _reg_device_##API_CLASS = []() -> bool {                                                               \
      std::shared_ptr<DeviceAPI> apiInstance = std::make_shared<API_CLASS>();                                          \
      register_deviceAPI(DEVICE_TYPE, apiInstance);                                                                    \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

} // namespace icicle