#pragma once

#include <stdexcept>
#include <string>
#include <memory>
#include <list>

#include "device.h"
#include "errors.h"

namespace icicle {

  /**
   * @brief Typedef for an abstract stream used for asynchronous operations.
   */
  typedef void* IcicleStreamHandle;

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

    // Memory management

    /**
     * @brief Allocates memory on the specified device.
     *
     * @param device The device on which to allocate memory.
     * @param ptr Pointer to the allocated memory.
     * @param size Size of the memory to allocate.
     * @return IcicleError Status of the memory allocation.
     */
    virtual IcicleError allocateMemory(const Device& device, void** ptr, size_t size) = 0;

    /**
     * @brief Asynchronously allocates memory on the specified device.
     *
     * @param device The device on which to allocate memory.
     * @param ptr Pointer to the allocated memory.
     * @param size Size of the memory to allocate.
     * @param stream Stream to use for the asynchronous operation.
     * @return IcicleError Status of the memory allocation.
     */
    virtual IcicleError
    allocateMemoryAsync(const Device& device, void** ptr, size_t size, IcicleStreamHandle stream) = 0;

    /**
     * @brief Frees memory on the specified device.
     *
     * @param device The device on which to free memory.
     * @param ptr Pointer to the memory to free.
     * @return IcicleError Status of the memory deallocation.
     */
    virtual IcicleError freeMemory(const Device& device, void* ptr) = 0;

    /**
     * @brief Asynchronously frees memory on the specified device.
     *
     * @param device The device on which to free memory.
     * @param ptr Pointer to the memory to free.
     * @param stream Stream to use for the asynchronous operation.
     * @return IcicleError Status of the memory deallocation.
     */
    virtual IcicleError freeMemoryAsync(const Device& device, void* ptr, IcicleStreamHandle stream) = 0;

    /**
     * @brief Gets the total and available memory on the specified device.
     *
     * @param device The device to query.
     * @param total Total memory available on the device (output parameter).
     * @param free Available memory on the device (output parameter).
     * @return IcicleError Status of the memory query.
     */
    virtual IcicleError getAvailableMemory(const Device& device, size_t& total /*OUT*/, size_t& free /*OUT*/) = 0;

    // Data transfer

    /**
     * @brief Copies data from the device to the host.
     *
     * @param device The device to copy data from.
     * @param dst Destination pointer on the host.
     * @param src Source pointer on the device.
     * @param size Size of the data to copy.
     * @return IcicleError Status of the data copy.
     */
    virtual IcicleError copyToHost(const Device& device, void* dst, const void* src, size_t size) = 0;

    /**
     * @brief Asynchronously copies data from the device to the host.
     *
     * @param device The device to copy data from.
     * @param dst Destination pointer on the host.
     * @param src Source pointer on the device.
     * @param size Size of the data to copy.
     * @param stream Stream to use for the asynchronous operation.
     * @return IcicleError Status of the data copy.
     */
    virtual IcicleError
    copyToHostAsync(const Device& device, void* dst, const void* src, size_t size, IcicleStreamHandle stream) = 0;

    /**
     * @brief Copies data from the host to the device.
     *
     * @param device The device to copy data to.
     * @param dst Destination pointer on the device.
     * @param src Source pointer on the host.
     * @param size Size of the data to copy.
     * @return IcicleError Status of the data copy.
     */
    virtual IcicleError copyToDevice(const Device& device, void* dst, const void* src, size_t size) = 0;

    /**
     * @brief Asynchronously copies data from the host to the device.
     *
     * @param device The device to copy data to.
     * @param dst Destination pointer on the device.
     * @param src Source pointer on the host.
     * @param size Size of the data to copy.
     * @param stream Stream to use for the asynchronous operation.
     * @return IcicleError Status of the data copy.
     */
    virtual IcicleError
    copyToDeviceAsync(const Device& device, void* dst, const void* src, size_t size, IcicleStreamHandle stream) = 0;

    // Synchronization

    /**
     * @brief Synchronizes the specified device or stream.
     *
     * @param device The device to synchronize.
     * @param stream The stream to synchronize (nullptr for device synchronization).
     * @return IcicleError Status of the synchronization.
     */
    virtual IcicleError synchronize(const Device& device, IcicleStreamHandle stream = nullptr) = 0;

    // Stream management

    /**
     * @brief Creates a stream on the specified device.
     *
     * @param device The device to create the stream on.
     * @param stream Pointer to the created stream.
     * @return IcicleError Status of the stream creation.
     */
    virtual IcicleError createStream(const Device& device, IcicleStreamHandle* stream) = 0;

    /**
     * @brief Destroys a stream on the specified device.
     *
     * @param device The device to destroy the stream on.
     * @param stream The stream to destroy.
     * @return IcicleError Status of the stream destruction.
     */
    virtual IcicleError destroyStream(const Device& device, IcicleStreamHandle stream) = 0;
  };

  /**
   * @brief Retrieve a DeviceAPI instance based on the device.
   *
   * @param deviceType The device type to register the DeviceAPI for.
   * @param api An instance of the derived api type to be used as deviceAPI interface.
   */
  extern "C" void registerDeviceAPI(const std::string& deviceType, std::shared_ptr<DeviceAPI> api);

  /**
   * @brief Register DeviceAPI instance for a device type.
   *
   * @param device The device to create the DeviceAPI instance for.
   * @return DeviceAPI* Pointer to the created DeviceAPI instance.
   */
  extern "C" DeviceAPI* getDeviceAPI(const Device* device);

  /**
   * @brief Retrieve a list of registered device types.
   *
   * @return A list of registered device types.
   */
  extern "C" std::list<std::string> getRegisteredDevices();


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
      registerDeviceAPI(DEVICE_TYPE, apiInstance);                                                                     \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

} // namespace icicle