#pragma once

#include <cstddef>
#include <fstream>
#include <vector>
#include <cstring>

namespace icicle {
  class Serializable
  {
  public:
    virtual ~Serializable() = default;

    virtual eIcicleError serialized_size(size_t& size) const { return eIcicleError::API_NOT_IMPLEMENTED; }

    virtual eIcicleError serialize(std::byte*& buffer, size_t& buffer_length) const
    {
      return eIcicleError::API_NOT_IMPLEMENTED;
    }

    virtual eIcicleError deserialize(std::byte*& buffer, size_t& buffer_length)
    {
      return eIcicleError::API_NOT_IMPLEMENTED;
    }

    eIcicleError serialize_to_file(const std::string& path) const
    {
      std::ofstream file(path, std::ios::binary);
      if (!file.is_open()) { return eIcicleError::INVALID_ARGUMENT; }

      size_t buffer_length;
      ICICLE_CHECK_IF_RETURN(serialized_size(buffer_length));
      std::vector<std::byte> buffer(buffer_length);
      std::byte* ptr = buffer.data();
      size_t remaining_length = buffer_length;
      ICICLE_CHECK_IF_RETURN(serialize(ptr, remaining_length));
      file.write(reinterpret_cast<const char*>(buffer.data()), buffer_length);
      return eIcicleError::SUCCESS;
    }

    eIcicleError deserialize_from_file(const std::string& path)
    {
      std::ifstream file(path, std::ios::binary | std::ios::ate);
      if (!file.is_open()) { return eIcicleError::INVALID_ARGUMENT; }

      std::streamsize size = file.tellg();
      if (size <= 0) { return eIcicleError::INVALID_ARGUMENT; }
      file.seekg(0, std::ios::beg);

      std::vector<std::byte> buffer(static_cast<size_t>(size));
      std::byte* buffer_ptr = buffer.data();
      if (!file.read(reinterpret_cast<char*>(buffer_ptr), size)) { return eIcicleError::INVALID_ARGUMENT; }

      size_t buffer_length = static_cast<size_t>(size);
      return deserialize(buffer_ptr, buffer_length);
    }

  protected:
    static eIcicleError
    memcpy_shift_destination(std::byte*& destination, size_t& remaining_length, const void* source, size_t copy_length)
    {
      if (remaining_length < copy_length) {
        ICICLE_LOG_ERROR << "memcpy_shift_destination failed: remaining_length < copy_length: " << remaining_length
                         << " < " << copy_length;
        return eIcicleError::INVALID_ARGUMENT;
      }
      std::memcpy(destination, source, copy_length);
      destination += copy_length;
      remaining_length -= copy_length;
      return eIcicleError::SUCCESS;
    }

    static eIcicleError
    memcpy_shift_source(void* destination, size_t& remaining_length, std::byte*& source, size_t copy_length)
    {
      if (remaining_length < copy_length) {
        ICICLE_LOG_ERROR << "memcpy_shift_source failed: remaining_length < copy_length: " << remaining_length << " < "
                         << copy_length;
        return eIcicleError::INVALID_ARGUMENT;
      }
      std::memcpy(destination, source, copy_length);
      source += copy_length;
      remaining_length -= copy_length;
      return eIcicleError::SUCCESS;
    }
  };
} // namespace icicle