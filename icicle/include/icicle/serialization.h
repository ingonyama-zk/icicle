#pragma once

#include <cstddef>
#include <fstream>
#include <vector>
#include <cstring>
#include "sumcheck/sumcheck_proof.h"
#include "fri/fri_proof.h"
#include "merkle/merkle_proof.h"

namespace icicle {
  inline eIcicleError
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

  inline eIcicleError
  memcpy_shift_source(void* destination, size_t& remaining_length, const std::byte*& source, size_t copy_length)
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

  template <typename T>
  struct BinarySerializeImpl {
    static eIcicleError serialized_size(const T& obj, size_t& size);
    static eIcicleError pack_and_advance(std::byte*& buffer, size_t& buffer_length, const T& obj);
    static eIcicleError unpack_and_advance(const std::byte*& buffer, size_t& buffer_length, T& obj);
  };

  template <typename T>
  struct BinarySerializer {
    static eIcicleError serialized_size(const T& obj, size_t& size)
    {
      return BinarySerializeImpl<T>::serialized_size(obj, size);
    }
    static eIcicleError serialize(std::byte* buffer, size_t buffer_length, const T& obj)
    {
      size_t size;
      ICICLE_RETURN_IF_ERR(serialized_size(obj, size));
      if (buffer_length < size) {
        ICICLE_LOG_ERROR << "Serialization failed: buffer_length < size: " << buffer_length << " < " << size;
        return eIcicleError::INVALID_ARGUMENT;
      }
      return BinarySerializeImpl<T>::pack_and_advance(buffer, buffer_length, obj);
    }

    static eIcicleError deserialize(const std::byte* buffer, size_t buffer_length, T& obj)
    {
      return BinarySerializeImpl<T>::unpack_and_advance(buffer, buffer_length, obj);
    }

    static eIcicleError serialize_to_file(const std::string& path, const T& obj)
    {
      std::ofstream file(path, std::ios::binary);
      if (!file.is_open()) {
        ICICLE_LOG_ERROR << "Serialization failed: file is not open";
        return eIcicleError::INVALID_ARGUMENT;
      }

      size_t buffer_length;
      ICICLE_RETURN_IF_ERR(BinarySerializeImpl<T>::serialized_size(obj, buffer_length));
      std::vector<std::byte> buffer(buffer_length);
      std::byte* ptr = buffer.data();
      ICICLE_RETURN_IF_ERR(BinarySerializeImpl<T>::serialize(ptr, buffer_length, obj));
      file.write(reinterpret_cast<const char*>(buffer.data()), buffer_length);
      return eIcicleError::SUCCESS;
    }

    static eIcicleError deserialize_from_file(const std::string& path, T& obj)
    {
      std::ifstream file(path, std::ios::binary | std::ios::ate);
      if (!file.is_open()) {
        ICICLE_LOG_ERROR << "Deserialization failed: file is not open";
        return eIcicleError::INVALID_ARGUMENT;
      }

      std::streamsize size = file.tellg();
      if (size <= 0) {
        ICICLE_LOG_ERROR << "Deserialization failed: file size is not positive";
        return eIcicleError::INVALID_ARGUMENT;
      }
      file.seekg(0, std::ios::beg);

      std::vector<std::byte> buffer(static_cast<size_t>(size));
      std::byte* buffer_ptr = buffer.data();
      if (!file.read(reinterpret_cast<char*>(buffer_ptr), size)) {
        ICICLE_LOG_ERROR << "Deserialization failed: failed to read file";
        return eIcicleError::INVALID_ARGUMENT;
      }

      return BinarySerializeImpl<T>::deserialize(buffer_ptr, static_cast<size_t>(size), obj);
    }
  };

} // namespace icicle