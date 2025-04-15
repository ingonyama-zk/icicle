#pragma once

#include <cstddef>
#include <fstream>
#include <vector>

namespace icicle {
class Serializer {
public:
    virtual ~Serializer() = default;

    virtual eIcicleError serialized_size(size_t& size) const
    {
        return eIcicleError::API_NOT_IMPLEMENTED;
    }

    virtual eIcicleError serialize(std::byte*& out) const
    {
        return eIcicleError::API_NOT_IMPLEMENTED;
    }

    virtual eIcicleError deserialize(std::byte*& in, size_t& length)
    {
        return eIcicleError::API_NOT_IMPLEMENTED;
    }

    eIcicleError serialize_to_file(const std::string& filename) const
    {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            return eIcicleError::INVALID_ARGUMENT;
        }

        size_t size;
        eIcicleError err = serialized_size(size);
        if (err != eIcicleError::SUCCESS) {
            return err;
        }
        std::vector<std::byte> buffer(size);
        std::byte* ptr = buffer.data();
        err = serialize(ptr);
        if (err != eIcicleError::SUCCESS) {
            return err;
        }
        file.write(reinterpret_cast<const char*>(buffer.data()), size);
        return eIcicleError::SUCCESS;
    }

    eIcicleError deserialize_from_file(const std::string& filename)
    {
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            return eIcicleError::INVALID_ARGUMENT;
        }

        std::streamsize size = file.tellg();
        if (size <= 0) {
            return eIcicleError::INVALID_ARGUMENT;
        }
        file.seekg(0, std::ios::beg);

        std::vector<std::byte> buffer(static_cast<size_t>(size));
        std::byte* ptr = buffer.data();
        if (!file.read(reinterpret_cast<char*>(ptr), size)) {
            return eIcicleError::INVALID_ARGUMENT;
        }

        size_t length = static_cast<size_t>(size);
        return deserialize(ptr, length);
    }
};
} // namespace icicle