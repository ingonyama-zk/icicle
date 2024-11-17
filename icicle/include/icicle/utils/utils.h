#pragma once
#include <string>
#include <memory>
#include <cxxabi.h>
#include <iomanip>
#include <iostream>

#define CONCAT_DIRECT(a, b) a##_##b
#define CONCAT_EXPAND(a, b) CONCAT_DIRECT(a, b) // expand a,b before concatenation
#define UNIQUE(a)           CONCAT_EXPAND(a, __LINE__)

// Template function to demangle the name of the type
template <typename T>
std::string demangle()
{
  int status = -4;
  std::unique_ptr<char, void (*)(void*)> res{
    abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, &status), std::free};

  return (status == 0) ? res.get() : typeid(T).name();
}

// Debug
static void print_bytes(const std::byte* data, const uint nof_elements, const uint element_size)
{
  for (uint element_idx = 0; element_idx < nof_elements; ++element_idx) {
    std::cout << "0x";
    for (int byte_idx = 0; byte_idx < element_size; ++byte_idx) {
      std::cout << std::hex << std::setw(2) << std::setfill('0')
                << static_cast<int>(data[element_idx * element_size + byte_idx]);
    }
    std::cout << std::dec << ",\n";
  }
}