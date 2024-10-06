#pragma once
#include <string>
#include <memory>
#include <cxxabi.h>

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
