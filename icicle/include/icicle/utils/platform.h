#pragma once
#include <string>
#include <sys/utsname.h>
#include <unistd.h>

#if defined(_MSC_VER)
  #include <intrin.h>
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
  #include <cpuid.h>
  #include <stdint.h>
#endif

inline bool is_arm_platform()
{
  struct utsname unameData;
  if (uname(&unameData) == 0) {
    std::string machine(unameData.machine);
    return machine == "arm64" || machine == "aarch64";
  }
  return false;
}

inline bool is_x86_platform()
{
  struct utsname unameData;
  if (uname(&unameData) == 0) {
    std::string machine(unameData.machine);
    return machine == "x86_64" || machine == "i386";
  }
  return false;
}

inline std::string get_cpu_vendor()
{
  if (is_arm_platform()) { return "ARM"; }

  if (is_x86_platform()) {
    // x86/x86_64: Use CPUID
    char vendor[0x20] = {};
#if defined(_MSC_VER)
    int cpuInfo[4] = {0};
    __cpuid(cpuInfo, 0);
    *reinterpret_cast<int*>(vendor) = cpuInfo[1];
    *reinterpret_cast<int*>(vendor + 4) = cpuInfo[3];
    *reinterpret_cast<int*>(vendor + 8) = cpuInfo[2];
#elif defined(__GNUC__) || defined(__clang__)
    uint32_t eax = 0, ebx = 0, ecx = 0, edx = 0;
  #if defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(0));
    *reinterpret_cast<uint32_t*>(vendor) = ebx;
    *reinterpret_cast<uint32_t*>(vendor + 4) = edx;
    *reinterpret_cast<uint32_t*>(vendor + 8) = ecx;
  #endif
#endif
    std::string vendor_str(vendor);
    if (vendor_str.find("GenuineIntel") != std::string::npos) return "Intel";
    if (vendor_str.find("AuthenticAMD") != std::string::npos) return "AMD";
  }

  return "Unknown";
}

inline int get_cpu_vendor_as_int()
{
  std::string vendor = get_cpu_vendor();
  if (vendor == "ARM") return 2;
  if (vendor == "AMD") return 1;
  return 0;
}
