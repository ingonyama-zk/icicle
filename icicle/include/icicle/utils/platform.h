#pragma once
#include <string>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif

inline std::string get_cpu_vendor()
{
#if defined(__APPLE__) && (defined(__arm64__) || defined(__aarch64__))
    return "Apple";
#elif defined(_MSC_VER) || defined(__GNUC__) || defined(__clang__)
    // x86/x86_64: Use CPUID
    char vendor[0x20] = {};
#if defined(_MSC_VER)
    int cpuInfo[4] = {0};
    __cpuid(cpuInfo, 0);
    *reinterpret_cast<int*>(vendor) = cpuInfo[1];
    *reinterpret_cast<int*>(vendor + 4) = cpuInfo[3];
    *reinterpret_cast<int*>(vendor + 8) = cpuInfo[2];
#elif defined(__GNUC__) || defined(__clang__)
    unsigned int eax, ebx, ecx, edx;
    __get_cpuid(0, &eax, &ebx, &ecx, &edx);
    *reinterpret_cast<unsigned int*>(vendor) = ebx;
    *reinterpret_cast<unsigned int*>(vendor + 4) = edx;
    *reinterpret_cast<unsigned int*>(vendor + 8) = ecx;
#endif
    std::string vendor_str(vendor);
    if (vendor_str.find("GenuineIntel") != std::string::npos)
        return "Intel";
    if (vendor_str.find("AuthenticAMD") != std::string::npos)
        return "AMD";
    return "Unknown";
#else
    return "Unknown";
#endif
}

inline int get_cpu_vendor_as_int()
{
    std::string vendor = get_cpu_vendor();
    if (vendor == "Apple")
        return 2;
    if (vendor == "AMD")
        return 1;
    return 0;
}
