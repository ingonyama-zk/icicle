#pragma once
#include <cstddef>

// Include headers at top level
#if defined(__APPLE__) && defined(__MACH__)
    #include <sys/types.h>
    #include <sys/sysctl.h>
#elif defined(_WIN32) || defined(_WIN64)
    #include <windows.h>
    #include <vector>
#elif defined(__linux__)
    #include <fstream>
#endif

std::size_t get_cache_line_size() {
#if defined(__APPLE__) && defined(__MACH__)
    {
        size_t lineSize = 0;
        size_t sizeOfLineSize = sizeof(lineSize);
        // sysctlbyname is available on macOS to retrieve cache line size
        if (sysctlbyname("hw.cachelinesize", &lineSize, &sizeOfLineSize, nullptr, 0) == 0) {
            return lineSize;
        }
        return 0; // Could not determine
    }

#elif defined(_WIN32) || defined(_WIN64)
    {
        DWORD bufferSize = 0;
        GetLogicalProcessorInformation(nullptr, &bufferSize);
        if (bufferSize == 0) {
            return 0; // Could not determine
        }

        std::vector<unsigned char> buffer(bufferSize);
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION*>(buffer.data());
        if (!GetLogicalProcessorInformation(info, &bufferSize)) {
            return 0; // Could not retrieve info
        }

        DWORD count = bufferSize / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
        for (DWORD i = 0; i < count; ++i) {
            if (info[i].Relationship == RelationCache && info[i].Cache.Level == 1) {
                return info[i].Cache.LineSize;
            }
        }
        return 0; // No cache info found
    }

#elif defined(__linux__)
    {
        std::ifstream ifs("/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size");
        if (!ifs) {
            return 0; // Could not open file
        }
        std::size_t lineSize = 0;
        ifs >> lineSize;
        return lineSize;
    }

#else
    // Fallback
    return 64;
#endif
}
