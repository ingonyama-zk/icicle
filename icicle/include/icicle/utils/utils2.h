#pragma once

// Only include what's needed based on enabled features
#if defined(ENABLE_NVTX_PRINT)
#include <chrono>
#include <iostream>
#endif

#if defined(ENABLE_NVTX_RANGE) || defined(ENABLE_NVTX_PRINT)
#include <stack>
#include <string>
#endif

#include <thread>

// NVTX features are disabled by default
// To enable, define either:
// ENABLE_NVTX_RANGE - for profiler markers
// ENABLE_NVTX_PRINT - for timing printouts
#if defined(ENABLE_NVTX_RANGE)
#include <nvToolsExt.h>
#include <cuda_runtime.h>
#endif

namespace utils2 {

#if defined(ENABLE_NVTX_RANGE) || defined(ENABLE_NVTX_PRINT)
// Full implementation when features are enabled
class nvtx_guard {
private:
    std::string description;
    #ifdef ENABLE_NVTX_PRINT
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    #endif

public:
    nvtx_guard(const std::string& desc, size_t indent)
        : description(std::string(indent, ' ') + desc)
        #ifdef ENABLE_NVTX_PRINT
        , start(std::chrono::high_resolution_clock::now())
        #endif
    {
        #ifdef ENABLE_NVTX_PRINT
        std::cout << "Starting: " << description << std::endl;
        #endif
    }

    nvtx_guard(nvtx_guard&& other) noexcept
        : description(std::move(other.description))
        #ifdef ENABLE_NVTX_PRINT
        , start(other.start)
        #endif
    {}

    nvtx_guard(const nvtx_guard&) = delete;
    nvtx_guard& operator=(const nvtx_guard&) = delete;
    nvtx_guard& operator=(nvtx_guard&&) = delete;
    ~nvtx_guard() = default;

    void pop() {
        #ifdef ENABLE_NVTX_PRINT
        auto duration = std::chrono::high_resolution_clock::now() - start;
        auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(duration);
        std::cout << "End of    " << description << ": time: " 
                  << microseconds.count() << "μs" << std::endl;
        #endif
    }
};

// Thread-local stack only when needed
thread_local std::stack<nvtx_guard> nvtx_stack;

#else
// Empty implementation when features are disabled
class nvtx_guard {
public:
    nvtx_guard(const std::string&, size_t) {}
    nvtx_guard(nvtx_guard&&) noexcept {}
    nvtx_guard(const nvtx_guard&) = delete;
    nvtx_guard& operator=(const nvtx_guard&) = delete;
    nvtx_guard& operator=(nvtx_guard&&) = delete;
    ~nvtx_guard() = default;
    void pop() {}
};
#endif

// Default to empty macros
#define NVTX_TIMED(desc) do {} while (0)
#define NVTX_TIMED_POP() do {} while (0)
#define _NVTX_RANGE_PUSH(desc, indent) do {} while (0)
#define _NVTX_RANGE_POP() do {} while (0)

// Override with actual implementations only if enabled
#if defined(ENABLE_NVTX_RANGE) || defined(ENABLE_NVTX_PRINT)
#undef NVTX_TIMED
#undef NVTX_TIMED_POP

#define NVTX_TIMED(desc) \
    do { \
        size_t indent = utils2::nvtx_stack.size(); \
        utils2::nvtx_stack.emplace(desc, indent); \
        _NVTX_RANGE_PUSH(desc, indent); \
    } while (0)

#define NVTX_TIMED_POP() \
    do { \
        if (!utils2::nvtx_stack.empty()) { \
            auto guard = std::move(utils2::nvtx_stack.top()); \
            utils2::nvtx_stack.pop(); \
            guard.pop(); \
            _NVTX_RANGE_POP(); \
        } \
    } while (0)
#endif

#ifdef ENABLE_NVTX_RANGE
#undef _NVTX_RANGE_PUSH
#undef _NVTX_RANGE_POP
#define _NVTX_RANGE_PUSH(desc, indent) nvtxRangePushA((std::string(indent, ' ') + desc).c_str())
#define _NVTX_RANGE_POP() nvtxRangePop()
#endif

} // namespace utils2 