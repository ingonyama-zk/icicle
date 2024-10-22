#pragma once
#if defined(__ANDROID__)
#include <android/log.h>
#endif
#include <sstream>

// Define log macros
#define ICICLE_LOG_VERBOSE Log(Log::Verbose)
#define ICICLE_LOG_DEBUG   Log(Log::Debug)
#define ICICLE_LOG_INFO    Log(Log::Info)
#define ICICLE_LOG_WARNING Log(Log::Warning)
#define ICICLE_LOG_ERROR   Log(Log::Error)

class Log
{
public:
    enum eLogLevel { Verbose, Debug, Info, Warning, Error };

    Log(eLogLevel level) : level{ level }
    {
        if (level >= s_min_log_level)
        {
            oss << "[" << logLevelToString(level) << "] ";
        }
    }

    ~Log()
    {
        if (level >= s_min_log_level)
        {
            // Output the log message using the platform-specific logging function
            logMessage(level, oss.str());
        }
    }

    template <typename T>
    Log& operator<<(const T& msg)
    {
        if (level >= s_min_log_level)
        {
            oss << msg;
        }
        return *this;
    }

    // Static method to set the log level
    static void set_min_log_level(eLogLevel level) { s_min_log_level = level; }

private:
    eLogLevel level;
    std::ostringstream oss;

    // Convert log level to string
    const char* logLevelToString(eLogLevel level) const
    {
        switch (level)
        {
            case Verbose:
                return "VERBOSE";
            case Debug:
                return "DEBUG";
            case Info:
                return "INFO";
            case Warning:
                return "WARNING";
            case Error:
                return "ERROR";
            default:
                return "";
        }
    }

    // Platform-specific log output
    void logMessage(eLogLevel level, const std::string& message)
    {
#ifdef __ANDROID__
        // Android-specific logging
        android_LogPriority priority = androidLogPriority(level);
        __android_log_print(priority, "IcicleNative", "%s", message.c_str());
#else
        // Other platforms (e.g., standard error output)
        std::cerr << message << std::endl;
#endif
    }

#ifdef __ANDROID__
    // Convert eLogLevel to android_LogPriority
    android_LogPriority androidLogPriority(eLogLevel level) const
    {
        switch (level)
        {
            case Verbose:
                return ANDROID_LOG_VERBOSE;
            case Debug:
                return ANDROID_LOG_DEBUG;
            case Info:
                return ANDROID_LOG_INFO;
            case Warning:
                return ANDROID_LOG_WARN;
            case Error:
                return ANDROID_LOG_ERROR;
            default:
                return ANDROID_LOG_UNKNOWN;
        }
    }
#endif

    // Static member for minimum log level
#if defined(NDEBUG)
    static inline eLogLevel s_min_log_level = eLogLevel::Info;
#else
    static inline eLogLevel s_min_log_level = eLogLevel::Debug;
#endif
};

