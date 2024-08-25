#pragma once

#include <iostream>
#include <sstream>

#define ICICLE_LOG_DEBUG   Log(Log::Debug)
#define ICICLE_LOG_INFO    Log(Log::Info)
#define ICICLE_LOG_WARNING Log(Log::Warning)
#define ICICLE_LOG_ERROR   Log(Log::Error)

class Log
{
public:
  enum eLogLevel { Debug, Info, Warning, Error };

  Log(eLogLevel level) : level{level}
  {
    if (level >= s_min_log_level) { oss << "[" << logLevelToString(level) << "] "; }
  }

  ~Log()
  {
    if (level >= s_min_log_level) { std::cerr << oss.str() << std::endl; }
  }

  template <typename T>
  Log& operator<<(const T& msg)
  {
    if (level >= s_min_log_level) { oss << msg; }
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
    switch (level) {
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

  // logging message with level>=s_min_log_level
#if defined(NDEBUG)
  static inline eLogLevel s_min_log_level = eLogLevel::Info;
#else
  static inline eLogLevel s_min_log_level = eLogLevel::Debug;
#endif
};
