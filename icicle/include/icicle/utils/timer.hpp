#include <iostream>
#include <string>
#include <chrono>

/**
 * @class Timer
 * @brief class for simple time measurements and benchmarking of other libraries.
 */
class Timer
{
private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_point;
  std::string m_timer_name;

public:
  Timer(std::string timer_name)
  {
    start_point = std::chrono::high_resolution_clock::now();
    m_timer_name = timer_name;
  }

  ~Timer() { Stop(); }

  void Stop()
  {
    auto end_point = std::chrono::high_resolution_clock::now();
    auto start_time = std::chrono::time_point_cast<std::chrono::microseconds>(start_point).time_since_epoch().count();
    auto end_time = std::chrono::time_point_cast<std::chrono::microseconds>(end_point).time_since_epoch().count();
    auto duration = end_time - start_time;

    double dur_s = duration * 0.001;
    ICICLE_LOG_INFO << "Time of " << m_timer_name << ":\t" << dur_s << "ms\n";
  }
};