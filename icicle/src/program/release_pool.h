#pragma once
#include <set>

template <typename T>
class ReleasePool
{
public:
  static ReleasePool& instance()
  {
    thread_local static ReleasePool<T> instance;
    return instance;
  }

  void insert(T* ptr) { m_pool.insert(ptr); }

  void clear()
  {
    // delete pointer before clearing the set
    for (auto ptr : m_pool) {
      delete ptr;
    }
    m_pool.clear();
  }

private:
  ReleasePool() = default;
  ~ReleasePool() = default;
  ReleasePool(const ReleasePool&) = delete;
  ReleasePool& operator=(const ReleasePool&) = delete;

  std::set<T*> m_pool;
};
