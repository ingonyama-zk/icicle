#include <set>
#include <mutex>

using namespace icicle;

template <typename T>
class FfiObjectPool {
public:
  static FfiObjectPool& instance()
  {
    // Use lock(m_mutex) here to protect creation of the single instance
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    static FfiObjectPool<T> instance;
    return instance;
  }

  void add(T* ptr)
  {
    m_mutex.lock();
    m_pool.insert(ptr);
  }

  void clear()
  {
    // Needs amount of unlocks equal to pool size (For each lock done when adding an element)
    for (auto ptr: m_pool) {
      delete ptr;
      m_mutex.unlock();
    }
    m_pool.clear();
  }

private:
  FfiObjectPool() = default;
  ~FfiObjectPool() = default;
  FfiObjectPool(const FfiObjectPool&) = delete;
  FfiObjectPool& operator=(const FfiObjectPool&) = delete;

  std::set<T*> m_pool;
  static std::recursive_mutex m_mutex;
};

template <typename T> std::recursive_mutex FfiObjectPool<T>::m_mutex;
