#pragma once 
#include <atomic>
#include <thread>
#include <stdexcept>
#include <cassert>

#define LOG_TASKS_PER_THREAD 2
#define TASKS_PER_THREAD (1 << LOG_TASKS_PER_THREAD)
#define TASK_IDX_MASK (TASKS_PER_THREAD - 1)
#define MANAGER_SLEEP_USEC 1
#define THREAD_SLEEP_USEC 1

/**
 * @class TaskBase
 * @brief abstract base for a task supported by `TasksManager`.
 * Important 
 */
class TaskBase {
public:
  /**
   * @brief constructor for `TaskBase`.
   */
  TaskBase() : m_status(IDLE) {}

  /**
   * @brief pure virtual function to be executed by `TasksManager`. Implemented by derived class:
   * This is the actual task to be calculated by `TaskManager` and its workers.
   */
  virtual void execute() = 0;

  /**
   * @brief Signal for the `Worker` owning the task that it is ready to be executed. 
   * User function.
   */
  void dispatch() { m_status.store(READY, std::memory_order_release); }

  // Getters and setter for the various states of the tasks `m_status`.
  inline bool is_ready()      { return m_status.load(std::memory_order_acquire) == READY; }
  inline bool is_completed()  { return m_status.load(std::memory_order_acquire) == COMPLETED; }
  inline bool is_idle()       { return m_status.load(std::memory_order_acquire) == IDLE; }
  inline void set_idle()      { m_status.store(IDLE, std::memory_order_release); }
  
  // COMMENT given that a setter for ready is implemented via dispatch, and a setter for idle is required to allow 
  // users to handle completed results 2/3 setters have been implemented. As such one of the functions bellow 
  // (set_completed, increment_status) is irrelevant - Help me choose which
  inline void set_completed() { assert(is_ready()); m_status.store(COMPLETED, std::memory_order_release); }
  
  inline void increment_status() {
    auto curr_status = m_status.load(std::memory_order_acquire);
    if (curr_status == COMPLETED) { m_status.store(IDLE, std::memory_order_release); }
    else { m_status.store(static_cast<eTaskStatus>(static_cast<int>(curr_status) + 1), std::memory_order_release); }
  }

  /**
   * @brief wait for a specific task to finish executing. This is a blocking function.
   */
  void wait_completed() {
    while (!is_completed()) {
      std::this_thread::sleep_for(std::chrono::microseconds(MANAGER_SLEEP_USEC));
    }
  }

private:
  /**
   * @enum containing the valid states of a task.
   */
  enum eTaskStatus {IDLE, READY, COMPLETED};
  std::atomic<eTaskStatus> m_status; // current task state. Atomic to ensure proper rd/wr order to sync threads.
};

/**
 * @class TasksManager
 * @brief Class for managing parallel executions of small `Task`s which are child class of `TaskBase` described bellow.
 * 
 * The class manages a vector of `Worker`s, which are threads and additional required data members for executions of 
 * `Task`s. The `Task`s are split in a thread-pool fashion - finding free slot in the `Worker`s for the user to set up 
 * additional tasks, and fetching completed `Task`s back to the user.
 * IMPORTANT NOTE: destroying this class or its worker members do not ensure handling of final task results, that is
 * the user's responsibility.
 */
template<class Task>
class TasksManager {
public:
  /**
   * @brief Constructor for `TaskManager`.
   * @param nof_workers - number of workers / threads to be ran simultaneously
   */
  TasksManager(const int nof_workers) : m_workers(nof_workers), m_next_worker_idx(0) {}

  /**
   * @brief Get free slot to insert new task to be executed. This is a blocking function - until a free task is found.
   * @return Task* - pointer to allow the user to edit in the new task. nullptr if no task is available.
   * NOTE: the users should check if the returned task is completed, and if they wish to handle the existing result.
   */
  Task* get_idle_or_completed_task();

  /**
   * @brief Get task that holds previous result to be handled by the user. This function blocks the code until a 
   * completed task is found or all tasks are idle with no result.
   * @return Task* - pointer to a completed task. nullptr if no task is available (all are idle without results).
   * NOTE: The task's status should be updated if new tasks are to be assigned / other completed tasks are requested.
   * Use dispatch 
   */
  Task* get_completed_task();

  /**
   * @brief Wait until all workers are done - i.e. all tasks are idle or completed.
   */
  void wait_done();
private:
  /**
   * @class Worker
   * @brief the equivalent of a thread and additional data members required for executing tasks in parallel to main.
   */
  class Worker {
  public:
    /**
     * @brief Constructor of `Worker`.
     * Inits default values for the class's members and launches the thread.
     */
    Worker();
    /**
     * @brief Destructor of `Worker`.
     * Signals the thread to terminate and joins it with main. The destructor does not handle existing tasks' results 
     * and assumes the user have already handled all the results via `TasksManager`'s api.
     */
    ~Worker();

    /**
     * @brief function to be ran by the thread.
     * Routinely checks for valid inputs in all of the worker's tasks. It executes valid tasks, later marking back that 
     * the tasks are complete. This loops until a kill signal is sent via the class's destructor.
     */
    void worker_loop();

    /**
     * @brief Get free slot to insert new task to be executed. This isn't a blocking function - it checks all worker's 
     * tasks and if no free one is found a nullptr is returned.
     * @return Task* - pointer to the internal task of the worker to allow the user to edit in the new task.
     * * NOTE: the users should check if the returned task is completed, and if they wish to handle the existing result.
     */
    Task* get_idle_or_completed_task();

    /**
     * @brief Get task that holds previous result to be handled by the user. This isn't a blocking function - it checks 
     * all worker's tasks and if no completed one is found a nullptr is returned.
     * @param is_idle - boolean flag indicating if all worker's tasks are idle.
     * @return Task* - pointer to a completed task. nullptr if no task is available.
     * NOTE: if using completed_task to assign additional tasks, the existing result must be handled before hand.
     */
    Task* get_completed_task(bool& is_idle);

    /**
     * @brief Blocking function until all worker's tasks are done - i.e. idle or completed.
     */
    void wait_done();
  private:
    std::thread task_executor; // Thread to be run parallel to main
    std::vector<Task> m_tasks; // vector containing the worker's task. a Vector is used to allow buffering.
    int m_next_task_idx; // Tail (input) idx of the fifo above. Checks for free task start at this idx.
    int m_head; // Head (output) idx of the fifo above, the thread works on task at this idx. // TODO remove after debug
    bool kill; // boolean to flag from main to the thread to finish.
  };

  std::vector<Worker> m_workers; // Vector of workers/threads to be ran simultaneously.
  int m_next_worker_idx;
};

template<class Task>
TasksManager<Task>::Worker::Worker() 
: m_tasks(TASKS_PER_THREAD), 
  m_next_task_idx(0),
  m_head(0),
  kill(false)
  {
    // Init thread only after finishing all other setup to avoid data races
    task_executor = std::thread(&TasksManager<Task>::Worker::worker_loop, this);
  }

template<class Task>
TasksManager<Task>::Worker::~Worker() {
  kill = true;
  task_executor.join();
}

template<class Task>
void TasksManager<Task>::Worker::worker_loop() {
  while (true) {
    bool all_tasks_idle = true;
    for (m_head = 0; m_head < m_tasks.size(); m_head++)
    {      
      Task* task = &m_tasks[m_head];
      if (task->is_ready()) {
        task->execute();
        task->set_completed(); 
        all_tasks_idle = false;
      }
    }
    if (kill) { return; }
    if (all_tasks_idle)
    {
      // Sleep as the thread apparently isn't fully utilized currently
      std::this_thread::sleep_for(std::chrono::microseconds(THREAD_SLEEP_USEC));  
    }
  }
}

template<class Task>
Task* TasksManager<Task>::Worker::get_idle_or_completed_task() {
  for (int i = 0; i < m_tasks.size(); i++)
  {
    // TASKS_PER_WORKER is a power of 2 so modulo is done via bitmask.
    m_next_task_idx = (1 + m_next_task_idx) & TASK_IDX_MASK; 

    if (m_tasks[m_next_task_idx].is_idle() || m_tasks[m_next_task_idx].is_completed())
    {
      return &m_tasks[m_next_task_idx];
    }
  }
  return nullptr;
}

template<class Task>
Task* TasksManager<Task>::Worker::get_completed_task(bool& is_idle) {
  for (int i = 0; i < m_tasks.size(); i++)
  {
    m_next_task_idx = (1 + m_next_task_idx) & TASK_IDX_MASK; 

    if (m_tasks[m_next_task_idx].is_completed())
    {
      is_idle = false;
      return &m_tasks[m_next_task_idx];
    }
    if (!m_tasks[m_next_task_idx].is_idle()){ is_idle = false; }
  }
  return nullptr;
}

template<class Task>
void TasksManager<Task>::Worker::wait_done() {
  bool all_done = false;
  while (!all_done) 
  {
    all_done = true;
    for (Task& task : m_tasks) 
    { 
      if (!(m_tasks[m_next_task_idx].is_idle() || m_tasks[m_next_task_idx].is_completed())) 
      { 
        all_done = false; 
      }
    }
  }
}

template<class Task>
Task* TasksManager<Task>::get_idle_or_completed_task() {
  Task* task = nullptr;
  while (true)
  {
    for (int i = 0; i < m_workers.size(); i++)
    {
      m_next_worker_idx = (m_next_worker_idx < m_workers.size() - 1)? m_next_worker_idx + 1 : 0;

      task = m_workers[m_next_worker_idx].get_idle_or_completed_task();
      if (task != nullptr) { return task; }
    }
    // std::this_thread::sleep_for(std::chrono::microseconds(MANAGER_SLEEP_USEC));
  }
}

template<class Task>
Task* TasksManager<Task>::get_completed_task() {
  Task* completed_task = nullptr;
  bool all_idle = false;
  while (!all_idle)
  {
    // Flag sent by reference to get_completed_task below - if a task that isn't idle is found the flag is set to false
    all_idle = true; 
    for (int i = 0; i < m_workers.size(); i++)
    {
      m_next_worker_idx = (m_next_worker_idx < m_workers.size() - 1)? m_next_worker_idx + 1 : 0;

      completed_task = m_workers[m_next_worker_idx].get_completed_task(all_idle);
      if (completed_task != nullptr) { return completed_task; }
    }
    // std::this_thread::sleep_for(std::chrono::microseconds(MANAGER_SLEEP_USEC));
  }
  // No completed tasks were found in the loop - return null.
  completed_task = nullptr;
}

template<class Task>
void TasksManager<Task>::wait_done() {
  for (Worker& worker : m_workers)
  {
    worker.wait_done();
  }
}