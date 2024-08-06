#pragma once 
#include <atomic>
#include <thread> // TODO check windows support
#include <stdexcept>

#define LOG_TASKS_PER_THREAD 2
#define TASKS_PER_THREAD (1 << LOG_TASKS_PER_THREAD)
#define TASK_IDX_MASK (TASKS_PER_THREAD - 1)
#define MANAGER_SLEEP_USEC 10
#define THREAD_SLEEP_USEC 1

/**
 * @class TasksManager
 * @brief Class for managing parallel executions of small `Task`s which are child class of `TaskBase` described bellow.
 * 
 * The class manages a vector of `Worker`s, which are threads and additional required data members for executions of 
 * `Task`s. The `Task`s are split in a thread-pool fashion - finding free slot in the `Worker`s for the user to set up 
 * additional tasks, and fetching completed `Task`s back to the user.
 */
template<class Task>
class TasksManager {
public:
  /**
   * @brief Constructor for `TaskManager`.
   * @param nof_workers - number of workers / threads to be ran simultaneously
   */
  TasksManager(const int nof_workers) : workers(nof_workers) {}

  /**
   * @brief Get free slot to insert new task to be executed. This is a blocking function - until a free task is found.
   * @param task_slot - pointer to the internal task to allow the user to edit in the new task.
   * @return boolean value indication of the current content of the task - true indicates that `task_slot` holds a 
   * previous result that should be handled by the user before writing the new task.
   */
  bool get_free_task(Task*& task_slot);

  /**
   * @brief Get task that holds previous result to be handled by the user. This function blocks the code until a 
   * completed task is found or all tasks are idle with no result.
   * @param completed_task - pointer to the internal task holding a previous result. a nullptr is 
   * returned in case that all internal tasks are idle with no result (And as such no completed task will be found).
   * Again if using completed_task to assign additional tasks, the existing result must be handled before hand.
   */
  void get_completed_task(Task*& completed_task);
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
    void run();

    /**
     * @brief Get free slot to insert new task to be executed. This isn't a blocking function - it checks all worker's 
     * tasks and if no free one is found a nullptr is returned.
     * @param task_slot - pointer to the internal task of the worker to allow the user to edit in the new task.
     * @return boolean value indication of the current content of the task - true indicates that `task_slot` holds a 
     * previous result that should be handled by the user before writing the new task.
     */
    bool get_free_task(Task*& task_slot);

    /**
     * @brief Get task that holds previous result to be handled by the user. This isn't a blocking function - it checks 
     * all worker's tasks and if no completed one is found a nullptr is returned.
     * @param completed_task - pointer to the internal task holding a previous result. 
     * Again if using completed_task to assign additional tasks, the existing result must be handled before hand.
     */
    void get_completed_task(Task*& completed_task);

    /**
     * @brief Check if all worker's tasks are idle (free with no result to be handled)
     */
    bool are_all_idle();
  private:
    std::thread task_executor; // Thread to be run parallel to main
    std::vector<Task> mTasksFifo; // vector containing the worker's task. a Vector is used to allow buffering.
    int tail; // Tail (input) idx of the fifo above. Checks for free task start at this idx.
    int head; // Head (output) idx of the fifo above, the thread works on task at this idx.
    bool kill; // boolean to flag from main to the thread to finish.
  };

  std::vector<Worker> workers; // Vector of workers/thread to be ran simultaneously.
};

/**
 * @class TaskBase
 * @brief abstract base for the task supported by TasksManager above.
 */
class TaskBase {
public:
  /**
   * @brief constructor fo `TaskBase`.
   */
  TaskBase() : status(IDLE), father_fifo_tail(nullptr) {}

  /**
   * @brief function to be executed by `TasksManager`. Implemented by child class.
   */
  virtual void execute() = 0;

  /**
   * @brief Signal for the `Worker` owning the task that it is ready to be executed. 
   * Also increase's the workers tail idx.
   * User function.
   */
  void dispatch();

  // Getters and setter for the various states of the tasks `status`.
  inline bool is_ready_for_work(); // DISPATCHED
  inline bool can_push_task(); // IDLE or PENDING_RESULT
  inline bool has_result();
  inline bool is_idle();
  inline void set_working();
  inline void set_pending_result();
  inline void set_handled_result();

  /**
   * @brief link this task with the tail idx of the Worker it belongs to. Used only once after construction.
   */
  inline void link_father_tail(int* tail_pointer) { father_fifo_tail = tail_pointer; } 

  /**
   * @brief wait for a specific task to finish executing. This is a blocking function.
   */
  void wait_done(); // NOTE the user must handle this task's result before asking for new free tasks  

private:
  /**
   * @enum containing the valid states of a task.
   */
  enum eTaskStatus {IDLE, DISPATCHED, WORKING, PENDING_RESULT}; // TODO CAPS and eTaskStatus
  std::atomic<eTaskStatus> status; // current task state. Atomic to ensure proper sync between main and worker threads.
  int* father_fifo_tail; // pointer to father's tail.
};


bool TaskBase::is_ready_for_work() {
  return status.load(std::memory_order_acquire) == DISPATCHED;
}

bool TaskBase::can_push_task() {
  eTaskStatus curr_status = status.load(std::memory_order_acquire);
  return curr_status == PENDING_RESULT || curr_status == IDLE;
}

bool TaskBase::has_result() {
  return status.load(std::memory_order_acquire) == PENDING_RESULT;
}

bool TaskBase::is_idle() {
  return status.load(std::memory_order_acquire) == IDLE;
}

void TaskBase::dispatch() {
  (*father_fifo_tail)++;
  status.store(DISPATCHED, std::memory_order_release);
}

void TaskBase::set_working() {
  status.store(WORKING, std::memory_order_release);
}

void TaskBase::set_pending_result() {
  status.store(PENDING_RESULT, std::memory_order_release);
}

void TaskBase::set_handled_result() {
  status.store(IDLE, std::memory_order_release);
}

void TaskBase::wait_done() {
  while (!has_result()) {
    std::this_thread::sleep_for(std::chrono::microseconds(MANAGER_SLEEP_USEC));
  }
}

template<class Task>
TasksManager<Task>::Worker::Worker() 
: mTasksFifo(TASKS_PER_THREAD), 
  tail(0),
  head(0),
  kill(false)
  {
    // Tail idx linking can only be done after task and worker initialization
    for (Task& task : mTasksFifo) task.link_father_tail(&tail);
    // Init thread only after finishing all other setup to avoid data races
    task_executor = std::thread(&TasksManager<Task>::Worker::run, this);
  }

template<class Task>
TasksManager<Task>::Worker::~Worker() {
  kill = true;
  task_executor.join();
}

template<class Task>
void TasksManager<Task>::Worker::run() {
  while (true) {
    for (head = 0; head < mTasksFifo.size(); head++)
    {      
      Task* task = &mTasksFifo[head];
      if (!task->is_ready_for_work()) {
        // Sleep as the thread apparently isn't fully utilized currently
        std::this_thread::sleep_for(std::chrono::microseconds(THREAD_SLEEP_USEC));       
        continue;
      }
      task->set_working();
      task->execute();
      task->set_pending_result();
    }
    if (kill) return;
  }
}

template<class Task>
bool TasksManager<Task>::Worker::get_free_task(Task*& task_slot) {
  for (int i = 0; i < mTasksFifo.size(); i++)
  {
    // TASKS_PER_WORKER is a power of 2 so modulo is done via bitmask.
    int tail_adjusted_idx = (i + tail) & TASK_IDX_MASK; 

    if (mTasksFifo[tail_adjusted_idx].can_push_task()) // Either IDLE or PENDING_RESULT are valid.
    {
      task_slot = &mTasksFifo[tail_adjusted_idx];
      return task_slot->has_result();
    }
  }
  task_slot = nullptr;
  return false;
}

template<class Task>
void TasksManager<Task>::Worker::get_completed_task(Task*& task_slot) {
  for (int i = 0; i < mTasksFifo.size(); i++)
  {
    int tail_adjusted_idx = (i + tail) & TASK_IDX_MASK;

    if (mTasksFifo[tail_adjusted_idx].has_result()) // Only PENDING_RESULT is valid.
    {
      task_slot = &mTasksFifo[tail_adjusted_idx];
      return;
    }
  }
  task_slot = nullptr;
}

template<class Task>
bool TasksManager<Task>::Worker::are_all_idle()
{
  bool all_tasks_idle = true;
  for (Task& task : mTasksFifo) all_tasks_idle = all_tasks_idle && task.is_idle();
  return all_tasks_idle;
}

template<class Task>
bool TasksManager<Task>::get_free_task(Task*& task_slot) {
  bool has_task = false;
  do
  {
    for (Worker& worker : workers)
    {
      has_task = worker.get_free_task(task_slot);
      if (task_slot != nullptr) break;
    }
  } while (task_slot == nullptr);
  if (has_task) task_slot->set_handled_result(); // User will handle result outside before dispatching a new one.
  return has_task;
}

template<class Task>
void TasksManager<Task>::get_completed_task(Task*& completed_task) {
  completed_task = nullptr;
  bool all_tasks_idle = true;
  do
  {
    all_tasks_idle = true;
    for (Worker& worker : workers)
    {
      worker.get_completed_task(completed_task);
      if (completed_task != nullptr)
      {
        completed_task->set_handled_result(); // User will handle result outside.
        return;
      }

      all_tasks_idle = all_tasks_idle && worker.are_all_idle();
    }
  } while (!all_tasks_idle);
  // No completed tasks were found in the loop - return null.
  completed_task = nullptr;
}