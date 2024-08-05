#include "tasks_manager.hpp"
#include <cassert>
#include <stdexcept>

#define LOG_TASKS_PER_THREAD 2
#define TASKS_PER_THREAD (1 << LOG_TASKS_PER_THREAD)
#define TASK_IDX_MASK (TASKS_PER_THREAD - 1)
#define MANAGER_SLEEP_USEC 10
#define THREAD_SLEEP_USEC 1

inline bool TaskBase::is_ready_for_work() {
  return status.load(std::memory_order_acquire) == dispatched;
}

inline bool TaskBase::can_push_task() {
  TaskStatus curr_status = status.load(std::memory_order_acquire);
  return curr_status == pending_result || curr_status == idle;
}

inline bool TaskBase::has_result() {
  return status.load(std::memory_order_acquire) == pending_result;
}

inline bool TaskBase::is_idle() {
  return status.load(std::memory_order_acquire) == idle;
}

void TaskBase::dispatch() {
  assert(can_push_task());
  (*father_fifo_tail)++;
  status.store(dispatched, std::memory_order_release);
}

inline void TaskBase::set_working() {
  assert(is_ready_for_work());
  status.store(working, std::memory_order_release);
}

inline void TaskBase::set_pending_result() {
  assert(status.load(std::memory_order_acquire) == working);
  status.store(pending_result, std::memory_order_release);
}

inline void TaskBase::set_handled_result() {
  assert(has_result());
  status.store(idle, std::memory_order_release);
}

void TaskBase::wait_done() {
  while (has_result()) {
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
    int tail_adjusted_idx = (i + tail) & TASK_IDX_MASK; // TODO check % is optimized when base is power of 2

    if (mTasksFifo[tail_adjusted_idx].can_push_task()) // Either idle or pending_result are valid
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

    if (mTasksFifo[tail_adjusted_idx].has_result())
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
TasksManager<Task>::TasksManager(const int nof_workers) 
: workers(nof_workers) 
{
  // Link tasks with their fifo idx
  for (Worker& worker : workers)
  {
    for (Task& task : worker.mTasksFifo)
    {
      task.father_fifo_tail = &worker.tail; 
    }
  }
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
  if (has_task) task_slot->set_handled_result();
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
        completed_task->set_handled_result();
        return;
      }

      all_tasks_idle = all_tasks_idle && worker.are_all_idle();
    }
  } while (!all_tasks_idle);
  // No with_taskd tasks were found in the loop - no complete tasks left to be handled
  completed_task = nullptr;
}