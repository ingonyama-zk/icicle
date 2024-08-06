#ifndef TASKS_MANAGER
#define TASKS_MANAGER
#include <atomic>
#include <thread> // TODO check windows support

template<class Task>
class TasksManager {
public:
  // constructor
  TasksManager(const int nof_workers);

  // Get a task that isn't occupied
  // return value signals if there is a valid result to be handled before dispatching a new calculation
  // Blocking functions
  bool get_free_task(Task*& task_slot);
  void get_completed_task(Task*& completed_task); // NOTE the user must handle this task's result before asking for new free tasks
private:
  class Worker {
  public:
    Worker();
    ~Worker();
    void run();
    // Get a task that isn't occupied
    // return value signals if there is a valid result to be handled before dispatching a new calculation
    bool get_free_task(Task*& task_slot);
    void get_completed_task(Task*& completed_task); // NOTE the user must handle this task's result before asking for new free tasks
    bool are_all_idle();
  private:
    std::thread task_executor;
    std::vector<Task> mTasksFifo;
    int tail;
    int head;
    bool kill;

    friend TasksManager<Task>::TasksManager(const int nof_workers);
  };

  std::vector<Worker> workers;
};

class TaskBase {
  public:
    TaskBase() : status(idle), father_fifo_tail(nullptr) {}
    virtual void execute() = 0; // COMMENT should it be private? still needs to be friend of worker
    void dispatch(); // USER FUNC

    inline bool is_ready_for_work(); // TODO friend functions of worker
    inline bool can_push_task();
    inline bool has_result();
    inline bool is_idle();
    inline void set_working();
    inline void set_pending_result();
    inline void set_handled_result();
    // Blocking function
    void wait_done(); // NOTE the user must handle this task's result before asking for new free tasks  

  protected:
    enum TaskStatus {idle, set_task, dispatched, working, pending_result}; // TODO CAPS and eTaskStatus
    std::atomic<TaskStatus> status;
  private:
    int* father_fifo_tail;

    template <class Task>
    friend TasksManager<Task>::TasksManager(const int nof_workers);
};
#endif