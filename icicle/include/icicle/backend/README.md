# How to Register a New Function

For users to be able to use a backend function (whenever it's a CPU/GPU one), the function should be exposed to the user and be registered to the matching backend. We do not let the user call the backend function directly, instead a wrapping function, with identical signature to all backend, is created as a frontend (user-facing) function. Then, the backend specific functions are registered to their respective backends in such a way that a dispatcher will know to call the appropriate function (according to the device used) whenever the frontend function is used.
Here is a step-by-step explanation of how to do that registration process right.

## Declare the frondend API
First the frondend API should be defined at the appropriate header file (the header file the user should include to use the function). In this example the file is vec_ops.h.

```cpp
  template <typename T>
  eIcicleError execute_program(std::vector<T*>& data, Program<T>& program, uint64_t size, const VecOpsConfig& config);
```
Pay attention - unlike at the next steps, here `device` should not be an argument of the function.
Do not forget to add a doc string to this function.

## Define the macro and registration function
Before we define the macro there are two preliminary steps we should take care of.

First the function signature should be defined. It should be the signature of the backend specific functions, and should be defined at the same file as the registration macro. This file should be in the `include/icicle/backend` directory at the appropriate header file (here it can be found at vec_ops_backend.h).
In our example, the function signature will be:
```cpp
  using programExecutionImpl = std::function<eIcicleError(
    const Device& device, std::vector<scalar_t*>& data, Program<scalar_t>& program, uint64_t size, const VecOpsConfig& config)>;
```
Here `device` should be an argument to the function.

Last thing before defining the macro is declaring the registration function. It should be defined at the same file as the function signature, just before the macro:
```cpp
  void register_execute_program(const std::string& deviceType, programExecutionImpl);
```
It should have `void` as it return type and its name should be `register_<frontend_API's-function-name>` (in our example the frontend API's function is `execute_program`). Wrong function name here will cause a linking error at compilation time.

Now the macro which do the registration should be defined.
The macro should look like that:
```cpp
#define REGISTER_EXECUTE_PROGRAM_BACKEND(DEVICE_TYPE, FUNC)                                                            
  namespace {                                                                                                          
    static bool UNIQUE(_reg_program_execution) = []() -> bool {                                                        
      register_execute_program(DEVICE_TYPE, FUNC);                                                                     
      return true;                                                                                                     
    }();                                                                                                               
  }
```

## Create a dispatcher
The last step of preparations is to create the dispatcher. It should be placed at C++ file in `icicle/source` (vec_ops.cpp in our example). This is how the dispatcher should looks like:
```cpp
  ICICLE_DISPATCHER_INST(ExecuteProgramDispatcher, execute_program, programExecutionImpl)

  extern "C" eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, execute_program)(
    std::vector<scalar_t*>& data, Program<scalar_t>& program, uint64_t size, const VecOpsConfig& config)
  {
    return ExecuteProgramDispatcher::execute(data, program, size, config);
  }

  template <>
  eIcicleError
  execute_program(std::vector<scalar_t*>& data, Program<scalar_t>& program, uint64_t size, const VecOpsConfig& config)
  {
    return CONCAT_EXPAND(ICICLE_FFI_PREFIX, execute_program)(data, program, size, config);
  }
```

## The backend specific function
Now we finally get to write the actual backend specific implementation for the fronend's function. You should notice that its first argument must be `Device`. For example:
```cpp
template <typename T>
eIcicleError cpu_execute_program(const Device& device, std::vector<T*>& data, Program<T>& program, uint64_t size, const VecOpsConfig& config)
```
The function does not have to use the `device` argunemt, but is should be there.
After writing the backend function we register our function to the corresponding backend by calling the macro we created:
```cpp
REGISTER_EXECUTE_PROGRAM_BACKEND("CPU", cpu_execute_program<scalar_t>);
```
This macro call should come after the backend function you implemented (in this example this happened in cpu_vec_ops.cpp).
