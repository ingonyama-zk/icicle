# How to Register a New Function

For users to be able to use a backend function (whenever it's a CPU/GPU one), the function should be exposed to the user and be register to the matching backend. We do not let the user call the backend function directly, instead a wrapping function, with identical signature to all backend, is created as a user-facing function. Then, the backend specific functions are registered to their respective beckends in such a way that a dispatcher will know to call the appropriate function (acoorfing to the device used) whenever the user facing function is used.
Here is a step-by-step explanation of how to do that registration process right.

## The backend specific function
The first step is to write the backend specific function. You should notice that its first argument must be `Device`. For example:
```cpp
template <typename T>
eIcicleError cpu_execute_program(const Device& device, std::vector<T*>& data, Program<T>& program, uint64_t size, const VecOpsConfig& config)
```
The function does not have to use the `device` argunemt, but is should be there.

## Define the macro
Now the macro which do the registration should be defined. It should be defined at the `include/icicle/backend` directory at the appropriate header file.
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

### Define the registration function
The registration function (`register_execute_program` at the example above) should be defined just before the macro:
```cpp
  void register_execute_program(const std::string& deviceType, programExecutionImpl);
```
It should have `void` as it return type and its name should be `register_<user-facing-function's-name>` (it can be inferred that the user facing function of the example above is `execute_program`). Wrong function name here will cause a linking error at compilation time.

### Define function signature
Now the function signature should be defined (the `programExecutionImpl` at the example above). It should be the signature of the backend specific functions, and should be defined at the same file as the registration macro.
At our example, it will be:
```cpp
  using programExecutionImpl = std::function<eIcicleError(
    const Device& device, std::vector<scalar_t*>& data, Program<scalar_t>& program, uint64_t size, const VecOpsConfig& config)>;
```

## Declare the user-facing function
Now the user facing function should be defined at the appropriate header file (the header file the user should include to use the function).

```cpp
  template <typename T>
  eIcicleError execute_program(std::vector<T*>& data, Program<T>& program, uint64_t size, const VecOpsConfig& config);
```
Pay attention - here `device` should not be an argument of the function.
Do not forget to add a doc string to this function.

## Create a dispatcher
Now you should create the dispatcher. It should be placed at C++ file in `icicle/source`. This is how the dispatcher should looks like:
```cpp
  ICICLE_DISPATCHER_INST(ExecuteProgramDispatcher, execute_program, programExecutionImpl)

  extern "C" eIcicleError CONCAT_EXPAND(FIELD, execute_program)(
    std::vector<scalar_t*>& data, Program<scalar_t>& program, uint64_t size, const VecOpsConfig& config)
  {
    return ExecuteProgramDispatcher::execute(data, program, size, config);
  }

  template <>
  eIcicleError
  execute_program(std::vector<scalar_t*>& data, Program<scalar_t>& program, uint64_t size, const VecOpsConfig& config)
  {
    return CONCAT_EXPAND(FIELD, execute_program)(data, program, size, config);
  }
```

## Call the macro
The last step is to call the macro we created earlier.
```cpp
REGISTER_EXECUTE_PROGRAM_BACKEND("CPU", cpu_execute_program<scalar_t>);
```
This macro call should come after the backend function you implemented.
