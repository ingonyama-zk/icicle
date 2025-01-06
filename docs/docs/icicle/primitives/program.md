# Programs

## Overview

Program is a class that let users define expressions on vector elements, and have ICICLE compile it for the backends for a fused implementation. This solves memory bottlenecks and also let users customize algorithms such as sumcheck. Program can create only element-wise lambda functions.


## C++ API

### Symbol

Symbol is the basic (template) class that allow users to define their own program. The lambda function the user define will operate on symbols.

### Defining lambda function

To define a custom lambda function the user will use Symbol:
```cpp
void lambda_multi_result(std::vector<Symbol<scalar_t>>& vars)
{
  const Symbol<scalar_t>& A = vars[0];
  const Symbol<scalar_t>& B = vars[1];
  const Symbol<scalar_t>& C = vars[2];
  const Symbol<scalar_t>& EQ = vars[3];
  vars[4] = EQ * (A * B - C) + scalar_t::from(9);
  vars[5] = A * B - C.inverse();
  vars[6] = vars[5];
}
```

Each symbol element at the vector argument `var` represent an input or an output. The type of the symbol (`scalar_t` in this example) will be the type of the inputs and outputs. In this example we created a lambda function with four inputs and three outputs.

In this example there are four input variables and three three outputs. Inside the function the user can define custom expressions on them.

Program support few pre-defined programs. The user can use those pre-defined programs without creating a lambda function, as will be explained in the next section.

### Creating program

To execute the lambda function we just created we need to create a program from it.
To create program from lambda function we can use the following constructor:\

```cpp
Program(std::function<void(std::vector<Symbol<S>>&)> program_func, const int nof_parameters)
```

`program_func` is the lambda function (in the example above `lambda_multi_result`) and `nof_parameters` is the total number of parameter (inputs + outputs) for the lambda (seven in the above example).

#### Pre-defined programs

As mentioned before, there are few pre-defined programs the user can use without the need to create a lambda function first. The enum `PreDefinedPrograms` contains the pre-defined program. Using pre-defined function will lead to better performance compared to creating the equivalent lambda function.
To create a pre-defined program a different constructor is bing used:

```cpp
Program(PreDefinedPrograms pre_def)
```

`pre_def` is the pre-defined program (from `PreDefinedPrograms`).

##### PreDefinedPrograms

```cpp
enum PreDefinedPrograms {
  AB_MINUS_C = 0,
  EQ_X_AB_MINUS_C
};
```

`AB_MINUS_C` - the pre-defined program `AB - C` for the input vectors `A`, `B` and `C`

`EQ_X_AB_MINUS_C` - the pre-defined program `EQ(AB - C)` for the input vectors `A`, `B`, `C` and `EQ`


### Executing program

To execute the program the `execute_program` function from the vector operation API should be used. This operation is supported by the CPU and CUDA backends.


```cpp
template <typename T>
eIcicleError
execute_program(std::vector<T*>& data, const Program<T>& program, uint64_t size, const VecOpsConfig& config);
```

The `data` vector is a vector of pointers to the inputs and output vectors, `program` is the program to execute, `size` is the length of the vectors and `config` is the configuration of the operation.

For the configuration the field `is_a_on_device` determined whethere the data (*inputs and outputs*) is on device or not. After the execution `data` will reside in the same place as it did before (i.e. the field `is_result_on_device` is irrelevant.)

> **_NOTE:_**  Using program for executing lambdas is recommended only while using the CUDA backend. Program's primary use is to let users to customize algorithms (such as sumcheck). 
