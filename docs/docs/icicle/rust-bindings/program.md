# Rust FFI Bindings for Program

>**_NOTE:_**
>Please refer to the [Program overview](../primitives/program.md) page for additional detail. This section is a brief description of the Rust FFI bindings.

This documentation is designed to bring developers up to speed about Rust API for program implemented in the cpp backend.

## Introduction
Program is a class that let users define expressions on vector elements, and have ICICLE compile it for the backends for a fused implementation. This solves memory bottlenecks and also let users customize algorithms such as sumcheck. Program can create only element-wise lambda functions. Program itself works for definition while actual execution is handled through other functionalities like [Vector Operations](./vec-ops.md).


The Rust FFI bindings for both Program and Symbol serve as a "shallow wrapper" around the underlying C++ implementation. These bindings provide a straightforward Rust interface that directly calls functions from a C++ library, effectively bridging Rust and C++ operations. The Rust layer handles simple interface translations without delving into complex logic or data structures, which are managed on the C++ side. This design ensures efficient data handling, memory management, and execution while utilizing the existing backend directly via C++.

The following would list the implemented Rust functionality with some examples paralleling those given in the [original program overview](../primitives/program.md).
# Symbol
Symbol is the basic (template) class that allow users to define their own program, representing an arithmetic operation. The [function](#defining-a-function-for-program) the user define will operate on symbols.
## `Symbol` Trait Definition
The trait defines the functionality required by the user. The expected use-case of symbol is solely to be operated on to create the final arithmetic operation, which is reflected implemented functions and traits.
```rust
pub trait Symbol<F: FieldImpl>:
  Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> +
  Add<F, Output = Self> + Sub<F, Output = Self> + Mul<F, Output = Self> +
  AddAssign + SubAssign + MulAssign + AddAssign<F> + SubAssign<F> + MulAssign<F> +
  for<'a> Add<&'a Self, Output = Self> +
  for<'a> Sub<&'a Self, Output = Self> +
  for<'a> Mul<&'a Self, Output = Self> +
  for<'a> AddAssign<&'a Self> +
  for<'a> SubAssign<&'a Self> +
  for<'a> MulAssign<&'a Self> +
  Clone + Copy + Sized + Handle
{
  fn new_input(in_idx: u32) -> Result<Self, eIcicleError>;      // New input symbol for the execution function
  fn from_constant(constant: F) -> Result<Self, eIcicleError>;  // New symbol from a field element

  fn inverse(&self) -> Self; // Field inverse of the symbol
}
```
## `Symbol` Struct
The `Symbol` struct is implemented for each of the supported icicle fields, implementing the above trait for the specific field (field distinction is relevant for the input symbols and stored constants in the program). In its core it's just a handle to the cpp implementation.
```rust
pub struct Symbol {
  handle: SymbolHandle,
}
```
### Traits implemented and key methods
Additional traits the struct implements to fulfil `Symbol<F>` trait that should be noted.
#### Arithmetic operations
Symbol implements addition, subtraction and multiplication (as well as the assign variants of them) with other symbols / references as well as field elements. Applying the operations will generate a new symbol (Or overwrite the existing in the case of the assign operation) representing the arithmetic operations of the two operand symbols. The `inverse` function joins these operations to allow an additional arithmetic operation (division).
#### `Handle` 
Trait to guarantee linking the symbol to the appropriate cpp backend - providing a function to access the pointer to the backend implementation.

# Program
A program to be ran on the various Icicle backends. It can be either a user-defined program, or one of the members of `PredefinedProgram` enum. The program adheres to one of the following traits:
## `Program` Trait Definition
The trait defines the base functionality required for the user, which in this case is only creation (The execution functionality is exposed through Vector Operations). It is used as a program for running a function that takes a vector of field elements (both inputs and outputs) and has no return value (output is written to the given vector). It is executed through Vector Operations.
```rust
pub trait ReturningValueProgram<F>:
  Sized + Handle
where
  F:FieldImpl,
{
  type ProgSymbol: Symbol<F>;

  fn new(program_func: impl FnOnce(&mut Vec<Self::ProgSymbol>) -> Self::ProgSymbol, nof_parameters: u32) -> Result<Self, eIcicleError>;

  fn new_predefined(pre_def: PreDefinedProgram) -> Result<Self, eIcicleError>;
}
```
## `Program` Struct
The `Program` struct is implemented for each of the supported icicle fields, implementing the above trait for the specific field (field distinction is relevant for the input symbols and stored constants in the program). In its core it's just a handle to the cpp implementation.
```rust
pub struct Program {
  handle: ProgramHandle,
}
```
### Traits implemented and key methods
Additional trait the struct implements to fulfil `Program<F>` trait.
#### `Handle` 
Trait to guarantee linking the program to the appropriate cpp backend - providing a function to access the pointer to the backend implementation.
### `Drop`
Ensures proper resource management by releasing the backend allocated memory when a `Program` instance goes out of scope. This prevents memory leaks and ensures that resources are cleaned up correctly, adhering to Rust's RAII (Resource Acquisition Is Initialization) principles.

# Usage
This section will outline how to use Program and Symbol, mirroring the examples from the [cpp overview](../primitives/program.md). The program use-case splits to three steps:
1. Defining a function/lambda that describes the program to be ran (or choosing one of the predefined list?).
2. Creating a new program given the above function.
3. Executing the program using the Vector Operations API.
## Defining a Function for Program
A function operating on a vector of symbols, with outputs being written to said input vector. The input symbols in the vector represent inputs and outputs of field elements, and will be replaced by vectors of field elements when executed.
>**_NOTE:_**
> The defined function defines arithmetic operations to be done in series, without control-flow i.e. loops, conditions etc.
```rust
example_function<F, S>(vars: &mut Vec<S>)
where
  F: FieldImpl,
  S: Symbol<F>,
{
  let a = vars[0];
  let b = vars[1];
  let c = vars[2];
  let eq = vars[3];

  vars[4] = eq * (a * b - c) + F::from_u32(9);
  vars[5] = a * b - c.inverse();
  vars[6] = vars[5];
  vars[3] = (vars[0] + vars[1]) * F::from_u32(2); // all variables can be both inputs and outputs
}
```
## Creating a Program
Applying the constructor with the lambda:
```rust
let program = Program::new(example_lambda, 7 /*nof parameters for lambda = vars.size()*/);
```
## Executing the Program
Execution is done through the appropriate vecops function.
```rust
pub fn execute_program<F, Prog, Parameter>(
    data: &mut Vec<&Parameter>,
    program: &Prog,
    cfg: &VecOpsConfig
) -> Result<(), eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F>,
    Parameter: HostOrDeviceSlice<F> + ?Sized,
    Prog: Program<F> + Handle,
```

And in total with data setup we would get code like this (Example taken from check_program in vec_ops tests):
```rust
pub fn use_program<F, Prog>()
where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F> + GenerateRandom<F> + FieldArithmetic<F>,
    Prog: Program<F>,
{
    let example_lambda = |vars: &mut Vec<Prog::ProgSymbol>| {
        let a = vars[0]; // Shallow copies pointing to the same memory in the backend
        let b = vars[1];
        let c = vars[2];
        let d = vars[3];
    
        vars[4] = d * (a * b - c) + F::from_u32(9);
        vars[5] = a * b - c.inverse();
        vars[6] = vars[5];
        vars[3] = (vars[0] + vars[1]) * F::from_u32(2); // all variables can be both inputs and outputs
    };

    // Additional lines for initiating the slices of field elements for the parameters

    let mut parameters = vec![a_slice, b_slice, c_slice, eq_slice, var4_slice, var5_slice, var6_slice];
    
    let program = Prog::new(example_lambda, 7).unwrap();
    
    let cfg = VecOpsConfig::default();
    execute_program(&mut parameters, &program, &cfg).expect("Program Failed");
}
```