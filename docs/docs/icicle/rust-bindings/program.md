# Rust FFI Bindings for Program

:::note
Please refer to the [Program overview](../primitives/program.md) page for a deep overview. This section is a brief description of the Rust FFI bindings.
:::

This documantaion is designed to bring developers up to speed about Rust API for program implemented in the cpp backend.

## Introduction

The Rust FFI bindings for both Program ans Symbol serve as a "shallow wrapper" around the underlying C++ implementation. These bindings provide a straightforward Rust interface that directly calls functions from a C++ library, effectively bridging Rust and C++ operations. The Rust layer handles simple interface translations without delving into complex logic or data structures, which are managed on the C++ side. This design ensures efficient data handling, memory management, and execution while utilizing the existing backend directly via C++.

The following would list the implemented Rust functionality with some examples paralleling those given in the [original program overview](../primitives/program.md).
# Symbol
Symbol is the basic (template) class that allow users to define their own program, representing an arithmetic operation. The [lambda function](#defining-a-lambda-function) the user define will operate on symbols.
## `Symbol` Trait Definition
The trait defines the functionality required by the user. The expected use-case of symbol is solely to be operated on to create the final arithmetic operation, which is reflected implemented functions and traits.
```rust
pub trait Symbol<F: FieldImpl>:
  // Operator overloading for Symbol op Symbol, Symbol op Field scalar, Symbol op &Symbol
  Add + Sub + Mul + AddAssign + SubAssign + MulAssign + 
  Add<F, Output = Self> + Sub<F, Output = Self> + Mul<F, Output = Self> +
  AddAssign<F> + SubAssign<F> + MulAssign<F> +
  for<'a> Add<&'a Self, Output = Self> +
  for<'a> Sub<&'a Self, Output = Self> +
  for<'a> Mul<&'a Self, Output = Self> +
  for<'a> AddAssign<&'a Self> +
  for<'a> SubAssign<&'a Self> +
  for<'a> MulAssign<&'a Self> +
  Clone + Sized
{
  // Create a Symbol representing a constant from a field scalar
  fn from_constant(constant: F) -> Result<Self, eIcicleError>;

  // Get field inverse of the given Symbol
  fn inverse(&self) -> Self;

  // Set symbol as input (Which is required for correct Program generation)
  fn set_as_input(&self, in_index: u32);
}
```
## `Symbol` Struct
The `Symbol` sturct is implemented for each of the supported icicle fields, implementing the above trait for the specific field (field distinction is relevant for the input symbols and stored constants in the program). In its core it's just a handle to the cpp implementation.
```rust
pub struct Program {
  handle: ProgramHandle,
}
```
### Traits implemented and key methods
Additional trait the struct implements besides the `Symbol<F>` trait.
#### `Handle` 
Trait to guarantee linking the symbol to the appropriate cpp backend - providing a function to access the pointer to the backend implementation.
### `Drop`
Ensures proper resource management by releasing the backend allocated memory when a `Symbol` instance goes out of scope. This prevents memory leaks and ensures that resources are cleaned up correctly, adhering to Rust's RAII (Resource Acquisition Is Initialization) principles.

# Program
A configurebale program to be ran on the various Icicle backends. It can be one of the members of `PredefinedProgram` enum. The program adheres to the following trait:
## `Program` Trait Definition
The trait defines the base functionality required for the user, which in this case is only creation (The other execution functionality is exposed through Vector Operations).
```rust
pub trait Program<F: FieldImpl>: {
  fn new<S: Symbol<F>>(program_func: impl FnOnce(&mut Vec<S>), nof_parameters: u32) -> Result<Self, eIcicleError>;

  fn new_predefined(pre_def: PreDefinedProgram) -> Result<Self, eIcicleError>;
}
```
## `Program` Struct
The `Program` sturct is implemented for each of the supported icicle fields, implementing the above trait for the specific field (field distinction is relevant for the input symbols and stored constants in the program). In its core it's just a handle to the cpp implementation.
```rust
pub struct Program {
  handle: ProgramHandle,
}
```
### Traits implemented and key methods
Additional trait the struct implements besides the `Program<F>` trait.
#### `Handle` 
Trait to guarantee linking the program to the appropriate cpp backend - providing a function to access the pointer to the backend implementation.
### `Drop`
Ensures proper resource management by releasing the backend allocated memory when a `Program` instance goes out of scope. This prevents memory leaks and ensures that resources are cleaned up correctly, adhering to Rust's RAII (Resource Acquisition Is Initialization) principles.

# Usage
This section will outline how to use Program and Symbol, mirroring the examples from the [cpp overview](../primitives/program.md). The program use-case splits to three steps:
1. Defining a function/lambda that describes the program to be ran (or choosing one of the predefined list?).
2. Creating a new program given the above function.
3. Executing the program using the Vector Operations API.
## Defining a Function
```rust
example_function<F, S>(vars: &mut Vec<S>)
where
  F: FieldImpl, // TODO do we need both in types
  S: Symbol<F>,
  &'a S: SymbolRef<F, S>,
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
One thing to note when comparing this to the cpp version, is the abundance of `clone` calls - due to Symbol not being copyable and Rust's ownership it is required in most operations.
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

And in total with data setup we would get code like this:
```rust
const SIZE: usize = 1 << 10;
let a = F::Config::generate_random(SIZE);
let b = F::Config::generate_random(SIZE);
let c = F::Config::generate_random(SIZE);
let eq = F::Config::generate_random(SIZE);
let var4 = vec![F::zero(); SIZE];
let var5 = vec![F::zero(); SIZE];
let var6 = vec![F::zero(); SIZE];
let a_slice = HostSlice::from_slice(&a);
let b_slice = HostSlice::from_slice(&b);
let c_slice = HostSlice::from_slice(&c);
let eq_slice = HostSlice::from_slice(&eq);
let var4_slice = HostSlice::from_slice(&var4);
let var5_slice = HostSlice::from_slice(&var5);
let var6_slice = HostSlice::from_slice(&var6);
let mut parameters = vec![a_slice, b_slice, c_slice, eq_slice, var4_slice, var5_slice, var6_slice];

let program = Program::new(example_lambda, 7).unwrap();

let cfg = VecOpsConfig::default();
execute_program(&mut parameters, &program, &cfg).expect("Program Failed");
```