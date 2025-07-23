# Rust FFI Bindings for Program

:::note
Please refer to the [Program overview](api/cpp/program.md) page for additional detail. This section is a brief description of the Rust FFI bindings.
:::

This documentation is designed to bring developers up to speed about the Rust API wrapping the cpp implementation of program.

## Introduction
Program is a class that lets users define expressions on vector elements, and have ICICLE compile it for the backends for a fused implementation. This solves memory bottlenecks and also lets users customize algorithms such as sumcheck. Program can create only element-wise lambda functions.

The following lists the implemented Rust functionality with some examples paralleling those given in the [original program overview](api/cpp/program.md).

# Symbol
Symbol is the basic (template) class that allows users to define their own program, representing an arithmetic operation. The [function](#defining-a-function-for-program) the user defines will operate on symbols.

## `Symbol` Trait Definition
The trait defines the functionality required by the user. The expected use-case of symbol is solely to be operated on to create the final arithmetic operation, which is reflected in the implemented functions and traits.

```rust
pub trait Symbol<T: IntegerRing>:
    Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Add<T, Output = Self>
    + Sub<T, Output = Self>
    + Mul<T, Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + AddAssign<T>
    + SubAssign<T>
    + MulAssign<T>
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
    + for<'a> AddAssign<&'a Self>
    + for<'a> SubAssign<&'a Self>
    + for<'a> MulAssign<&'a Self>
    + Clone
    + Copy
    + Sized
    + Handle
{
    fn new_input(in_idx: u32) -> Result<Self, IcicleError>; // New input symbol for the execution function
    fn from_constant(constant: T) -> Result<Self, IcicleError>; // New symbol from a ring element
}
```

## `RingSymbol` Struct
The `RingSymbol` struct is implemented for each of the supported Icicle fields, providing an implementation of the above trait for each specific field. The distinction between fields is relevant for input symbols and stored constants in the program. At its core, it is simply a handle to the C++ implementation. 

:::note
Despite the name, `RingSymbol` can be instantiated for both rings and fields.
:::

```rust
pub struct RingSymbol {
    m_handle: SymbolHandle,
}
```

### Traits implemented and key methods
Additional traits the struct implements to fulfill `Symbol<T>` trait that should be noted.

#### Arithmetic operations
`RingSymbol` implements addition, subtraction, and multiplication (as well as the assign variants of them) with other symbols/references as well as field elements. Applying the operations will generate a new symbol (or overwrite the existing in the case of the assign operation) representing the arithmetic operations of the two operand symbols.

# Program
A program to be run on the various Icicle backends. It can be either a user-defined program, or one of the members of `PreDefinedProgram` enum. The program adheres to one of the following traits:

## `ProgramImpl` Trait Definition
The trait defines the base functionality required for the user, which in this case is creation and execution. It is used as a program for running a function that takes a vector of field elements (both inputs and outputs) and has no return value (output is written to the given vector). It is executed through Vector Operations.

```rust
pub trait ProgramImpl<T>: Sized + Handle
where
    T: IntegerRing,
{
    type ProgSymbol: Symbol<T>;

    fn new(program_func: impl FnOnce(&mut Vec<Self::ProgSymbol>), nof_parameters: u32) -> Result<Self, IcicleError>;

    fn new_predefined(pre_def: PreDefinedProgram) -> Result<Self, IcicleError>;

    fn execute_program<Data>(&self, data: &mut Vec<&Data>, cfg: &VecOpsConfig) -> Result<(), IcicleError>
    where
        Data: HostOrDeviceSlice<T> + ?Sized;
}
```

## `ReturningValueProgramImpl` Trait Definition
This trait is for programs that return a value (symbol) as output.

```rust
pub trait ReturningValueProgramImpl: Sized + Handle {
    type Ring: IntegerRing;
    type ProgSymbol: Symbol<Self::Ring>;

    fn new(
        program_func: impl FnOnce(&mut Vec<Self::ProgSymbol>) -> Self::ProgSymbol,
        nof_parameters: u32,
    ) -> Result<Self, IcicleError>;

    fn new_predefined(pre_def: PreDefinedProgram) -> Result<Self, IcicleError>;
}
```

## `Program` Struct
The `Program` struct is implemented for each of the supported icicle fields, implementing the above trait for the specific field (field distinction is relevant for the input symbols and stored constants in the program). At its core, it's just a handle to the cpp implementation.

```rust
pub struct Program {
    m_handle: ProgramHandle,
}
```

# Usage
This section will outline how to use Program and Symbol, mirroring the examples from the [cpp overview](api/cpp/program.md). The program use-case splits to three steps:
1. Defining a function/lambda that describes the program to be run (or choosing one of the predefined list).
2. Creating a new program given the above function.
3. Executing the program using the Vector Operations API.

## Defining a Function for Program
A function operating on a vector of symbols, with outputs being written to said input vector. The input symbols in the vector represent inputs and outputs of field elements, and will be replaced by vectors of field elements when executed.

:::note
The defined function defines arithmetic operations to be done in series, and could be represented as a set of equations (for each output). Practically, control flow (e.g., loops, conditions) is not parsed, instead the computation follows the exact execution path taken during tracing, which determines the final computation that will be performed.
:::

```rust
fn example_function<T, S>(vars: &mut Vec<S>)
where
    T: IntegerRing,
    S: Symbol<T>,
{
    let a = vars[0];
    let b = vars[1];
    let c = vars[2];
    let eq = vars[3];

    vars[4] = eq * (a * b - c) + T::from_u32(9);
    vars[5] = a * b - c; // For division, use an Invertible trait if implemented
    vars[6] = vars[5];
    vars[3] = (vars[0] + vars[1]) * T::from_u32(2); // all variables can be both inputs and outputs
}
```

## Creating a Program
Applying the constructor with the lambda:

```rust
let program = Program::new(example_function, 7 /* number of parameters for lambda = vars.len() */)?;
```

## Executing the Program
Execution is done through the appropriate vecops function:

```rust
program.execute_program(&mut parameters, &cfg)?;
```

Where `parameters` is a `Vec<&Parameter>` and `cfg` is a `VecOpsConfig`.

### Examples
Example taken from check_program in vec_ops tests.

#### Program functionality with a custom function
```rust
pub fn check_program<T, Prog>()
where
    T: IntegerRing,
    Prog: ProgramImpl<T>,
{
    let example_lambda = |vars: &mut Vec<Prog::ProgSymbol>| {
        let a = vars[0];
        let b = vars[1];
        let c = vars[2];
        let d = vars[3];

        vars[4] = d * (a * b - c) + T::from_u32(9);
        vars[5] = a * b - c; // For division, use an Invertible trait if implemented
        vars[6] = vars[5];
        vars[3] = (vars[0] + vars[1]) * T::from_u32(2);
    };

    // Additional lines for initiating the slices of field elements for the parameters

    let mut parameters = vec![a_slice, b_slice, c_slice, eq_slice, var4_slice, var5_slice, var6_slice];

    let program = Prog::new(example_lambda, 7).unwrap();

    let cfg = VecOpsConfig::default();
    program.execute_program(&mut parameters, &cfg).expect("Program Failed");
}
```

#### Program functionality with predefined programs
```rust
pub fn check_predefined_program<T, Prog>()
where
    T: IntegerRing,
    Prog: ProgramImpl<T>,
{
    // Additional lines for initiating the slices of field elements for the parameters
    let mut parameters = vec![a_slice, b_slice, c_slice, eq_slice, var4_slice];

    let program = Prog::new_predefined(PreDefinedProgram::EQtimesABminusC).unwrap();

    let cfg = VecOpsConfig::default();
    program.execute_program(&mut parameters, &cfg).expect("Program Failed");
}
```