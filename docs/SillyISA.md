# SillyISA: Overview & Design (Highly WIP)

## Introduction

SillyISA is the custom neuro-symbolic ISA for SillyAI, allowing for formalization of proofs and steps to solving problems. By combining novel methods with existing bytecode virtual machine technologies, SillyISA seeks to refine and optimize reasoning and problem-solving capabilities found in SillyAI.

## High-Level Overview

SillyISA combines typical instructions and features found in load-store architectures and VLIW/EPIC (where every instruction is explicitly parallelized), though in SillyISA's case all instructions and routines are ran asynchronously to prevent blocking and increase throughput. Lastly, SillyISA has instructions related to SillyAI-specific features such as concept graph manipulation and neural representations.

## Registers

SillyISA has the typical GPRs (`R0`-`Rn`), vector registers (`V0-Vn`), as well as concept registers (`C0`-`Cn`), an instruction pointer (`ip`), a state register (`sr`), a stack pointer (`sp`), a base pointer (`bp`), and a frame pointer (`fp`).

## Memory

SillyISA segments memory into three areas: concept memory (high), data memory (middle), and code (low), with optional tagged access enforcement, per-zone protection flags, and dynamic resizing to accommodate growing abstractions or runtime demands; concept memory may additionally support lazy paging, logical compaction, or persistent mapping for symbolic graphs, while data memory can employ NaN-boxing or region-based allocation to enhance performance and safety under high-throughput workloads.

## Instruction List

| **Mnemonic**   | **Category**         | **Description**                                                                 |
|----------------|----------------------|---------------------------------------------------------------------------------|
| `ASSERT`       | Control/Meta         | Asserts a condition or logical statement.                                      |
| `HLT`          | Control              | Halts execution.                                                               |
| `NOP`          | Control              | No-operation.                                                                  |
| `LOAD`         | Memory               | Loads a value from memory into a register.                                     |
| `LOADC`        | Concept Memory       | Loads a concept from concept memory.                                           |
| `STORE`        | Memory               | Stores a register value into memory.                                           |
| `STOREC`       | Concept Memory       | Stores a concept to concept memory.                                            |
| `PUSH`         | Stack                | Pushes a value onto the stack.                                                 |
| `POP`          | Stack                | Pops a value from the stack.                                                   |
| `MOV`          | Data Movement        | Moves a value from one register to another.                                    |
| `JM`           | Control Flow         | Unconditional jump to a specified address.                                     |
| `CJM`          | Control Flow         | Conditional jump based on a register or flag.                                  |
| `CALL`         | Control Flow         | Calls a routine or function.                                                   |
| `RET`          | Control Flow         | Returns from a routine, optionally with a value.                               |
| `WAIT`         | Timing               | Waits for a number of milliseconds.                                            |
| `SLEEP`        | Timing               | Sleeps or halts the thread for a duration.                                     |

### 🧮 Arithmetic & Math

| **Mnemonic**   | **Category**         | **Description**                                                                 |
|----------------|----------------------|---------------------------------------------------------------------------------|
| `ADD`          | Arithmetic            | Adds two values.                                                               |
| `SUB`          | Arithmetic            | Subtracts one value from another.                                              |
| `MUL`          | Arithmetic            | Multiplies two values.                                                         |
| `DIV`          | Arithmetic            | Divides one value by another.                                                  |
| `POW`          | Arithmetic            | Exponentiates a base to a power.                                               |
| `SQRT`         | Arithmetic            | Square root.                                                                   |
| `EXP`          | Arithmetic            | Exponential (e^x).                                                             |
| `SIN`          | Trigonometric         | Sine function.                                                                 |
| `COS`          | Trigonometric         | Cosine function.                                                               |
| `TAN`          | Trigonometric         | Tangent function.                                                              |
| `ASIN`         | Trigonometric         | Arcsine function.                                                              |
| `ACOS`         | Trigonometric         | Arccosine function.                                                            |
| `ATAN`         | Trigonometric         | Arctangent function.                                                           |
| `LOG`          | Math                  | Logarithm base 2.                                                              |
| `LOG10`        | Math                  | Logarithm base 10.                                                             |
| `LN`           | Math                  | Natural logarithm.                                                             |
| `ERR`          | Math                  | Error function (Gaussian integral approximation).                              |

### 🔁 Logical & Boolean Ops

| **Mnemonic**   | **Category**         | **Description**                                                                 |
|----------------|----------------------|---------------------------------------------------------------------------------|
| `AND`          | Logic                 | Logical conjunction.                                                           |
| `OR`           | Logic                 | Logical disjunction.                                                           |
| `NOT`          | Logic                 | Logical negation.                                                              |
| `XOR`          | Logic                 | Exclusive or.                                                                  |
| `NAND`         | Logic                 | Negated AND.                                                                   |
| `NOR`          | Logic                 | Negated OR.                                                                    |
| `IMPLIES`      | Logic                 | Logical implication.                                                           |
| `IFF`          | Logic                 | If-and-only-if (bi-implication).                                               |
| `FORALL`       | Symbolic Logic        | Universal quantification (∀).                                                  |
| `EXISTS`       | Symbolic Logic        | Existential quantification (∃).                                                |
| `UNIFY`        | Symbolic Logic        | Unify two terms/symbols under a shared binding.                                |
| `RESOLVE`      | Symbolic Logic        | Logical resolution step (e.g. for inference chaining).                         |
| `CONTRADICTS`  | Symbolic Logic        | Checks for contradiction in current logical frame.                             |

### 🧠 Concept Memory Ops

| **Mnemonic**   | **Category**         | **Description**                                                                 |
|----------------|----------------------|---------------------------------------------------------------------------------|
| `CORR`         | Concept Ops           | Correlates two concepts with a specified weight.                               |
| `ENER`         | Concept Ops           | Stores the energy of a concept to memory.                                      |
| `CASSERT`      | Concept Logic         | Asserts a concept relationship or state.                                       |
| `CBIND`        | Concept Logic         | Binds a logical or symbolic variable to a concept.                             |
| `CQUERY`       | Concept Logic         | Queries the concept graph for a specific structure/relation.                   |
| `CAND`         | Concept Logic         | Conceptual "AND" — inference condition.                                        |
| `COR`          | Concept Logic         | Conceptual "OR".                                                               |
| `CNOT`         | Concept Logic         | Negates or disables a concept node.                                            |
| `CIF`          | Concept Logic         | Triggers a concept if a condition is met.                                      |
| `CCMP`         | Concept Logic         | Compares two concepts for similarity, identity, or energy difference.          |
| `CLEAN`        | Concept Graph         | Prunes or resets the least-used portions of the concept memory.                |

## 🧬 SillyISA Data Types

To make the development experience nicer, SillyISA assembler uses static typing and comes with default data types for a variety of primitives:

| Type      | Description |
|-----------|-------------|
| `int`     | General integer (default platform word size) |
| `f32`     | 32-bit IEEE floating point |
| `f64`     | 64-bit IEEE floating point |
| `i8`–`i128` | Signed integers (8 to 128 bits) |
| `u8`–`u128` | Unsigned integers (8 to 128 bits) |
| `str`     | String type (immutable or ref-counted slice) |
| `bool`    | Boolean value (`true` or `false`) |
| `Prop`    | Logical proposition (used for reasoning, assertions) |
| `Conc`    | Concept representation (used in concept registers/memory) |
| `Vecn<T>` | Vector of dimension `n`, containing type `T` |
| `Matn<T>` | Square matrix `n x n`, of type `T` |

## 🧠 Design Highlights

| Feature | Implication |
|---------|-------------|
| **Type-agnostic GPRs** | GPRs are like raw registers in most real-world ISAs — they’re just slots. Typing is enforced in *routines and static checks*, not instruction semantics. |
| **Rich typing in routines** | Lets you write reusable, type-checked logic. Think like a type-aware macro system but still hardware-ish. |
| **Conceptual separation** | Execution is fast and dumb; analysis, optimization, and correctness happen above the ISA level. Clean separation. |
| **Concept/Prop special types** | Enables deep symbolic reasoning and neural-symbolic bridges (like propagating concept graphs or evaluating logical statements). |

## Routines and Variables

SillyISA supports routines, which can take parameters and return values, the basic structure of a routine looks like this:

```
.MyRoutine(a: r1<int>, b: r2<int>) {
    // Do stuff, for our case we'll just add the numbers
    RET a + b
}

MyRoutine(1, 2)
```

In addition to this, variables are also supported as aliases to registers. This allows for more expressive and higher-level code, all while maintaining low-level precision and control:

```
.AddAndMultiply(a: r0<int>, b: r1<int>, c: r2<int>) {
    %result: r3<int>
    result = (a + b) * c
    RET result
}
```

## Example of Bytecode: Pythagorean Theorem Proof

```
.Proof() {
    // Declare variables with initialization
    %a: r0<int> = 3           // First triangle side
    %b: r1<int> = 4           // Second triangle side
    %c: r2<int> = 5           // Hypotenuse
    
    // Declare temporary variables for calculations
    %a_squared: r4<int>
    %b_squared: r5<int>
    %sum: r3<int>
    %c_squared: r6<int>
    %difference: r7<int>
    %zero: r8<int> = 0        // For comparison
    %result: r9<bool>         // For assertion result
    
    // Calculate a² + b² = c²
    POW a, 2, a_squared       // a² = 9
    POW b, 2, b_squared      // b² = 16
    ADD a_squared, b_squared, sum     // sum = a² + b² = 25
    POW c, 2, c_squared      // c² = 25
    SUB sum, c_squared, difference   // difference = sum - c² = 0
    
    // Compare difference with zero using IFF (if and only if)
    IFF difference, zero, result  // True if difference == 0
    ASSERT result                 // Assert the equality
    HLT
}

// Run the proof
Proof()
```