# SillyISA: Overview & Design (Highly WIP)

## Introduction

SillyISA is the custom neuro-symbolic ISA for Nanite, allowing for formalization of proofs and steps to solving problems. By combining novel methods with existing bytecode virtual machine technologies, SillyISA seeks to refine and optimize reasoning and problem-solving capabilities found in Nanite.

## High-Level Overview

SillyISA combines typical instructions and features found in load-store architectures and VLIW/EPIC (where every instruction is explicitly parallelized), though in SillyISA's case all instructions and routines are ran asynchronously to prevent blocking and increase throughput. The ISA integrates geometric algebra operations, complex number handling, and concept graph manipulation for advanced symbolic reasoning.

## Registers

SillyISA has the following register types:
- **General Purpose Registers** (`R0`-`R511`): Type-agnostic registers for general computation
- **Concept Registers** (`C0`-`C511`): Specialized registers for concept graph operations
- **Control Registers** (implicit):
  - Instruction Pointer (`ip`): Current execution position
  - State Register (`sr`): Execution state and flags
  - Stack Pointer (`sp`): Current stack position
  - Base Pointer (`bp`): Current stack frame base
  - Frame Pointer (`fp`): Current function frame

### Memory

SillyISA segments memory into three areas:
1. **Concept Memory** (High): Stores concept graph nodes and relationships
   - Supports lazy paging
   - Logical compaction
   - Persistent mapping for symbolic graphs
   - NaN-boxing for efficient type storage
2. **Data Memory** (Middle): General purpose data storage
   - NaN-boxing for efficient type storage
   - Region-based allocation
   - Dynamic resizing
3. **Code Memory** (Low): Program instructions
   - Protected execution
   - Optional code signing
   - Dynamic loading

## Instruction List

### Base Operations (0x0000-0x00FF)
| **Mnemonic**   | **Opcode** | **Description**                                                                 |
|----------------|------------|---------------------------------------------------------------------------------|
| `NOP`          | 0x0000     | No-operation.                                                                  |
| `HLT`          | 0x0001     | Halts execution.                                                               |
| `LOAD`         | 0x0002     | Loads a value from memory into a register.                                     |
| `LOADC`        | 0x0003     | Loads a constant from the constants table into a register.                     |
| `STORE`        | 0x0004     | Stores a register value into memory.                                           |
| `STOREC`       | 0x0005     | Stores a concept to concept memory.                                            |
| `PUSH`         | 0x0006     | Pushes a value onto the stack.                                                 |
| `POP`          | 0x0007     | Pops a value from the stack.                                                   |
| `MOV`          | 0x0008     | Moves a value from one register to another.                                    |

### Arithmetic Operations (0x0100-0x01FF)
| **Mnemonic**   | **Opcode** | **Description**                                                                 |
|----------------|------------|---------------------------------------------------------------------------------|
| `ADD`          | 0x0100     | Adds two values.                                                               |
| `SUB`          | 0x0101     | Subtracts one value from another.                                              |
| `MUL`          | 0x0102     | Multiplies two values.                                                         |
| `DIV`          | 0x0103     | Divides one value by another.                                                  |
| `POW`          | 0x0104     | Exponentiates a base to a power.                                               |
| `SQRT`         | 0x0105     | Square root.                                                                   |
| `EXP`          | 0x0106     | Exponential (e^x).                                                             |
| `LOG`          | 0x0107     | Logarithm base 2.                                                              |
| `LOG10`        | 0x0108     | Logarithm base 10.                                                             |
| `LN`           | 0x0109     | Natural logarithm.                                                             |
| `ERR`          | 0x010A     | Error function (Gaussian integral approximation).                              |

### Trigonometric Operations (0x0200-0x02FF)
| **Mnemonic**   | **Opcode** | **Description**                                                                 |
|----------------|------------|---------------------------------------------------------------------------------|
| `SIN`          | 0x0200     | Sine function.                                                                 |
| `COS`          | 0x0201     | Cosine function.                                                               |
| `TAN`          | 0x0202     | Tangent function.                                                              |
| `ASIN`         | 0x0203     | Arcsine function.                                                              |
| `ACOS`         | 0x0204     | Arccosine function.                                                            |
| `ATAN`         | 0x0205     | Arctangent function.                                                           |

### Geometric Operations (0x0300-0x03FF)
| **Mnemonic**   | **Opcode** | **Description**                                                                 |
|----------------|------------|---------------------------------------------------------------------------------|
| `GEOM_PROD`    | 0x0300     | Geometric product of two multivectors.                                         |
| `OUTER_PROD`   | 0x0301     | Outer product of two multivectors.                                             |
| `INNER_PROD`   | 0x0302     | Inner product of two multivectors.                                             |
| `REVERSE`      | 0x0303     | Reverse of a multivector.                                                      |
| `DUAL`         | 0x0304     | Dual of a multivector.                                                         |

### Complex Operations (0x0400-0x04FF)
| **Mnemonic**   | **Opcode** | **Description**                                                                 |
|----------------|------------|---------------------------------------------------------------------------------|
| `COMPLEX_ADD`  | 0x0400     | Complex number addition.                                                       |
| `COMPLEX_MUL`  | 0x0401     | Complex number multiplication.                                                 |
| `COMPLEX_DIV`  | 0x0402     | Complex number division.                                                       |
| `COMPLEX_CONJ` | 0x0403     | Complex conjugate.                                                             |

### Logical Operations (0x0500-0x05FF)
| **Mnemonic**   | **Opcode** | **Description**                                                                 |
|----------------|------------|---------------------------------------------------------------------------------|
| `AND`          | 0x0500     | Logical conjunction.                                                           |
| `OR`           | 0x0501     | Logical disjunction.                                                           |
| `NOT`          | 0x0502     | Logical negation.                                                              |
| `XOR`          | 0x0503     | Exclusive or.                                                                  |
| `NAND`         | 0x0504     | Negated AND.                                                                   |
| `NOR`          | 0x0505     | Negated OR.                                                                    |
| `IMPLIES`      | 0x0506     | Logical implication.                                                           |
| `IFF`          | 0x0507     | If-and-only-if (bi-implication).                                               |

### Symbolic Logic Operations (0x0600-0x06FF)
| **Mnemonic**   | **Opcode** | **Description**                                                                 |
|----------------|------------|---------------------------------------------------------------------------------|
| `FORALL`       | 0x0600     | Universal quantification (∀).                                                  |
| `EXISTS`       | 0x0601     | Existential quantification (∃).                                                |
| `UNIFY`        | 0x0602     | Unify two terms/symbols under a shared binding.                                |
| `RESOLVE`      | 0x0603     | Logical resolution step (e.g. for inference chaining).                         |
| `CONTRADICTS`  | 0x0604     | Checks for contradiction in current logical frame.                             |

### Concept Operations (0x0700-0x07FF)
| **Mnemonic**   | **Opcode** | **Description**                                                                 |
|----------------|------------|---------------------------------------------------------------------------------|
| `CORR`         | 0x0700     | Correlates two concepts with a specified weight.                               |
| `ENER`         | 0x0701     | Stores the energy of a concept to memory.                                      |
| `CASSERT`      | 0x0702     | Asserts a concept relationship or state.                                       |
| `CBIND`        | 0x0703     | Binds a logical or symbolic variable to a concept.                             |
| `CQUERY`       | 0x0704     | Queries the concept graph for a specific structure/relation.                   |
| `CAND`         | 0x0705     | Conceptual "AND" — inference condition.                                        |
| `COR`          | 0x0706     | Conceptual "OR".                                                               |
| `CNOT`         | 0x0707     | Negates or disables a concept node.                                            |
| `CIF`          | 0x0708     | Triggers a concept if a condition is met.                                      |
| `CCMP`         | 0x0709     | Compares two concepts for similarity, identity, or energy difference.          |
| `CLEAN`        | 0x070A     | Prunes or resets the least-used portions of the concept memory.                |

### Control Flow (0x0800-0x08FF)
| **Mnemonic**   | **Opcode** | **Description**                                                                 |
|----------------|------------|---------------------------------------------------------------------------------|
| `JM`           | 0x0800     | Unconditional jump to a specified address.                                     |
| `CJM`          | 0x0801     | Conditional jump based on a register or flag.                                  |
| `CALL`         | 0x0802     | Calls a routine or function.                                                   |
| `RET`          | 0x0803     | Returns from a routine, optionally with a value.                               |

### Special Operations (0x0900-0x09FF)
| **Mnemonic**   | **Opcode** | **Description**                                                                 |
|----------------|------------|---------------------------------------------------------------------------------|
| `WAIT`         | 0x0900     | Waits for a number of milliseconds.                                            |
| `SLEEP`        | 0x0901     | Sleeps or halts the thread for a duration.                                     |
| `ASSERT`       | 0x0902     | Asserts a condition or logical statement.                                      |

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
| `Complex` | Complex number representation |
| `Multivector` | Geometric algebra multivector |

## 🧮 Geometric Algebra Integration

SillyISA integrates geometric algebra operations through the `MultivectorOps` backend, providing:

1. **Basic Operations**
   - Geometric product (`GEOM_PROD`)
   - Outer product (`OUTER_PROD`)
   - Inner product (`INNER_PROD`)
   - Reverse operation (`REVERSE`)
   - Dual operation (`DUAL`)

2. **Complex Number Support**
   - Complex addition (`COMPLEX_ADD`)
   - Complex multiplication (`COMPLEX_MUL`)
   - Complex division (`COMPLEX_DIV`)
   - Complex conjugate (`COMPLEX_CONJ`)

3. **Concept Graph Integration**
   - Concept correlation using geometric products
   - Energy calculation through inner products
   - Similarity comparison using geometric operations

4. **Performance Optimizations**
   - JIT-compiled operations
   - Efficient tensor conversions
   - Async execution support

## 🧠 Design Highlights

| Feature | Implication |
|---------|-------------|
| **Type-agnostic GPRs** | GPRs are like raw registers in most real-world ISAs — they're just slots. Typing is enforced in *routines and static checks*, not instruction semantics. |
| **Rich typing in routines** | Lets you write reusable, type-checked logic. Think like a type-aware macro system but still hardware-ish. |
| **Conceptual separation** | Execution is fast and dumb; analysis, optimization, and correctness happen above the ISA level. Clean separation. |
| **Concept/Prop special types** | Enables deep symbolic reasoning and neural-symbolic bridges (like propagating concept graphs or evaluating logical statements). |
| **Geometric Algebra Integration** | Provides powerful mathematical operations for complex number and multivector manipulation. |
| **Asynchronous Execution** | All operations are non-blocking, enabling high throughput and parallel execution. |
| **Constraint Checking** | Runtime validation of operations and constraints for safety and correctness. |
| **Context Tracking** | Maintains execution context for debugging and optimization. |

## Advanced Features

### Instruction Pipeline

The VM uses an instruction pipeline for efficient execution:

- **Prefetching**: Instructions are prefetched into bundles
- **Bundling**: Instructions are executed in bundles for efficiency
- **Async Execution**: Bundles can be executed asynchronously

### Memory Management

- **NaN-boxing**: Efficient type storage using NaN bit patterns
- **Concept Memory**: Specialized memory for concept graph operations
- **Region-based Allocation**: Memory is organized into regions for efficiency

### Constraint System

- **Runtime Validation**: Operations are validated against constraints
- **Logical Constraints**: Support for logical pre/post conditions
- **Temporal Constraints**: Time-based constraint checking

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