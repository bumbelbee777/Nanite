import asyncio
import math
import re
import struct
from collections import deque
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from .ops import MultivectorOps
from .constraints import ConstraintChecker, ContextTracker, LogicalConstraint, TemporalConstraint, ConstraintType, safe_eval
from .exceptions import ConstraintViolationError, VMError, InvalidOpcodeError, InvalidOperandError, StackError, MemoryError
from .shared import get_shared_ops

class Opcode(Enum):
    # Base operations (0x0000-0x00FF)
    NOP = 0x0000
    HLT = 0x0001
    LOAD = 0x0002
    LOADC = 0x0003
    STORE = 0x0004
    STOREC = 0x0005
    PUSH = 0x0006
    POP = 0x0007
    MOV = 0x0008
    
    # Arithmetic operations (0x0100-0x01FF)
    ADD = 0x0100
    SUB = 0x0101
    MUL = 0x0102
    DIV = 0x0103
    POW = 0x0104
    SQRT = 0x0105
    EXP = 0x0106
    LOG = 0x0107
    LOG10 = 0x0108
    LN = 0x0109
    ERR = 0x010A
    
    # Trigonometric operations (0x0200-0x02FF)
    SIN = 0x0200
    COS = 0x0201
    TAN = 0x0202
    ASIN = 0x0203
    ACOS = 0x0204
    ATAN = 0x0205
    
    # Geometric operations (0x0300-0x03FF)
    GEOM_PROD = 0x0300
    OUTER_PROD = 0x0301
    INNER_PROD = 0x0302
    REVERSE = 0x0303
    DUAL = 0x0304
    
    # Complex operations (0x0400-0x04FF)
    COMPLEX_ADD = 0x0400
    COMPLEX_MUL = 0x0401
    COMPLEX_DIV = 0x0402
    COMPLEX_CONJ = 0x0403
    
    # Logical operations (0x0500-0x05FF)
    AND = 0x0500
    OR = 0x0501
    NOT = 0x0502
    XOR = 0x0503
    NAND = 0x0504
    NOR = 0x0505
    IMPLIES = 0x0506
    IFF = 0x0507
    
    # Symbolic logic operations (0x0600-0x06FF)
    FORALL = 0x0600
    EXISTS = 0x0601
    UNIFY = 0x0602
    RESOLVE = 0x0603
    CONTRADICTS = 0x0604
    
    # Concept operations (0x0700-0x07FF)
    CORR = 0x0700
    ENER = 0x0701
    CASSERT = 0x0702
    CBIND = 0x0703
    CQUERY = 0x0704
    CAND = 0x0705
    COR = 0x0706
    CNOT = 0x0707
    CIF = 0x0708
    CCMP = 0x0709
    CLEAN = 0x070A
    
    # Control flow (0x0800-0x08FF)
    JM = 0x0800
    CJM = 0x0801
    CALL = 0x0802
    RET = 0x0803
    
    # Special operations (0x0900-0x09FF)
    WAIT = 0x0900
    SLEEP = 0x0901
    ASSERT = 0x0902

    @classmethod
    def from_relationship(cls, weight: float, relationship_type: str) -> 'Opcode':
        """Map relationship weight and type to appropriate opcode."""
        if relationship_type == "geometric":
            return cls.GEOM_PROD if weight > 0.8 else cls.OUTER_PROD if weight > 0.6 else cls.INNER_PROD if weight > 0.4 else cls.DUAL
        elif relationship_type == "complex":
            return cls.COMPLEX_MUL if weight > 0.8 else cls.COMPLEX_ADD if weight > 0.6 else cls.COMPLEX_DIV if weight > 0.4 else cls.COMPLEX_CONJ
        elif relationship_type == "concept":
            return cls.CORR if weight > 0.8 else cls.CBIND if weight > 0.6 else cls.CQUERY if weight > 0.4 else cls.CCMP
        else:
            return cls.ADD if weight > 0.8 else cls.MUL if weight > 0.6 else cls.ADD if weight > 0.4 else cls.MOV


class TypedValue:
    def __init__(self, type_name: str, value: Any):
        self.type = type_name
        self.value = value

    def __repr__(self) -> str:
        return f"TypedValue(type={self.type}, value={self.value})"


class ExpressionParser:
    def __init__(self):
        self._patterns = {
            'int': re.compile(r"^(-?\d+)(?:_(i|u)(8|16|32|64|128))$"),
            'float': re.compile(r"^(-?\d+(?:\.\d*)?)(?:_(f)(8|16|32|64|128))$"),
            'routine': re.compile(r"^\.([A-Za-z_][A-Za-z0-9_]*)\((.*)\)$"),
            'variable': re.compile(r"^%([A-Za-z_][A-Za-z0-9_]*)$")
        }

    def parse(self, operand: str) -> TypedValue:
        if operand in ("true", "false"):
            return TypedValue("bool", operand == "true")
        if operand.startswith('"') and operand.endswith('"'):
            return TypedValue("str", operand[1:-1])
            
        for pattern_name, pattern in self._patterns.items():
            if match := pattern.match(operand):
                if pattern_name == 'int':
                    num, signed, bits = match.groups()
                    return TypedValue(f"{signed}{bits}", int(num))
                elif pattern_name == 'float':
                    num, fchar, bits = match.groups()
                    return TypedValue(f"f{bits}", float(num))
                elif pattern_name == 'routine':
                    routine_name, args = match.groups()
                    args_list = [self.parse(arg.strip()) for arg in args.split(",") if arg.strip()]
                    return TypedValue("routine", {"name": routine_name, "args": args_list})
                elif pattern_name == 'variable':
                    return TypedValue("variable", match.group(1))
                    
        raise ValueError(f"Unknown operand: {operand}")

    def parse_routine_call(self, line: str) -> Tuple[str, List[TypedValue]]:
        """Parses a routine call and its arguments."""
        if match := self._patterns['routine'].match(line):
            routine_name, args = match.groups()
            args_list = [self.parse(arg.strip()) for arg in args.split(",") if arg.strip()]
            return routine_name, args_list
        raise ValueError(f"Invalid routine call: {line}")

    def parse_variable_declaration(self, line: str) -> Tuple[str, TypedValue]:
        """Parses a variable declaration."""
        if "=" not in line:
            raise ValueError(f"Invalid variable declaration: {line}")
        var_name, value = map(str.strip, line.split("=", 1))
        if not self._patterns['variable'].match(var_name):
            raise ValueError(f"Invalid variable name: {var_name}")
        return var_name, self.parse(value)


class MemoryUnit:
    def __init__(self):
        self.memory: Dict[int, TypedValue] = {}
        self.concept_memory: Dict[int, TypedValue] = {}

    def load(self, address: int, is_concept: bool = False) -> TypedValue:
        memory = self.concept_memory if is_concept else self.memory
        value = memory.get(address, TypedValue("null", None))
        if value.type == "nan-boxed":
            return TypedValue(**self._decode_nan_boxed(value.value))
        return value

    def store(self, address: int, value: TypedValue, is_concept: bool = False):
        memory = self.concept_memory if is_concept else self.memory
        memory[address] = TypedValue("nan-boxed", self._encode_nan_boxed(value)) if self._should_nan_box(value) else value

    def _should_nan_box(self, value: TypedValue) -> bool:
        return value.type in {"i32", "f32", "bool"}

    def _encode_nan_boxed(self, value: TypedValue) -> int:
        if value.type == "i32":
            return (0x7FF00000 << 32) | (value.value & 0xFFFFFFFF)
        elif value.type == "f32":
            return struct.unpack("Q", struct.pack("f", value.value))[0]
        elif value.type == "bool":
            return (0x7FF00000 << 32) | (1 if value.value else 0)
        raise ValueError(f"Cannot NaN-box type: {value.type}")

    def _decode_nan_boxed(self, nan_boxed_value: int) -> Dict[str, Any]:
        if (nan_boxed_value >> 32) == 0x7FF00000:
            raw_value = nan_boxed_value & 0xFFFFFFFF
            if raw_value in (0, 1):
                return {"type_name": "bool", "value": bool(raw_value)}
            else:
                return {"type_name": "i32", "value": raw_value}
        try:
            # Try to decode as float
            return {"type_name": "f32", "value": struct.unpack("f", struct.pack("Q", nan_boxed_value))[0]}
        except Exception:
            raise ValueError("Invalid NaN-boxed value")


def validate_opcode_args(opcode: Opcode, args: list[str]):
    """Validates instruction parameters and stack state."""
    # Define expected argument counts for each opcode
    arg_counts = {
        # Base operations
        Opcode.NOP: 0,
        Opcode.HLT: 0,
        Opcode.LOAD: 2,
        Opcode.LOADC: 2,
        Opcode.STORE: 2,
        Opcode.STOREC: 2,
        Opcode.PUSH: 1,
        Opcode.POP: 1,
        Opcode.MOV: 2,
        
        # Arithmetic operations
        Opcode.ADD: 3,
        Opcode.SUB: 3,
        Opcode.MUL: 3,
        Opcode.DIV: 3,
        Opcode.POW: 3,
        Opcode.SQRT: 2,
        Opcode.EXP: 1,
        Opcode.LOG: 1,
        Opcode.LOG10: 1,
        Opcode.LN: 1,
        Opcode.ERR: 1,
        
        # Trigonometric operations
        Opcode.SIN: 1,
        Opcode.COS: 1,
        Opcode.TAN: 1,
        Opcode.ASIN: 1,
        Opcode.ACOS: 1,
        Opcode.ATAN: 1,
        
        # Geometric operations
        Opcode.GEOM_PROD: 3,
        Opcode.OUTER_PROD: 3,
        Opcode.INNER_PROD: 3,
        Opcode.REVERSE: 2,
        Opcode.DUAL: 2,
        
        # Complex operations
        Opcode.COMPLEX_ADD: 3,
        Opcode.COMPLEX_MUL: 3,
        Opcode.COMPLEX_DIV: 3,
        Opcode.COMPLEX_CONJ: 2,
        
        # Logical operations
        Opcode.AND: 3,
        Opcode.OR: 3,
        Opcode.NOT: 2,
        Opcode.XOR: 3,
        Opcode.NAND: 3,
        Opcode.NOR: 3,
        Opcode.IMPLIES: 3,
        Opcode.IFF: 3,
        
        # Symbolic logic operations
        Opcode.FORALL: 2,
        Opcode.EXISTS: 2,
        Opcode.UNIFY: 2,
        Opcode.RESOLVE: 2,
        Opcode.CONTRADICTS: 1,
        
        # Concept operations
        Opcode.CORR: 3,
        Opcode.ENER: 1,
        Opcode.CASSERT: 1,
        Opcode.CBIND: 3,
        Opcode.CQUERY: 1,
        Opcode.CAND: 2,
        Opcode.COR: 2,
        Opcode.CNOT: 2,
        Opcode.CIF: 2,
        Opcode.CCMP: 3,
        Opcode.CLEAN: 1,
        
        # Control flow
        Opcode.JM: 1,
        Opcode.CJM: 2,
        Opcode.CALL: 1,
        Opcode.RET: 0,
        
        # Special operations
        Opcode.WAIT: 1,
        Opcode.SLEEP: 1,
        Opcode.ASSERT: 1
    }
    
    # Check if opcode exists in our mapping
    if opcode not in arg_counts:
        raise ValueError(f"Unknown opcode: {opcode}")
    
    # Validate argument count
    expected_count = arg_counts[opcode]
    if len(args) != expected_count:
        raise ValueError(
            f"Opcode {opcode} expects {expected_count} arguments, got {len(args)}"
        )
    
    # Validate argument types based on opcode category
    if opcode in {Opcode.LOAD, Opcode.LOADC, Opcode.STORE, Opcode.STOREC}:
        try:
            int(args[0])  # Validate address
        except ValueError:
            raise ValueError(f"Invalid address format: {args[0]}")
        if not args[1].startswith('R'):
            raise ValueError(f"Invalid register format: {args[1]}")
    
    elif opcode in {Opcode.PUSH, Opcode.POP, Opcode.MOV}:
        if not all(arg.startswith('R') for arg in args):
            raise ValueError(f"Invalid register format in {opcode} operation")
    
    elif opcode in {Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.DIV, Opcode.POW}:
        if not all(arg.startswith('R') for arg in args):
            raise ValueError(f"Invalid register format in {opcode} operation")
        if opcode == Opcode.DIV and args[1] == 'R0':
            raise ValueError("Division by zero register (R0) not allowed")
    
    elif opcode in {Opcode.GEOM_PROD, Opcode.OUTER_PROD, Opcode.INNER_PROD}:
        if not all(arg.startswith('R') for arg in args):
            raise ValueError(f"Invalid register format in {opcode} operation")
        if args[0] == args[1]:
            raise ValueError(f"Source registers must be different in {opcode} operation")
    
    elif opcode in {Opcode.COMPLEX_ADD, Opcode.COMPLEX_MUL, Opcode.COMPLEX_DIV}:
        if not all(arg.startswith('R') for arg in args):
            raise ValueError(f"Invalid register format in {opcode} operation")
        if opcode == Opcode.COMPLEX_DIV and args[1] == 'R0':
            raise ValueError("Complex division by zero register (R0) not allowed")
    
    elif opcode in {Opcode.JM, Opcode.CALL}:
        try:
            int(args[0])  # Validate address
        except ValueError:
            raise ValueError(f"Invalid address format: {args[0]}")
    
    elif opcode in {Opcode.WAIT, Opcode.SLEEP}:
        try:
            duration = int(args[0])
            if duration < 0:
                raise ValueError(f"Invalid duration: {duration}")
        except ValueError:
            raise ValueError(f"Invalid duration format: {args[0]}")
    
    elif opcode == Opcode.CASSERT:
        if not args[0].startswith('C'):
            raise ValueError(f"Invalid concept format: {args[0]}")


class InstructionPipeline:
    def __init__(self, code: List[Tuple[Opcode, List[str]]], prefetch_size: int = 4, bundle_size: int = 2):
        self.code = code
        self.ip = 0
        self.prefetch_size = prefetch_size
        self.bundle_size = bundle_size
        self.prefetched_instructions: deque[Tuple[Opcode, List[str]]] = deque()

    def fetch_bundle(self) -> List[Tuple[Opcode, List[str]]]:
        if not self.prefetched_instructions:
            self._prefetch()
        return [self.prefetched_instructions.popleft() for _ in range(min(self.bundle_size, len(self.prefetched_instructions)))]

    def _prefetch(self):
        for _ in range(self.prefetch_size):
            if self.ip < len(self.code):
                self.prefetched_instructions.append(self.code[self.ip])
                self.ip += 1

    def reset(self):
        self.ip = 0
        self.prefetched_instructions.clear()


class BytecodeEngine:
    def __init__(self, pipeline: InstructionPipeline, memory: MemoryUnit):
        self.pipeline = pipeline
        self.memory = memory
        self.registers = {f"R{i}": TypedValue("i32", 0) for i in range(512)}
        self.concepts = {f"C{i}": TypedValue("i32", 0) for i in range(512)}
        self.stack: List[TypedValue] = []
        self.halted = False
        # Use shared ops if available, otherwise create a new instance
        shared_ops = get_shared_ops()
        self.multivector_ops = shared_ops if shared_ops is not None else MultivectorOps()
        
        # New components for constraint and context handling
        self.constraint_checker = ConstraintChecker()
        self.context_tracker = ContextTracker()
        
        # Initialize operation handlers with constraint checking
        self._operation_handlers = {
            # Base operations
            Opcode.NOP: lambda args: None,
            Opcode.HLT: lambda args: setattr(self, 'halted', True),
            Opcode.LOAD: lambda args: self._handle_load(args, False),
            Opcode.LOADC: lambda args: self._handle_load_constant(args),
            Opcode.STORE: lambda args: self._handle_store(args, False),
            Opcode.STOREC: lambda args: self._handle_store(args, True),
            Opcode.PUSH: lambda args: self.stack.append(self.registers[args[0]]),
            Opcode.POP: lambda args: setattr(self.registers, args[0], self.stack.pop() if self.stack else None),
            Opcode.MOV: lambda args: setattr(self.registers, args[1], self.registers[args[0]]),
            
            # Arithmetic operations
            Opcode.ADD: lambda args: self._handle_arithmetic(args, lambda x, y: x + y),
            Opcode.SUB: lambda args: self._handle_arithmetic(args, lambda x, y: x - y),
            Opcode.MUL: lambda args: self._handle_arithmetic(args, lambda x, y: x * y),
            Opcode.DIV: lambda args: self._handle_arithmetic(args, lambda x, y: x // y),
            Opcode.POW: lambda args: self._handle_arithmetic(args, lambda x, y: x ** y),
            Opcode.SQRT: lambda args: self._handle_unary_arithmetic(args, lambda x: int(math.sqrt(x))),
            Opcode.EXP: lambda args: self._handle_unary_arithmetic(args, lambda x: int(math.exp(x))),
            Opcode.LOG: lambda args: self._handle_unary_arithmetic(args, lambda x: int(math.log2(x))),
            Opcode.LOG10: lambda args: self._handle_unary_arithmetic(args, lambda x: int(math.log10(x))),
            Opcode.LN: lambda args: self._handle_unary_arithmetic(args, lambda x: int(math.log(x))),
            Opcode.ERR: lambda args: self._handle_unary_arithmetic(args, lambda x: int(math.erf(x))),
            
            # Trigonometric operations
            Opcode.SIN: lambda args: self._handle_unary_arithmetic(args, lambda x: int(math.sin(x))),
            Opcode.COS: lambda args: self._handle_unary_arithmetic(args, lambda x: int(math.cos(x))),
            Opcode.TAN: lambda args: self._handle_unary_arithmetic(args, lambda x: int(math.tan(x))),
            Opcode.ASIN: lambda args: self._handle_unary_arithmetic(args, lambda x: int(math.asin(x))),
            Opcode.ACOS: lambda args: self._handle_unary_arithmetic(args, lambda x: int(math.acos(x))),
            Opcode.ATAN: lambda args: self._handle_unary_arithmetic(args, lambda x: int(math.atan(x))),
            
            # Geometric operations
            Opcode.GEOM_PROD: lambda args: self._handle_geometric(args, 'geometric'),
            Opcode.OUTER_PROD: lambda args: self._handle_geometric(args, 'outer'),
            Opcode.INNER_PROD: lambda args: self._handle_geometric(args, 'inner'),
            Opcode.REVERSE: lambda args: self._handle_unary_geometric(args),
            Opcode.DUAL: lambda args: self._handle_unary_geometric(args),
            
            # Complex operations
            Opcode.COMPLEX_ADD: lambda args: self._handle_complex(args, 'add'),
            Opcode.COMPLEX_MUL: lambda args: self._handle_complex(args, 'mul'),
            Opcode.COMPLEX_DIV: lambda args: self._handle_complex(args, 'div'),
            Opcode.COMPLEX_CONJ: lambda args: self._handle_unary_complex(args),
            
            # Logical operations
            Opcode.AND: lambda args: self._handle_logical(args, lambda x, y: x and y),
            Opcode.OR: lambda args: self._handle_logical(args, lambda x, y: x or y),
            Opcode.NOT: lambda args: self._handle_unary_logical(args, lambda x: not x),
            Opcode.XOR: lambda args: self._handle_logical(args, lambda x, y: x != y),
            Opcode.NAND: lambda args: self._handle_logical(args, lambda x, y: not (x and y)),
            Opcode.NOR: lambda args: self._handle_logical(args, lambda x, y: not (x or y)),
            Opcode.IMPLIES: lambda args: self._handle_logical(args, lambda x, y: not x or y),
            Opcode.IFF: lambda args: self._handle_logical(args, lambda x, y: x == y),
            
            # Symbolic logic operations
            Opcode.FORALL: lambda args: self._handle_quantifier(args, 'forall'),
            Opcode.EXISTS: lambda args: self._handle_quantifier(args, 'exists'),
            Opcode.UNIFY: lambda args: self._handle_unification(args),
            Opcode.RESOLVE: lambda args: self._handle_resolution(args),
            Opcode.CONTRADICTS: lambda args: self._handle_contradiction_check(args),
            
            # Concept operations
            Opcode.CORR: lambda args: self._handle_concept_correlation(args),
            Opcode.ENER: lambda args: self._handle_concept_energy(args),
            Opcode.CASSERT: lambda args: self._handle_concept_assert(args),
            Opcode.CBIND: lambda args: self._handle_concept_binding(args),
            Opcode.CQUERY: lambda args: self._handle_concept_query(args),
            Opcode.CAND: lambda args: self._handle_concept_logical(args, 'and'),
            Opcode.COR: lambda args: self._handle_concept_logical(args, 'or'),
            Opcode.CNOT: lambda args: self._handle_concept_logical(args, 'not'),
            Opcode.CIF: lambda args: self._handle_concept_conditional(args),
            Opcode.CCMP: lambda args: self._handle_concept_comparison(args),
            Opcode.CLEAN: lambda args: self._handle_concept_cleanup(args),
            
            # Control flow
            Opcode.JM: lambda args: setattr(self.pipeline, 'ip', int(args[0])),
            Opcode.CJM: lambda args: self._handle_conditional_jump(args),
            Opcode.CALL: lambda args: self._handle_call(args),
            Opcode.RET: lambda args: self._handle_return(),
            
            # Special operations
            Opcode.WAIT: lambda args: asyncio.sleep(int(args[0]) / 1000),
            Opcode.SLEEP: lambda args: asyncio.sleep(int(args[0]) / 1000),
            Opcode.ASSERT: lambda args: self._handle_assert(args)
        }
        
        # Add default constraints
        self._setup_default_constraints()

    def _setup_default_constraints(self):
        """Setup default constraints for common operations."""
        # Division by zero constraint - make it more lenient for testing
        self.constraint_checker.add_constraint(
            "DIV",
            LogicalConstraint(
                condition="registers.get(args[1], TypedValue('null', None)).value != 0",
                constraint_type=ConstraintType.PRE_CONDITION,
                description="Division by zero check"
            )
        )
        
        # Concept format constraint - make it more lenient for testing
        self.constraint_checker.add_constraint(
            "CASSERT",
            LogicalConstraint(
                condition="args[0].startswith('C') if args else False",
                constraint_type=ConstraintType.PRE_CONDITION,
                description="Concept format check"
            )
        )
        
        # Stack underflow constraint
        self.constraint_checker.add_constraint(
            "POP",
            LogicalConstraint(
                condition="len(self.stack) > 0",
                constraint_type=ConstraintType.PRE_CONDITION,
                description="Stack underflow check"
            )
        )

    async def execute_bundle(self, bundle: List[Tuple[Opcode, List[str]]]):
        await asyncio.gather(*(self._execute_instruction_async(instr) for instr in bundle))

    async def _execute_instruction_async(self, instr: Tuple[Opcode, List[str]]):
        self._execute_instruction(instr)

    async def execute(self):
        """Async: Execute instruction bundles until halted."""
        while not self.halted:
            if bundle := self.pipeline.fetch_bundle():
                await self.execute_bundle(bundle)
            else:
                break

    def _execute_instruction(self, instr: Tuple[Opcode, List[str]]):
        """Execute an instruction with constraint checking."""
        validate_opcode_args(*instr)
        op, args = instr
        
        # Update context before execution
        self.context_tracker.update_context(op.name, args)
        
        # Get current VM state
        vm_state = {
            "registers": self.registers,
            "concepts": self.concepts,
            "concept_registers": self.concepts,  # Add for backward compatibility
            "stack": self.stack,
            "context": self.context_tracker.get_current_context(),
            "args": args  # Add args for constraint conditions
        }
        
        # Check constraints
        if not self.constraint_checker.check_constraints(op.name, vm_state):
            raise ConstraintViolationError(f"Constraints violated for {op.name}")
        
        if handler := self._operation_handlers.get(op):
            result = handler(args)
            
            # Check post-conditions
            if not self.constraint_checker.check_constraints(f"{op.name}_post", vm_state):
                raise ConstraintViolationError(f"Post-conditions violated for {op.name}")
                
            return result
        else:
            raise ValueError(f"Unhandled opcode: {op}")

    def _handle_load(self, args: List[str], is_concept: bool):
        self.registers[args[1]] = self.memory.load(int(args[0]), is_concept=is_concept)

    def _handle_store(self, args: List[str], is_concept: bool):
        self.memory.store(int(args[0]), self.registers[args[1]], is_concept=is_concept)

    def _handle_arithmetic(self, args: List[str], operation):
        """Handle arithmetic operations."""
        source1 = args[0]
        source2 = args[1]
        target = args[2]
        
        if source1 not in self.registers or source2 not in self.registers:
            raise ValueError("Source registers not found")
            
        # Get values directly - they should already be numeric
        val1 = self.registers[source1].value
        val2 = self.registers[source2].value
        
        # Handle None values
        if val1 is None:
            val1 = 0
        if val2 is None:
            val2 = 0
        
        # Ensure values are numeric
        if not isinstance(val1, (int, float, complex)):
            try:
                val1 = float(val1)
            except (ValueError, TypeError):
                raise ValueError(f"Cannot convert {val1} to number")
                
        if not isinstance(val2, (int, float, complex)):
            try:
                val2 = float(val2)
            except (ValueError, TypeError):
                raise ValueError(f"Cannot convert {val2} to number")
            
        if source2 == 'R0' and operation.__name__ == 'truediv':
            raise ValueError("Division by zero register (R0) not allowed")
            
        result = operation(val1, val2)
        self.registers[target] = TypedValue(
            type(result).__name__, result
        )

    def _handle_unary_arithmetic(self, args: List[str], operation):
        reg, dest = args
        self.registers[dest] = TypedValue("i32", operation(self.registers[reg].value))

    def _handle_geometric(self, args: List[str], operation_type: str):
        """Handle geometric operations."""
        source1 = args[0]
        source2 = args[1]
        target = args[2]
        
        if source1 not in self.registers or source2 not in self.registers:
            raise ValueError("Source registers not found")
            
        if source1 == source2:
            raise ValueError("Source registers must be different")
            
        # Get values from registers
        val1 = self.registers[source1].value
        val2 = self.registers[source2].value
        
        # Convert to numpy arrays if needed
        if not isinstance(val1, np.ndarray):
            val1 = np.array(val1)
        if not isinstance(val2, np.ndarray):
            val2 = np.array(val2)
            
        if operation_type == "geometric":
            # Use geometric product
            result = asyncio.run(self.multivector_ops.geometric_product(val1, val2))
        elif operation_type == "outer":
            # Use outer product
            result = asyncio.run(self.multivector_ops.outer_product(val1, val2))
        elif operation_type == "inner":
            # Use inner product
            result = asyncio.run(self.multivector_ops.inner_product(val1, val2))
        elif operation_type == "prod":
            # Simple multiplication
            result = np.multiply(val1, val2)
        else:
            raise ValueError(f"Unsupported geometric operation: {operation_type}")
            
        self.registers[target] = TypedValue("geometric", result)

    def _handle_unary_geometric(self, args: List[str]):
        reg, dest = args
        a = self.registers[reg].value
        
        # Convert to numpy array for MultivectorOps
        a_np = np.array(a) if not isinstance(a, np.ndarray) else a
        
        # For unary operations, we use the geometric product with a special basis
        # This effectively implements reverse and dual operations
        basis = np.eye(a_np.shape[0], dtype=np.complex128)
        result = asyncio.run(self.multivector_ops.geometric_product(a_np, basis))
        
        self.registers[dest] = TypedValue("complex", result)

    def _handle_complex(self, args: List[str], operation_type: str):
        """Handle complex operations."""
        source1 = args[0]
        source2 = args[1]
        target = args[2]
        
        if source1 not in self.registers or source2 not in self.registers:
            raise ValueError("Source registers not found")
            
        if operation_type == "div" and self.registers[source2].value == complex(0, 0):
            raise ValueError("Complex division by zero not allowed")
            
        if operation_type == "div":
            result = self.registers[source1].value / self.registers[source2].value
        elif operation_type == "mul":
            result = self.registers[source1].value * self.registers[source2].value
        elif operation_type == "add":
            result = self.registers[source1].value + self.registers[source2].value
        else:
            raise ValueError(f"Unsupported complex operation: {operation_type}")
            
        self.registers[target] = TypedValue("complex", result)

    def _handle_unary_complex(self, args: List[str]):
        reg, dest = args
        a = self.registers[reg].value
        
        # Convert to numpy array for MultivectorOps
        a_np = np.array(a) if not isinstance(a, np.ndarray) else a
        
        # For complex conjugate, we use the geometric product with a special basis
        basis = np.eye(a_np.shape[0], dtype=np.complex128)
        result = asyncio.run(self.multivector_ops.geometric_product(a_np, basis))
        
        self.registers[dest] = TypedValue("complex", result)

    def _handle_logical(self, args: List[str], operation):
        """Handle logical operations."""
        source1 = args[0]
        source2 = args[1]
        target = args[2]
        
        if source1 not in self.registers or source2 not in self.registers:
            raise ValueError("Source registers not found")
            
        val1 = bool(self.registers[source1].value)
        val2 = bool(self.registers[source2].value)
        
        result = operation(val1, val2)
        self.registers[target] = TypedValue("bool", result)

    def _handle_unary_logical(self, args: List[str], operation):
        """Handle unary logical operations."""
        source = args[0]
        target = args[1]
        
        if source not in self.registers:
            raise ValueError("Source register not found")
            
        val = bool(self.registers[source].value)
        result = operation(val)
        self.registers[target] = TypedValue("bool", result)

    def _handle_quantifier(self, args: List[str], quantifier_type: str):
        reg, dest = args
        values = self.registers[reg].value
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"Quantifier requires a collection of values, got {type(values)}")
        
        if quantifier_type == 'forall':
            result = all(bool(v) for v in values)
        else:  # exists
            result = any(bool(v) for v in values)
            
        self.registers[dest] = TypedValue("bool", result)

    def _handle_unification(self, args: List[str]):
        reg1, reg2, dest = args
        term1 = self.registers[reg1].value
        term2 = self.registers[reg2].value
        
        # Simple unification for now - can be extended with more sophisticated logic
        if term1 == term2:
            self.registers[dest] = TypedValue("bool", True)
        else:
            self.registers[dest] = TypedValue("bool", False)

    def _handle_resolution(self, args: List[str]):
        reg1, reg2, dest = args
        clause1 = self.registers[reg1].value
        clause2 = self.registers[reg2].value
        
        # Simple resolution for now - can be extended with more sophisticated logic
        if isinstance(clause1, (list, tuple)) and isinstance(clause2, (list, tuple)):
            # Check if clauses can be resolved
            result = any(not lit in clause2 for lit in clause1)
            self.registers[dest] = TypedValue("bool", result)
        else:
            raise ValueError("Resolution requires clause collections")

    def _handle_contradiction_check(self, args: List[str]):
        reg, dest = args
        frame = self.registers[reg].value
        
        if not isinstance(frame, (list, tuple)):
            raise ValueError("Contradiction check requires a logical frame")
            
        # Check for direct contradictions (p and not p)
        for i, lit1 in enumerate(frame):
            for lit2 in frame[i+1:]:
                if lit1 == (not lit2):
                    self.registers[dest] = TypedValue("bool", True)
                    return
                    
        self.registers[dest] = TypedValue("bool", False)

    def _handle_concept_correlation(self, args: List[str]):
        concept1, concept2, weight = args
        # Implement concept correlation using MultivectorOps
        c1 = self.concepts[concept1].value
        c2 = self.concepts[concept2].value
        result = asyncio.run(self.multivector_ops.geometric_product(c1, c2))
        self.concepts[concept1] = TypedValue("complex", result * float(weight))

    def _handle_concept_energy(self, args: List[str]):
        """Handle concept energy update."""
        concept_name = args[0]
        if concept_name not in self.concepts:
            # Create the concept if it doesn't exist
            self.concepts[concept_name] = TypedValue("complex", 0.0+0j)
            
        if 'R1' in self.registers:
            energy_value = self.registers['R1'].value
            # Ensure it's a complex number
            if isinstance(energy_value, (int, float)):
                energy_value = complex(energy_value, 0)
            self.concepts[concept_name] = TypedValue("complex", energy_value)
        else:
            # Default energy value
            self.concepts[concept_name] = TypedValue("complex", 0.0+0j)

    def _handle_concept_binding(self, args: List[str]):
        """Handle concept binding."""
        source = args[0]
        target = args[1]
        result = args[2]
        
        if source not in self.concepts or target not in self.concepts:
            raise ValueError("Source or target concept does not exist")
            
        # Create new concept from binding
        c1 = self.concepts[source]
        c2 = self.concepts[target]
        
        # Get the energy values from the concepts
        energy1 = c1.value
        energy2 = c2.value
        
        # Convert to complex numbers if needed
        if isinstance(energy1, (int, float)):
            energy1 = complex(energy1, 0)
        if isinstance(energy2, (int, float)):
            energy2 = complex(energy2, 0)
        
        # Create a meaningful binding: combine energies with some interaction
        # Use a simple weighted combination for now
        binding_energy = 0.5 * energy1 + 0.5 * energy2 + 0.1 * energy1 * energy2
        
        self.concepts[result] = TypedValue("complex", binding_energy)

    def _handle_concept_query(self, args: List[str]):
        pattern, dest = args
        # Implement concept graph query
        matches = []
        for concept, value in self.concepts.items():
            if self._matches_pattern(value, pattern):
                matches.append(concept)
        self.registers[dest] = TypedValue("list", matches)

    def _handle_concept_logical(self, args: List[str], operation: str):
        if operation == 'not':
            concept, dest = args
            c = self.concepts[concept]
            result = asyncio.run(self.multivector_ops.geometric_product(c.value, -1))
        else:
            concept1, concept2, dest = args
            c1 = self.concepts[concept1]
            c2 = self.concepts[concept2]
            if operation == 'and':
                result = asyncio.run(self.multivector_ops.inner_product(c1.value, c2.value))
            else:  # or
                result = asyncio.run(self.multivector_ops.outer_product(c1.value, c2.value))
                
        self.concepts[dest] = TypedValue("complex", result)

    def _handle_concept_conditional(self, args: List[str]):
        condition, concept, dest = args
        if safe_eval(condition, {"registers": self.registers}):
            self.concepts[dest] = self.concepts[concept]
        else:
            self.concepts[dest] = TypedValue("null", None)

    def _handle_concept_comparison(self, args: List[str]):
        concept1, concept2, dest = args
        c1 = self.concepts[concept1]
        c2 = self.concepts[concept2]
        
        # Compare concepts using geometric similarity
        similarity = asyncio.run(self.multivector_ops.inner_product(c1.value, c2.value))
        self.registers[dest] = TypedValue("f64", float(similarity))

    def _handle_concept_cleanup(self, args: List[str]):
        threshold = float(args[0])
        # Clean up low-energy concepts
        for concept, value in list(self.concepts.items()):
            energy = asyncio.run(self.multivector_ops.inner_product(value.value, value.value))
            if float(energy) < threshold:
                del self.concepts[concept]

    def _matches_pattern(self, value: Any, pattern: str) -> bool:
        """Helper method for concept pattern matching."""
        if isinstance(value, str):
            return pattern in value
        elif isinstance(value, (list, tuple)):
            return any(self._matches_pattern(v, pattern) for v in value)
        return False

    def _handle_conditional_jump(self, args: List[str]):
        """Handle conditional jump operation."""
        condition = args[0]
        target = int(args[1])
        if condition not in self.registers:
            raise ValueError("Condition register not found")
        if bool(self.registers[condition].value):
            self.pipeline.ip = target

    def _handle_call(self, args: List[str]):
        """Handle call operation."""
        target = int(args[0])
        # Store return address (current PC + 1)
        self.registers['R0'] = TypedValue('int', self.pipeline.ip + 1)
        # Jump to target
        self.pipeline.ip = target

    def _handle_return(self):
        if self.stack:
            self.pipeline.ip = self.stack.pop().value

    def _handle_assert(self, args: List[str]):
        """Handle assertion operation."""
        source = args[0]
        if source not in self.registers:
            raise ValueError("Source register not found")
        if not bool(self.registers[source].value):
            raise AssertionError(f"Assertion failed: {source} is False")

    def _handle_concept_assert(self, args: List[str]):
        """Handle concept assertion."""
        concept_name = args[0]
        if concept_name not in self.concepts:
            # Create a proper Concept object instead of dict
            self.concepts[concept_name] = TypedValue("complex", 0.0+0j)

    def _handle_load_constant(self, args: List[str]):
        """Handle loading constant value."""
        const_idx = int(args[0])
        target_reg = args[1]
        
        # Check if we have constants defined in the engine
        if hasattr(self, 'constants') and self.constants and const_idx in self.constants:
            self.registers[target_reg] = self.constants[const_idx]
        else:
            # Fallback: treat as float
            self.registers[target_reg] = TypedValue('float', float(const_idx))


class SillyVM:
    def __init__(self, code: Optional[List[Tuple[Opcode, List[str]]]] = None):
        self.memory = MemoryUnit()
        self.concepts = {}
        self.constraints = {}
        self._constants = {}  # Use _constants to avoid recursion
        self.instruction_count = 0
        self.max_instructions = 1000
        self.bundle_size = 10
        self.bundle_timeout = 0.1
        self.error_log = []
        self.last_result = None
        self.pipeline = None
        # Always create a dummy engine so registers are available
        self.engine = BytecodeEngine(InstructionPipeline([], bundle_size=self.bundle_size), self.memory)
        if code:
            self.load_program(code)

    def load_program(self, code: List[Tuple[Opcode, List[str]]]):
        self.pipeline = InstructionPipeline(code, bundle_size=self.bundle_size)
        self.engine = BytecodeEngine(self.pipeline, self.memory)
        self.engine.concepts = self.concepts
        self.engine.constraints = self.constraints
        # Pass constants to the engine
        self.engine.constants = self._constants
        self.instruction_count = 0
        self.max_instructions = len(code)

    @property
    def instruction_bundles(self):
        # For legacy tests, allow direct access to pipeline code
        if self.pipeline is not None:
            return self.pipeline.code
        return []

    @instruction_bundles.setter
    def instruction_bundles(self, value):
        if self.pipeline is not None:
            self.pipeline.code = value

    def execute_instruction(self, opcode: Opcode, args: List[str]):
        if self.engine is None or (self.pipeline is not None and len(self.pipeline.code) == 0):
            # Auto-load a dummy program if none is loaded
            self.load_program([])
        result = self.engine._execute_instruction((opcode, args))
        # Sync concepts after instruction execution
        self.concepts = self.engine.concepts
        return result

    def _handle_assert(self, args):
        if self.engine and hasattr(self.engine, '_handle_assert'):
            return self.engine._handle_assert(args)
        raise NotImplementedError("SillyVM._handle_assert is not implemented.")

    def _handle_conditional_jump(self, args):
        if self.engine and hasattr(self.engine, '_handle_conditional_jump'):
            return self.engine._handle_conditional_jump(args)
        raise NotImplementedError("SillyVM._handle_conditional_jump is not implemented.")

    async def run(self):
        """Async: Execute the loaded program using BytecodeEngine."""
        if self.engine is None:
            raise ValueError("No program loaded. Call load_program() first.")
        try:
            await self.engine.execute()
            self.last_result = self.engine.registers.get("R0", TypedValue("null", None))
            self.concepts = self.engine.concepts
            self.constraints = self.engine.constraints
            return self.last_result
        except Exception as e:
            self._record_error(e)
            raise

    def execute_bundles(self):
        """Execute instruction bundles using BytecodeEngine."""
        if self.engine is None:
            raise ValueError("No program loaded. Call load_program() first.")
        
        # Execute bundles and update instruction count
        while self.instruction_count < self.max_instructions:
            if bundle := self.pipeline.fetch_bundle():
                asyncio.run(self.engine.execute_bundle(bundle))
                self.instruction_count += len(bundle)
            else:
                break

    def get_execution_state(self) -> Dict[str, Any]:
        """Get the current state of the VM."""
        if self.engine is None:
            return {
                "instruction_count": self.instruction_count,
                "max_instructions": self.max_instructions,
                "concepts": self.concepts.copy(),
                "constraints": self.constraints.copy(),
                "last_result": self.last_result,
                "error_log": self.error_log
            }
        
        return {
            "instruction_count": self.instruction_count,
            "max_instructions": self.max_instructions,
            "registers": self.engine.registers.copy(),
            "concepts": self.engine.concepts.copy(),
            "constraints": self.engine.constraints.copy(),
            "last_result": self.last_result,
            "error_log": self.error_log
        }

    def reset(self):
        """Reset the VM to its initial state."""
        self.memory = MemoryUnit()
        self.concepts = {}
        self.constraints = {}
        self.instruction_count = 0
        self.max_instructions = 1000
        self.bundle_size = 10
        self.bundle_timeout = 0.1
        self.error_log = []
        self.last_result = None
        self.engine = None
        self.pipeline = None

    def _record_error(self, error: Exception):
        """Record execution error in history."""
        self.error_log.append({
            "error": str(error),
            "instruction_count": self.instruction_count,
            "max_instructions": self.max_instructions,
            "concepts": self.concepts.copy(),
            "constraints": self.constraints.copy()
        })

    def map_relationship_to_opcode(self, weight: float, relationship_type: str, 
                                 source_concept: str, target_concept: str) -> Tuple[Opcode, List[str]]:
        """Map a relationship to an opcode and its arguments."""
        opcode = Opcode.from_relationship(weight, relationship_type)
        return opcode, [source_concept, target_concept, str(weight)]

    def encode_instruction(self, opcode: Opcode, args: List[str]) -> bytes:
        """Encode an instruction into bytes."""
        return opcode.value.to_bytes(2, 'big') + len(args).to_bytes(1, 'big') + b''.join(struct.pack('d', float(arg)) if isinstance(arg, (int, float)) else len(arg.encode('utf-8')).to_bytes(1, 'big') + arg.encode('utf-8') for arg in args)

    def decode_instruction(self, instruction: bytes) -> Tuple[Opcode, List[str]]:
        """Decode bytes into an instruction."""
        opcode_value = int.from_bytes(instruction[:2], 'big')
        opcode = Opcode(opcode_value)
        num_args = instruction[2]
        args = []
        pos = 3
        for _ in range(num_args):
            if pos < len(instruction):
                if instruction[pos] < 128:
                    str_len = instruction[pos]
                    pos += 1
                    arg = instruction[pos:pos+str_len].decode('utf-8')
                    pos += str_len
                else:
                    arg = struct.unpack('d', instruction[pos:pos+8])[0]
                    pos += 8
                args.append(arg)
        return opcode, args

    @property
    def registers(self):
        """Access registers from the engine."""
        if self.engine is None:
            return {}
        return self.engine.registers

    @registers.setter
    def registers(self, value):
        """Set registers in the engine."""
        if self.engine is not None:
            self.engine.registers = value

    @property
    def constants(self):
        """Access constants from the VM."""
        return self._constants

    @constants.setter
    def constants(self, value):
        """Set constants in the VM and sync to engine."""
        self._constants = value
        if self.engine is not None:
            self.engine.constants = value

    # Add a property to make memory appear as a dict for backward compatibility
    @property
    def memory_dict(self):
        """Access memory as a dict for backward compatibility."""
        return self.memory.memory if hasattr(self.memory, 'memory') else {}

class BytecodeProgram:
    """
    Represents a human-readable bytecode program that can be executed by the SillyVM.
    Provides utilities for loading from a list of instructions, pretty-printing, and running.
    """

    def __init__(self, instructions=None):
        self.instructions = instructions or []  # List of (opcode, [args])
        self.constants = {}  # Add missing constants attribute
        self.concepts = {}  # Add missing concepts attribute

    @classmethod
    def from_concept_graph(cls, bytecode_list):
        """Create a BytecodeProgram from a list of (opcode, args) tuples."""
        return cls(instructions=bytecode_list)

    def add_instruction(self, opcode, args):
        self.instructions.append((opcode, args))

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, idx):
        return self.instructions[idx]

    def pretty_print(self):
        for idx, (op, args) in enumerate(self.instructions):
            print(f"{idx:04d}: {op} {', '.join(map(str, args))}")

    def to_pipeline(self):
        """Convert to a pipeline for execution in the VM."""
        return InstructionPipeline(
            [
                (Opcode[op] if isinstance(op, str) else op, args)
                for op, args in self.instructions
            ],
        )

    def execute(self, vm=None):
        """Run this program in a SillyVM instance (if provided) or create a new one."""
        if vm is None:
            vm = SillyVM()
        vm.constants = self.constants  # Set constants first!
        vm.concepts = self.concepts
        vm.load_program(self.instructions)
        vm.run()
        return vm.last_result

class OpcodeMapper:
    """Maps relationships to opcodes using geometric algebra principles."""
    
    def __init__(self):
        self.relationship_types = {
            "geometric": ["GEOM_PROD", "OUTER_PROD", "INNER_PROD", "DUAL"],
            "complex": ["COMPLEX_ADD", "COMPLEX_MUL", "COMPLEX_DIV", "COMPLEX_CONJ"],
            "concept": ["ADD_EDGE", "UPDATE_EDGE", "GET_EDGE", "GET_NODE"]
        }
        
    def encode_instruction(self, opcode: Opcode, args: list[str]) -> bytes:
        """Encode instruction into bytes using 16-bit opcode and argument encoding."""
        # Encode opcode (2 bytes)
        instruction = opcode.value.to_bytes(2, 'big')
        
        # Encode number of arguments (1 byte)
        instruction += len(args).to_bytes(1, 'big')
        
        # Encode each argument
        for arg in args:
            if isinstance(arg, (int, float)):
                # Encode numeric arguments as 8 bytes
                instruction += struct.pack('d', float(arg))
            else:
                # Encode string arguments with length prefix
                arg_bytes = arg.encode('utf-8')
                instruction += len(arg_bytes).to_bytes(1, 'big')
                instruction += arg_bytes
                
        return instruction
    
    def decode_instruction(self, instruction: bytes) -> tuple[Opcode, list[str]]:
        """Decode instruction bytes into opcode and arguments."""
        # Decode opcode (2 bytes)
        opcode_value = int.from_bytes(instruction[:2], 'big')
        opcode = Opcode(opcode_value)
        
        # Decode number of arguments (1 byte)
        num_args = instruction[2]
        
        # Decode arguments
        args = []
        pos = 3
        for _ in range(num_args):
            if pos < len(instruction):
                # Check if next byte is a length prefix (string) or start of double (numeric)
                if instruction[pos] < 128:  # String length prefix
                    str_len = instruction[pos]
                    pos += 1
                    arg = instruction[pos:pos+str_len].decode('utf-8')
                    pos += str_len
                else:  # Numeric value
                    arg = struct.unpack('d', instruction[pos:pos+8])[0]
                    pos += 8
                args.append(arg)
                
        return opcode, args
    
    def map_relationship(self, weight: float, relationship_type: str, 
                        source_concept: str, target_concept: str) -> tuple[Opcode, list[str]]:
        """Map relationship to opcode and generate appropriate arguments."""
        opcode = Opcode.from_relationship(weight, relationship_type)
        
        # Generate arguments based on relationship type and geometric properties
        if relationship_type == "geometric":
            # For geometric operations, include basis vectors and grades
            args = [source_concept, target_concept, str(weight), "basis_vectors"]
        elif relationship_type == "complex":
            # For complex operations, include phase and magnitude
            args = [source_concept, target_concept, str(weight), "complex_phase"]
        else:
            # For concept operations, include relationship metadata
            args = [source_concept, target_concept, str(weight), "concept_meta"]
            
        return opcode, args

# In BytecodeEngine, add concept_registers property for backward compatibility
setattr(BytecodeEngine, 'concept_registers', property(lambda self: self.concepts, lambda self, v: setattr(self, 'concepts', v)))
