import re
import asyncio
import struct
import math
import time
from collections import deque
from enum import Enum, auto
from typing import Any, Dict, List, Tuple, Union, Optional, Deque, Set
from dataclasses import dataclass

# Initialize concept graph

def get_concept_graph():
    from sillyai.concept import ConceptGraph
    return ConceptGraph()

TypeName = str  # e.g. 'i8', 'u128', 'f32', 'f128', 'Prop', 'Conc', 'Vec3<f32>'
Value = Union[int, float, bool, str, complex, Any]

# Regex patterns for typed literals
_INT_PATTERN = re.compile(r'^(-?\d+)(?:_(i|u)(8|16|32|64|128))$')
_FLOAT_PATTERN = re.compile(r'^(-?\d+(?:\.\d*)?)(?:_(f)(32|64|128))$')

@dataclass
class LogicalBinding:
    """Represents a logical binding for symbolic operations"""
    name: str
    value: Any
    constraints: Set[str]

class TypedValue:
    """Wrapper for a value with an associated static type."""
    def __init__(self, type_name: TypeName, value: Value):
        self.type = type_name
        self.value = value

    def __repr__(self):
        return f"TypedValue(type={self.type}, value={self.value})"

class Opcode(Enum):
    NOP = auto(); HLT = auto()
    LOAD = auto(); LOADC = auto()
    STORE = auto(); STOREC = auto()
    PUSH = auto(); POP = auto()
    MOV = auto()
    ADD = auto(); SUB = auto(); MUL = auto(); DIV = auto()
    POW = auto(); SQRT = auto(); EXP = auto()
    SIN = auto(); COS = auto(); TAN = auto()
    ASIN = auto(); ACOS = auto(); ATAN = auto()
    LOG = auto(); LOG10 = auto(); LN = auto(); ERR = auto()
    ASSERT = auto(); JM = auto(); CJM = auto()
    CALL = auto(); RET = auto(); WAIT = auto(); SLEEP = auto()
    AND = auto(); OR = auto(); NOT = auto()
    XOR = auto(); NAND = auto(); NOR = auto()
    IMPLIES = auto(); IFF = auto(); FORALL = auto(); EXISTS = auto()
    UNIFY = auto(); RESOLVE = auto(); CONTRADICTS = auto()
    CORR = auto(); ENER = auto(); CASSERT = auto(); CBIND = auto(); CQUERY = auto()
    CAND = auto(); COR = auto(); CNOT = auto(); CIF = auto(); CCMP = auto(); CLEAN = auto()
    PAR = auto()

Instruction = Tuple[Opcode, List[str]]

class SymbolicEvaluator:
    """Handles symbolic computation and unification"""
    def __init__(self):
        self.bindings: Dict[str, LogicalBinding] = {}
    
    def unify(self, term1: str, term2: str) -> bool:
        # Implement unification algorithm
        if term1 in self.bindings and term2 in self.bindings:
            return self.bindings[term1].value == self.bindings[term2].value
        elif term1 in self.bindings:
            self.bindings[term2] = self.bindings[term1]
            return True
        elif term2 in self.bindings:
            self.bindings[term1] = self.bindings[term2]
            return True
        else:
            binding = LogicalBinding(term1, None, set())
            self.bindings[term1] = binding
            self.bindings[term2] = binding
            return True

class SillyVM:
    def __init__(self):
        self.code: List[Instruction] = []
        # General-purpose and concept registers
        self.regs: Dict[str, TypedValue] = {f"R{i}": TypedValue('i32', 0) for i in range(16)}
        self.regs.update({f"C{i}": TypedValue('Conc', None) for i in range(16)})
        self.concepts = get_concept_graph()
        self.ip: int = 0
        self.stack: List[TypedValue] = []
        self.call_stack: Deque[int] = deque()
        self.halted: bool = False
        self.symbolic = SymbolicEvaluator()
        self.type_registry = {}
        self.variables = {}  # Maps variable names to registers

    def parse(self, asm: str) -> None:
        lines = asm.splitlines()
        labels: Dict[str, int] = {}
        code_lines: List[str] = []
        current_routine = None
        param_bindings = {}
        self.variables = {}  # Reset variables for new parse
        
        # First pass: routines and labels
        for line in lines:
            clean = line.split('//')[0].strip()
            if not clean:
                continue
                
            # Handle variable declarations
            if clean.startswith('%'):
                var_decl = self.extract_var_declaration(clean)
                if var_decl:
                    name, reg, typ, init_val = var_decl
                    self.variables[name] = (reg, typ)
                    if init_val:
                        # Add initialization instruction
                        code_lines.append(f"MOV {init_val}, {reg}")
                    continue
                    
            # Handle routine declarations
            if clean.startswith('.'):
                if '{' in clean:
                    # Parse routine name and parameters
                    routine_decl = clean[1:].split('{')[0].strip()
                    routine_name = routine_decl.split('(')[0]
                    # Extract parameters if any
                    if '(' in routine_decl:
                        params_str = routine_decl[routine_decl.find('(')+1:routine_decl.rfind(')')]
                        params = [p.strip().split(':') for p in params_str.split(',') if p.strip()]
                        param_bindings[routine_name] = {
                            name.strip(): reg.strip() 
                            for name, reg in params
                        }
                    
                    labels[routine_name] = len(code_lines)
                    current_routine = routine_name
                    continue
                    
            elif clean == '}':
                current_routine = None
                self.variables.clear()  # Clear variables at routine end
                continue
                
            elif clean.endswith(':'):
                labels[clean[:-1]] = len(code_lines)
                
            else:
                # Handle routine calls with or without parentheses
                if not any(op in clean for op in Opcode.__members__):
                    # Extract routine name, removing parentheses if present
                    call_name = clean.split('(')[0].strip()
                    args = []
                    if '(' in clean:
                        args_str = clean[clean.find('(')+1:clean.rfind(')')]
                        args = [a.strip() for a in args_str.split(',') if a.strip()]
                    
                    # If routine has parameter bindings, add MOV instructions
                    if call_name in param_bindings:
                        param_regs = list(param_bindings[call_name].values())
                        for i, arg in enumerate(args):
                            if i < len(param_regs):
                                code_lines.append(f"MOV {arg}, {param_regs[i]}")
                    
                    code_lines.append(f"CALL {call_name}")
                    continue
                
                # Handle variable references in expressions
                if current_routine:
                    # Replace variable names with their registers
                    for var_name, (reg, _) in self.variables.items():
                        clean = re.sub(r'\b' + var_name + r'\b', reg, clean)
                
                code_lines.append(clean)
        
        # Second pass: instructions
        for line in code_lines:
            if line.startswith('PAR '):
                subs = [s.strip() for s in line[4:].split(';') if s.strip()]
                self.code.append((Opcode.PAR, subs))
                continue
                
            parts = re.split(r'[ ,]+', line)
            op_str, *args = parts
            
            try:
                op = Opcode[op_str]
            except KeyError:
                raise ValueError(f"Unknown opcode: {op_str}")
                
            if op_str in ('JM', 'CJM', 'CALL'):
                args = [str(labels.get(a, a)) for a in args]
                
            self.code.append((op, args))

    def fetch(self) -> Optional[Instruction]:
        if not (0 <= self.ip < len(self.code)):
            return None
        instr = self.code[self.ip]
        self.ip += 1
        return instr

    def _apply_int_range(self, val: int, typ: str) -> int:
        m = re.match(r'^([iu])(8|16|32|64|128)$', typ)
        if not m:
            return val
        signed, bits = m.group(1), int(m.group(2))
        mask = (1 << bits) - 1
        val &= mask
        if signed == 'i' and (val & (1 << (bits - 1))):
            val -= (1 << bits)
        return val

    def _cast_float(self, val: float, typ: str) -> float:
        if typ == 'f32':
            return struct.unpack('!f', struct.pack('!f', val))[0]
        # f64/f128 use Python's float
        return val

    def get_typed(self, operand: str) -> TypedValue:
        # Check if operand is a variable name
        if operand in self.variables:
            reg, typ = self.variables[operand]
            return self.regs[reg]
        
        if operand in self.regs:
            return self.regs[operand]
        m = _INT_PATTERN.match(operand)
        if m:
            num, signed, bits = m.groups()
            val = int(num)
            typ = f"{signed}{bits}"
            val = self._apply_int_range(val, typ)
            return TypedValue(typ, val)
        m2 = _FLOAT_PATTERN.match(operand)
        if m2:
            num, fchar, bits = m2.groups()
            val = float(num)
            typ = f"f{bits}"
            val = self._cast_float(val, typ)
            return TypedValue(typ, val)
        if operand in ('true', 'false'):
            return TypedValue('bool', operand == 'true')
        if operand.startswith('"') and operand.endswith('"'):
            return TypedValue('str', operand[1:-1])
        try:
            val = eval(operand, {}, {})
            if isinstance(val, int):
                return TypedValue('i32', val)
            if isinstance(val, float):
                return TypedValue('f64', val)
            if isinstance(val, complex):
                return TypedValue('complex', val)
        except Exception:
            pass
        raise ValueError(f"Unknown operand: {operand}")

    def set_reg(self, reg: str, tv: TypedValue) -> None:
        self.regs[reg] = tv

    def step(self) -> None:
        instr = self.fetch()
        if instr is None:
            self.halted = True
            return
        op, args = instr
        handler = getattr(self, f"op_{op.name.lower()}", None)
        if handler is None:
            raise NotImplementedError(f"No handler for {op}")
        result = handler(args)
        if asyncio.iscoroutine(result):
            asyncio.get_event_loop().run_until_complete(result)

    def run(self) -> None:
        while not self.halted:
            self.step()

    async def run_async(self) -> None:
        while not self.halted:
            await asyncio.sleep(0)
            self.step()

    @staticmethod
    def run_parallel(vms: List['SillyVM']) -> None:
        asyncio.run(asyncio.gather(*(vm.run_async() for vm in vms)))

    def run_bytecode_from_file(self, filepath: str) -> None:
        """
        Reads, parses and executes bytecode from a file.
        
        Args:
            filepath: Path to the .sisa or .txt file containing SillyISA bytecode
            
        Raises:
            FileNotFoundError: If the bytecode file doesn't exist
            ValueError: If the bytecode contains syntax errors
        """
        try:
            with open(filepath, 'r') as f:
                bytecode = f.read()
                
            try:
                self.parse(bytecode)  # Parse the bytecode
                self.run()           # Execute the bytecode
            except ValueError as e:
                raise ValueError(f"Error parsing bytecode: {str(e)}")
            except Exception as e:
                raise RuntimeError(f"Error executing bytecode: {str(e)}")
                
        except FileNotFoundError:
            raise FileNotFoundError(f"Bytecode file not found: {filepath}")

    # Opcode Handlers
    def op_nop(self, args: List[str]) -> None:
        pass

    def op_hlt(self, args: List[str]) -> None:
        self.halted = True

    def op_wait(self, args: List[str]) -> None:
        ms = int(args[0])
        time.sleep(ms / 1000)

    def op_sleep(self, args: List[str]) -> None:
        self.halted = True

    def op_load(self, args: List[str]) -> None:
        dest, addr = args
        tv = self.stack[int(addr)]
        self.set_reg(dest, tv)

    def op_store(self, args: List[str]) -> None:
        src, addr = args
        tv = self.get_typed(src)
        self.stack.insert(int(addr), tv)

    def op_loadc(self, args: List[str]) -> None:
        reg, name = args
        self.concepts.add_concept(name)
        self.set_reg(reg, TypedValue('Conc', name))

    def op_storec(self, args: List[str]) -> None:
        reg = args[0]
        name = self.get_typed(reg).value
        self.concepts.add_concept(name)

    def op_push(self, args: List[str]) -> None:
        tv = self.get_typed(args[0])
        self.stack.append(tv)

    def op_pop(self, args: List[str]) -> None:
        reg = args[0]
        tv = self.stack.pop()
        self.set_reg(reg, tv)

    def op_mov(self, args: List[str]) -> None:
        src, dest = args
        tv = self.get_typed(src)
        self.set_reg(dest, tv)

    def _binary_op(self, args: List[str], fn) -> None:
        v1 = self.get_typed(args[0])
        v2 = self.get_typed(args[1])
        dest = args[2]
        if v1.type != v2.type:
            raise TypeError(f"Type mismatch: {v1.type} vs {v2.type}")
        result = fn(v1.value, v2.value)
        typ = v1.type
        if typ.startswith(('i', 'u')):
            result = self._apply_int_range(int(result), typ)
        if typ.startswith('f'):
            result = self._cast_float(float(result), typ)
        self.set_reg(dest, TypedValue(typ, result))

    def op_add(self, args: List[str]) -> None:
        self._binary_op(args, lambda x, y: x + y)

    def op_sub(self, args: List[str]) -> None:
        self._binary_op(args, lambda x, y: x - y)

    def op_mul(self, args: List[str]) -> None:
        self._binary_op(args, lambda x, y: x * y)

    def op_div(self, args: List[str]) -> None:
        def safe_div(x, y):
            if y == 0:
                raise ZeroDivisionError("Division by zero")
            return x / y if not isinstance(x, int) else x // y
        self._binary_op(args, safe_div)

    def op_pow(self, args: List[str]) -> None:
        self._binary_op(args, math.pow)

    def op_sqrt(self, args: List[str]) -> None:
        src, dest = args
        v = self.get_typed(src)
        self.set_reg(dest, TypedValue(v.type, math.sqrt(v.value)))

    def op_exp(self, args: List[str]) -> None:
        src, dest = args
        v = self.get_typed(src)
        self.set_reg(dest, TypedValue(v.type, math.exp(v.value)))

    def op_sin(self, args: List[str]) -> None:
        src, dest = args
        v = self.get_typed(src)
        self.set_reg(dest, TypedValue(v.type, math.sin(v.value)))

    def op_cos(self, args: List[str]) -> None:
        src, dest = args
        v = self.get_typed(src)
        self.set_reg(dest, TypedValue(v.type, math.cos(v.value)))

    def op_tan(self, args: List[str]) -> None:
        src, dest = args
        v = self.get_typed(src)
        self.set_reg(dest, TypedValue(v.type, math.tan(v.value)))

    def op_asin(self, args: List[str]) -> None:
        src, dest = args
        v = self.get_typed(src)
        self.set_reg(dest, TypedValue(v.type, math.asin(v.value)))

    def op_acos(self, args: List[str]) -> None:
        src, dest = args
        v = self.get_typed(src)
        self.set_reg(dest, TypedValue(v.type, math.acos(v.value)))

    def op_atan(self, args: List[str]) -> None:
        src, dest = args
        v = self.get_typed(src)
        self.set_reg(dest, TypedValue(v.type, math.atan(v.value)))

    def op_log(self, args: List[str]) -> None:
        src, dest = args
        v = self.get_typed(src)
        self.set_reg(dest, TypedValue(v.type, math.log2(v.value)))

    def op_log10(self, args: List[str]) -> None:
        src, dest = args
        v = self.get_typed(src)
        self.set_reg(dest, TypedValue(v.type, math.log10(v.value)))

    def op_ln(self, args: List[str]) -> None:
        src, dest = args
        v = self.get_typed(src)
        self.set_reg(dest, TypedValue(v.type, math.log(v.value)))

    def op_err(self, args: List[str]) -> None:
        src, dest = args
        v = self.get_typed(src)
        self.set_reg(dest, TypedValue(v.type, math.erf(v.value)))

    def op_assert(self, args: List[str]) -> None:
        tv = self.get_typed(args[0])
        if not bool(tv.value):
            raise AssertionError("ASSERT failed")

    def op_jm(self, args: List[str]) -> None:
        self.ip = int(args[0])

    def op_cjm(self, args: List[str]) -> None:
        tv = self.get_typed(args[0])
        if bool(tv.value):
            self.ip = int(args[1])

    def op_call(self, args: List[str]) -> None:
        self.call_stack.append(self.ip)
        self.ip = int(args[0])

    def op_ret(self, args: List[str]) -> None:
        if self.call_stack:
            self.ip = self.call_stack.pop()
        else:
            self.halted = True

    def op_and(self, args: List[str]) -> None:
        v1 = self.get_typed(args[0])
        v2 = self.get_typed(args[1])
        dest = args[2]
        self.set_reg(dest, TypedValue('bool', v1.value and v2.value))

    def op_or(self, args: List[str]) -> None:
        v1 = self.get_typed(args[0])
        v2 = self.get_typed(args[1])
        dest = args[2]
        self.set_reg(dest, TypedValue('bool', v1.value or v2.value))

    def op_not(self, args: List[str]) -> None:
        v = self.get_typed(args[0])
        dest = args[1]
        self.set_reg(dest, TypedValue('bool', not v.value))

    def op_xor(self, args: List[str]) -> None:
        v1 = self.get_typed(args[0])
        v2 = self.get_typed(args[1])
        dest = args[2]
        self.set_reg(dest, TypedValue('bool', bool(v1.value) ^ bool(v2.value)))

    def op_nand(self, args: List[str]) -> None:
        v1 = self.get_typed(args[0])
        v2 = self.get_typed(args[1])
        dest = args[2]
        self.set_reg(dest, TypedValue('bool', not (v1.value and v2.value)))

    def op_nor(self, args: List[str]) -> None:
        v1 = self.get_typed(args[0])
        v2 = self.get_typed(args[1])
        dest = args[2]
        self.set_reg(dest, TypedValue('bool', not (v1.value or v2.value)))

    def op_implies(self, args: List[str]) -> None:
        v1 = self.get_typed(args[0])
        v2 = self.get_typed(args[1])
        dest = args[2]
        self.set_reg(dest, TypedValue('bool', (not v1.value) or v2.value))

    def op_iff(self, args: List[str]) -> None:
        v1 = self.get_typed(args[0])
        v2 = self.get_typed(args[1])
        dest = args[2]
        self.set_reg(dest, TypedValue('bool', v1.value == v2.value))

    def op_forall(self, args: List[str]) -> None:
        """Universal quantification"""
        var, domain, predicate = args
        result = True
        for val in self.evaluate_domain(domain):
            self.symbolic.bindings[var] = LogicalBinding(var, val, set())
            if not self.evaluate_predicate(predicate):
                result = False
                break
        self.set_reg('R0', TypedValue('bool', result))

    def op_exists(self, args: List[str]) -> None:
        """Existential quantification"""
        var, domain, predicate = args
        result = False
        for val in self.evaluate_domain(domain):
            self.symbolic.bindings[var] = LogicalBinding(var, val, set())
            if self.evaluate_predicate(predicate):
                result = True
                break
        self.set_reg('R0', TypedValue('bool', result))

    def op_unify(self, args: List[str]) -> None:
        """Unify two terms"""
        term1, term2, dest = args
        result = self.symbolic.unify(term1, term2)
        self.set_reg(dest, TypedValue('bool', result))

    def op_resolve(self, args: List[str]) -> None:
        """Logical resolution"""
        clause1, clause2, dest = args
        c1 = self.evaluate_clause(clause1)
        c2 = self.evaluate_clause(clause2)
        resolvent = self.resolve_clauses(c1, c2)
        self.set_reg(dest, TypedValue('Clause', resolvent))

    def op_contradicts(self, args: List[str]) -> None:
        """Check for contradictions"""
        stmt1, stmt2, dest = args
        s1 = self.evaluate_statement(stmt1)
        s2 = self.evaluate_statement(stmt2)
        result = s1.contradicts(s2)
        self.set_reg(dest, TypedValue('bool', result))

    def op_cbind(self, args: List[str]) -> None:
        """Bind concept to logical variable"""
        var, concept, dest = args
        success = self.concepts.bind_variable(var, concept)
        self.set_reg(dest, TypedValue('bool', success))

    def op_cand(self, args: List[str]) -> None:
        """Concept AND operation"""
        c1, c2, dest = args
        energy = min(self.concepts.get_energy(c1), 
                    self.concepts.get_energy(c2))
        self.set_reg(dest, TypedValue('f64', energy))

    def op_cor(self, args: List[str]) -> None:
        """Concept OR operation"""
        c1, c2, dest = args
        energy = max(self.concepts.get_energy(c1), 
                    self.concepts.get_energy(c2))
        self.set_reg(dest, TypedValue('f64', energy))

    def op_cnot(self, args: List[str]) -> None:
        """Concept NOT operation"""
        concept, dest = args
        energy = 1.0 - self.concepts.get_energy(concept)
        self.set_reg(dest, TypedValue('f64', energy))

    def op_cif(self, args: List[str]) -> None:
        """Conditional concept activation"""
        condition, concept, dest = args
        cond_val = self.get_typed(condition).value
        if cond_val:
            self.concepts.activate(concept)
            self.set_reg(dest, TypedValue('bool', True))
        else:
            self.set_reg(dest, TypedValue('bool', False))

    def op_ccmp(self, args: List[str]) -> None:
        """Compare concepts"""
        c1, c2, dest = args
        similarity = self.concepts.compare(c1, c2)
        self.set_reg(dest, TypedValue('f64', similarity))

    def op_clean(self, args: List[str]) -> None:
        self.concepts.update_n_cluster()
        self.concepts.propagate_energy()

    def extract_var_declaration(self, line: str) -> Optional[Tuple[str, str, str, Optional[str]]]:
        """
        Extracts variable declaration components.
        Format: %name: reg<type> [= value]
        Returns: (name, register, type, initial_value or None)
        """
        # Remove whitespace and comments
        line = line.split('//')[0].strip()
        if not line.startswith('%'):
            return None
            
        # Split declaration and initialization
        decl_parts = line[1:].split('=', 1)
        decl = decl_parts[0].strip()
        init_val = decl_parts[1].strip() if len(decl_parts) > 1 else None
        
        # Parse name, register and type
        try:
            name_reg, type_part = decl.split(':', 1)
            name = name_reg.strip()
            reg_type = type_part.strip()
            reg, typ = reg_type.split('<', 1)
            typ = typ.rstrip('>')
            return (name, reg.strip(), typ.strip(), init_val)
        except ValueError:
            return None