import re
import asyncio
import struct
import math
import time
from collections import deque
from enum import Enum, auto
from typing import Any, Dict, List, Tuple, Union, Optional, Deque

# Initialize concept graph

def get_concept_graph():
    from sillyai.concept import ConceptGraph
    return ConceptGraph()

TypeName = str  # e.g. 'i8', 'u128', 'f32', 'f128', 'Prop', 'Conc', 'Vec3<f32>'
Value = Union[int, float, bool, str, complex, Any]

# Regex patterns for typed literals
_INT_PATTERN = re.compile(r'^(-?\d+)(?:_(i|u)(8|16|32|64|128))$')
_FLOAT_PATTERN = re.compile(r'^(-?\d+(?:\.\d*)?)(?:_(f)(32|64|128))$')

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

    def parse(self, asm: str) -> None:
        lines = asm.splitlines()
        labels: Dict[str, int] = {}
        code_lines: List[str] = []
        # First pass: labels
        for line in lines:
            clean = line.split('//')[0].strip()
            if not clean:
                continue
            if clean.endswith(':'):
                labels[clean[:-1]] = len(code_lines)
            else:
                code_lines.append(clean)
        # Second pass: instructions
        for line in code_lines:
            if line.startswith('PAR '):
                subs = [s.strip() for s in line[4:].split(';') if s.strip()]
                self.code.append((Opcode.PAR, subs))
                continue
            parts = re.split(r'[ ,]+', line)
            op_str, *args = parts
            if op_str in ('JM', 'CJM', 'CALL'):
                args = [str(labels.get(a, a)) for a in args]
            try:
                op = Opcode[op_str]
            except KeyError:
                raise ValueError(f"Unknown opcode: {op_str}")
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
        raise NotImplementedError("FORALL not implemented yet")

    def op_exists(self, args: List[str]) -> None:
        raise NotImplementedError("EXISTS not implemented yet")

    def op_unify(self, args: List[str]) -> None:
        raise NotImplementedError("UNIFY not implemented")

    def op_resolve(self, args: List[str]) -> None:
        raise NotImplementedError("RESOLVE not implemented")

    def op_contradicts(self, args: List[str]) -> None:
        raise NotImplementedError("CONTRADICTS not implemented")

    def op_corr(self, args: List[str]) -> None:
        src, tgt, w = args
        self.concepts.add_connection(src, tgt, float(w))

    def op_ener(self, args: List[str]) -> None:
        name, val = args
        self.concepts.add_concept(name)
        self.concepts.concepts[name].energy = float(val)

    def op_cassert(self, args: List[str]) -> None:
        self.concepts.add_concept(args[0])

    def op_cbind(self, args: List[str]) -> None:
        raise NotImplementedError("CBIND not implemented")

    def op_cquery(self, args: List[str]) -> None:
        name, dest = args
        energy = self.concepts.concepts.get(name, type('X', (), {'energy': 0.0})).energy
        self.set_reg(dest, TypedValue('f64', energy))

    def op_cand(self, args: List[str]) -> None:
        raise NotImplementedError("CAND not implemented")

    def op_cor(self, args: List[str]) -> None:
        raise NotImplementedError("COR not implemented")

    def op_cnot(self, args: List[str]) -> None:
        raise NotImplementedError("CNOT not implemented")

    def op_cif(self, args: List[str]) -> None:
        raise NotImplementedError("CIF not implemented")

    def op_ccmp(self, args: List[str]) -> None:
        raise NotImplementedError("CCMP not implemented")

    def op_clean(self, args: List[str]) -> None:
        self.concepts.update_n_cluster()
        self.concepts.propagate_energy()