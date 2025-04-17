import re
import asyncio

class BytecodeEngine:
    def __init__(self):
        # The list to hold the bytecode instructions
        self.code = []

        # The stack for holding values
        self.stack = []

        # The registers for holding temporary results
        self.regs = [0] * 10  # Let's assume we have 10 general-purpose registers for simplicity

        # Instruction pointer
        self.ip = 0  # Program Counter (Instruction Pointer)

    def load_code(self, bytecode):
        """Load bytecode into the engine."""
        self.code = bytecode.splitlines()
        self.ip = 0  # Reset instruction pointer

    def fetch(self):
        """Fetch the current instruction."""
        if self.ip >= len(self.code):
            return None  # End of code
        instruction = self.code[self.ip]
        self.ip += 1
        return instruction.strip()

    def decode(self, instruction):
        """Decode the instruction into a tuple of operation and operands."""
        parts = re.split(r'\s+', instruction)
        op = parts[0]
        args = parts[1:]
        return op, args

    def execute(self, op, args):
        """Execute a decoded instruction."""
        if op == "ADD":
            self.add(args)
        elif op == "MUL":
            self.mul(args)
        elif op == "MOV":
            self.mov(args)
        elif op == "PUSH":
            self.push(args)
        elif op == "POP":
            self.pop(args)
        elif op == "RET":
            self.ret(args)
        else:
            raise ValueError(f"Unknown instruction: {op}")

    def add(self, args):
        """Perform addition: ADD x, y, Rz"""
        x, y, rz = args
        x_val = self.get_value(x)
        y_val = self.get_value(y)
        self.regs[int(rz[1])] = x_val + y_val

    def sub(self, args):
        """Perform subtraction: SUB x, y, Rz"""
        x, y, rz = args
        x_val = self.get_value(x)
        y_val = self.get_value(y)
        self.regs[int(rz[1])] = x_val - y_val

    def mul(self, args):
        """Perform multiplication: MUL x, y, Rz"""
        x, y, rz = args
        x_val = self.get_value(x)
        y_val = self.get_value(y)
        self.regs[int(rz[1])] = x_val * y_val

    def div(self, args):
        """Perform division: DIV x, y, Rz"""
        x, y, rz = args
        x_val = self.get_value(x)
        y_val = self.get_value(y)
        self.regs[int(rz[1])] = x_val // y_val  # Integer division

    def mov(self, args):
        """Move value: MOV x, Ry"""
        x, ry = args
        self.regs[int(ry[1])] = self.get_value(x)

    def store(self, args):
        """Store value: STORE Ry, x"""
        ry, x = args
        self.regs[int(ry[1])] = self.get_value(x)

    def push(self, args):
        """Push a value onto the stack: PUSH x"""
        x = args[0]
        self.stack.append(self.get_value(x))

    def pop(self, args):
        """Pop a value from the stack: POP Ry"""
        ry = args[0]
        self.regs[int(ry[1])] = self.stack.pop()

    def ret(self, args):
        """Return from procedure (just for simulation)."""
        return

    def get_value(self, value):
        """Resolve a value (either a register or immediate)."""
        if value.startswith("R"):
            return self.regs[int(value[1])]
        else:
            return int(value)  # Assuming it's an immediate integer

    def run(self):
        """Run the bytecode instructions asynchronously."""
        while self.ip < len(self.code):
            instruction = self.fetch()
            if instruction:
                op, args = self.decode(instruction)
                self.execute(op, args)
            else:
                break
        return self.regs

    async def async_run(self):
        """Run bytecode asynchronously, allowing for non-blocking operations."""
        while self.ip < len(self.code):
            instruction = self.fetch()
            if instruction:
                op, args = self.decode(instruction)
                self.execute(op, args)
                await asyncio.sleep(0)  # Yield control to other tasks
            else:
                break
        return self.regs
    
    def __str__(self):
        return f"VM: {self.regs}\n Stack: {self.stack}\nCode:\n" + "\n".join(self.code) + f"\nIP: {self.ip}"