from datetime import datetime
from typing import Any, Dict, Optional

class NaniteError(Exception):
    """Base exception class for Nanite."""
    pass

class ConstraintViolationError(NaniteError):
    """Exception raised when a constraint is violated during VM execution."""
    
    def __init__(self, message: str, constraint: Optional[Any] = None, 
                 vm_state: Optional[Dict[str, Any]] = None):
        self.message = message
        self.constraint = constraint
        self.vm_state = vm_state
        self.timestamp = datetime.now()
        
        # Build detailed error message
        details = [f"Constraint Violation: {message}"]
        
        if constraint:
            details.extend([
                f"\nConstraint Details:",
                f"- Type: {constraint.constraint_type.value}",
                f"- Priority: {constraint.priority}",
                f"- Description: {constraint.description}",
                f"- Condition: {constraint.condition}"
            ])
            
            if hasattr(constraint, 'satisfaction_history') and constraint.satisfaction_history:
                details.append(f"- Satisfaction History: {constraint.satisfaction_history[-5:]}")
                
        if vm_state:
            details.extend([
                f"\nVM State:",
                f"- Registers: {list(vm_state.get('registers', {}).keys())}",
                f"- Concept Registers: {list(vm_state.get('concept_registers', {}).keys())}",
                f"- Stack Size: {len(vm_state.get('stack', []))}",
                f"- Context: {vm_state.get('context', {})}"
            ])
            
        super().__init__("\n".join(details))

class VMError(NaniteError):
    """Base exception class for VM-related errors."""
    pass

class InvalidOpcodeError(VMError):
    """Exception raised when an invalid opcode is encountered."""
    pass

class InvalidOperandError(VMError):
    """Exception raised when an invalid operand is encountered."""
    pass

class StackError(VMError):
    """Exception raised for stack-related errors (overflow/underflow)."""
    pass

class MemoryError(VMError):
    """Exception raised for memory-related errors."""
    pass 