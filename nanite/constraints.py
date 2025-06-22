from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import math
import logging

from .types import RelationshipType, Rule, TypedValue

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler with formatting
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def safe_eval(expression: str, context: dict[str, Any] | None = None) -> Any:
    allowed_builtins = {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "round": round,
        "pow": pow,
        "math": math,
    }
    context = context or {}
    context["__builtins__"] = allowed_builtins
    return eval(expression, context)

class ConstraintType(Enum):
    PRE_CONDITION = "pre"
    POST_CONDITION = "post"
    INVARIANT = "invariant"
    TEMPORAL = "temporal"
    CONTEXT = "context"
    RELATIONSHIP = "relationship"
    CAUSAL = "causal"
    LOGICAL = "logical"
    GEOMETRIC = "geometric"

class LogicalConstraint:
    """Base class for logical constraints."""
    def __init__(self, 
                 constraint_type: ConstraintType,
                 condition: str,
                 priority: float = 1.0,
                 context: Dict[str, Any] = None,
                 description: str = "",
                 satisfaction_history: List[bool] = None,
                 rules: List[Rule] = None):
        self.constraint_type = constraint_type
        self.condition = condition
        self.priority = priority
        self.context = context or {}
        self.description = description
        self.satisfaction_history = satisfaction_history or []
        self.rules = rules or []
        
    def evaluate(self, vm_state: Dict[str, Any]) -> bool:
        """Evaluate the constraint in the current VM state."""
        try:
            # First check if any rules apply
            for rule in self.rules:
                if rule.evaluate(vm_state):
                    return True
                    
            # Then evaluate the condition
            result = safe_eval(self.condition, {
                "registers": vm_state["registers"],
                "concepts": vm_state["concept_registers"],
                "context": vm_state.get("context", {}),
                "args": vm_state.get("args", []),
                **self.context
            })
            self.satisfaction_history.append(bool(result))
            return bool(result)
        except Exception as e:
            logger.error(f"Error evaluating logical constraint: {str(e)}")
            return False

class RelationshipConstraint(LogicalConstraint):
    """Constraint for concept relationships."""
    def __init__(self,
                 relationship_type: RelationshipType,
                 source_concept: str,
                 target_concept: str,
                 min_weight: float = 0.0,
                 max_weight: float = 1.0,
                 priority: float = 1.0,
                 description: str = "",
                 satisfaction_history: List[bool] = None,
                 rules: List[Rule] = None):
        super().__init__(
            constraint_type=ConstraintType.RELATIONSHIP,
            condition="",
            priority=priority,
            description=description,
            satisfaction_history=satisfaction_history,
            rules=rules
        )
        self.relationship_type = relationship_type
        self.source_concept = source_concept
        self.target_concept = target_concept
        self.min_weight = min_weight
        self.max_weight = max_weight
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate relationship constraint."""
        try:
            # Get concepts from context
            source = context.get("source_concept")
            target = context.get("target_concept")
            if not source or not target:
                return False
                
            # Check relationship type
            if source.relationship_type != self.relationship_type:
                return False
                
            # Check weight bounds
            weight = source.weight
            if not (self.min_weight <= weight <= self.max_weight):
                return False
                
            # Check target concept
            if source.target != self.target_concept:
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error evaluating relationship constraint: {str(e)}")
            return False

class CausalConstraint(LogicalConstraint):
    """Constraint for causal relationships."""
    def __init__(self,
                 cause: str,
                 effect: str,
                 min_confidence: float = 0.0,
                 max_delay: Optional[int] = None,
                 priority: float = 1.0,
                 description: str = "",
                 satisfaction_history: List[bool] = None,
                 rules: List[Rule] = None):
        super().__init__(
            constraint_type=ConstraintType.CAUSAL,
            condition="",
            priority=priority,
            description=description,
            satisfaction_history=satisfaction_history,
            rules=rules
        )
        self.cause = cause
        self.effect = effect
        self.min_confidence = min_confidence
        self.max_delay = max_delay
    
    def evaluate(self, vm_state: Dict[str, Any]) -> bool:
        """Evaluate causal constraint."""
        try:
            # Get causal relationship from VM state
            causal = vm_state.get("causal_relationships", {}).get(
                (self.cause, self.effect)
            )
            
            if not causal:
                return False
                
            # Check confidence
            if causal["confidence"] < self.min_confidence:
                return False
                
            # Check delay if specified
            if self.max_delay is not None:
                if causal["delay"] > self.max_delay:
                    return False
                    
            # Evaluate additional conditions
            return super().evaluate(vm_state)
        except Exception as e:
            logger.error(f"Error evaluating causal constraint: {str(e)}")
            return False

class GeometricConstraint(LogicalConstraint):
    """Constraint for geometric properties."""
    def __init__(self,
                 basis: str,
                 grade: int,
                 tolerance: float = 1e-6,
                 priority: float = 1.0,
                 description: str = "",
                 satisfaction_history: List[bool] = None,
                 rules: List[Rule] = None):
        super().__init__(
            constraint_type=ConstraintType.GEOMETRIC,
            condition="",
            priority=priority,
            description=description,
            satisfaction_history=satisfaction_history,
            rules=rules
        )
        self.basis = basis
        self.grade = grade
        self.tolerance = tolerance
    
    def evaluate(self, vm_state: Dict[str, Any]) -> bool:
        """Evaluate geometric constraint."""
        try:
            # Get geometric value from VM state
            value = vm_state.get("geometric_values", {}).get(self.basis)
            
            if not value:
                return False
                
            # Check grade
            if value.grade != self.grade:
                return False
                
            # Check additional geometric properties
            return super().evaluate(vm_state)
        except Exception as e:
            logger.error(f"Error evaluating geometric constraint: {str(e)}")
            return False

class TemporalConstraint(LogicalConstraint):
    """Constraint for temporal properties."""
    def __init__(self,
                 start_time: Optional[int] = None,
                 end_time: Optional[int] = None,
                 period: Optional[int] = None,
                 priority: float = 1.0,
                 description: str = "",
                 satisfaction_history: List[bool] = None,
                 rules: List[Rule] = None):
        super().__init__(
            constraint_type=ConstraintType.TEMPORAL,
            condition="",
            priority=priority,
            description=description,
            satisfaction_history=satisfaction_history,
            rules=rules
        )
        self.start_time = start_time
        self.end_time = end_time
        self.period = period
    
    def evaluate(self, vm_state: Dict[str, Any]) -> bool:
        """Evaluate temporal constraint."""
        current_time = vm_state.get("time", 0)
        
        # Check temporal bounds
        if self.start_time is not None and current_time < self.start_time:
            return True
        if self.end_time is not None and current_time > self.end_time:
            return True
            
        # Check periodicity
        if self.period is not None:
            if current_time % self.period != 0:
                return True
                
        return super().evaluate(vm_state)

class ConstraintChecker:
    def __init__(self):
        self.constraints: Dict[str, List[LogicalConstraint]] = {}
        self.violation_handlers: Dict[str, callable] = {}
        
    def add_constraint(self, 
                      name: str, 
                      constraint: LogicalConstraint,
                      violation_handler: Optional[callable] = None):
        """Add a constraint with an optional violation handler."""
        if name not in self.constraints:
            self.constraints[name] = []
        self.constraints[name].append(constraint)
        
        if violation_handler:
            self.violation_handlers[name] = violation_handler
            
    def check_constraints(self, 
                         name: str, 
                         vm_state: Dict[str, Any]) -> bool:
        """Check all constraints for a given name."""
        if name not in self.constraints:
            return True
            
        for constraint in self.constraints[name]:
            if not constraint.evaluate(vm_state):
                if name in self.violation_handlers:
                    self.violation_handlers[name](constraint, vm_state)
                return False
        return True
        
    def get_relevant_constraints(self, 
                               op: str, 
                               context: Dict[str, Any]) -> List[LogicalConstraint]:
        """Get constraints relevant to the current operation and context."""
        relevant = []
        for name, constraints in self.constraints.items():
            if self._is_relevant(name, op, context):
                relevant.extend(constraints)
        return sorted(relevant, key=lambda c: c.priority, reverse=True)
        
    def _is_relevant(self, 
                    name: str, 
                    op: str, 
                    context: Dict[str, Any]) -> bool:
        """Determine if a constraint is relevant to the current operation."""
        # Check if constraint name matches operation
        if name == op:
            return True
            
        # Check if constraint is in current context
        if name in context.get("active_constraints", []):
            return True
            
        # Check if constraint is a pre/post condition for the operation
        if name.startswith(f"{op}_") and name.endswith(("_pre", "_post")):
            return True
            
        return False

class ContextTracker:
    def __init__(self):
        self.context_stack: List[Dict[str, Any]] = [{}]
        self.context_history: List[Dict[str, Any]] = []
        self.time: int = 0
        
    def update_context(self, op: str, args: List[str]):
        """Update the current context based on operation and arguments."""
        current_context = self.context_stack[-1].copy()
        
        # Update time
        self.time += 1
        current_context["time"] = self.time
        
        # Update based on operation type
        if op.startswith("CALL"):
            self._handle_call_context(op, args, current_context)
        elif op.startswith(("JM", "CJM")):
            self._handle_jump_context(op, args, current_context)
        elif op.startswith(("CASSERT", "ASSERT")):
            self._handle_assertion_context(op, args, current_context)
            
        self.context_stack.append(current_context)
        self.context_history.append(current_context)
        
    def get_current_context(self) -> Dict[str, Any]:
        """Get the current context."""
        return self.context_stack[-1]
        
    def _handle_call_context(self, op: str, args: List[str], context: Dict[str, Any]):
        """Handle context updates for call operations."""
        context["call_depth"] = context.get("call_depth", 0) + 1
        context["call_stack"] = context.get("call_stack", []) + [args[0]]
        
    def _handle_jump_context(self, op: str, args: List[str], context: Dict[str, Any]):
        """Handle context updates for jump operations."""
        context["jump_target"] = args[-1]
        if op.startswith("CJM"):
            context["jump_condition"] = args[0]
            
    def _handle_assertion_context(self, op: str, args: List[str], context: Dict[str, Any]):
        """Handle context updates for assertion operations."""
        context["assertions"] = context.get("assertions", []) + [args[0]]
        if op.startswith("CASSERT"):
            context["concept_assertions"] = context.get("concept_assertions", []) + [args[0]] 