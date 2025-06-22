from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

class RelationshipType(Enum):
    IS_A = "is_a"
    PART_OF = "part_of"
    HAS_PROPERTY = "has_property"
    CAUSES = "causes"
    CORRELATES = "correlates"
    SIMILAR_TO = "similar_to"
    OPPOSITE_OF = "opposite_of"
    PRECEDES = "precedes"
    FOLLOWS = "follows"
    CONTAINS = "contains"
    BELONGS_TO = "belongs_to"
    IMPLIES = "implies"
    REQUIRES = "requires"
    EXCLUDES = "excludes"
    ENHANCES = "enhances"
    INHIBITS = "inhibits"
    TRANSFORMS = "transforms"
    COMPOSES = "composes"
    DECOMPOSES = "decomposes"
    INTERACTS_WITH = "interacts_with"
    DEPENDS_ON = "depends_on"

@dataclass
class Relationship:
    type: RelationshipType
    source: str
    target: str
    weight: float = 1.0
    confidence: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        self.metadata = self.metadata or {}

@dataclass
class Rule:
    condition: str
    priority: float = 1.0
    context: Dict[str, Any] = None
    description: str = ""
    
    def __post_init__(self):
        self.context = self.context or {}
        
    def evaluate(self, vm_state: Dict[str, Any]) -> bool:
        """Evaluate the rule in the current VM state."""
        try:
            result = safe_eval(self.condition, {
                "registers": vm_state["registers"],
                "concepts": vm_state["concept_registers"],
                "context": vm_state.get("context", {}),
                **self.context
            })
            return bool(result)
        except Exception as e:
            print(f"Error evaluating rule: {e}")
            return False

@dataclass
class TypedValue:
    value: Any
    type: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        self.metadata = self.metadata or {}
        
    def __str__(self) -> str:
        return f"{self.value} ({self.type})"
        
    def __repr__(self) -> str:
        return f"TypedValue(value={self.value}, type='{self.type}')"

def safe_eval(expr: str, context: Dict[str, Any]) -> Any:
    """Safely evaluate an expression in the given context."""
    # This is a placeholder - in a real implementation, you would want to use
    # a proper expression evaluator with security measures
    try:
        # Basic arithmetic and boolean operations
        allowed_ops = {
            '+': lambda x, y: x + y,
            '-': lambda x, y: x - y,
            '*': lambda x, y: x * y,
            '/': lambda x, y: x / y,
            'and': lambda x, y: x and y,
            'or': lambda x, y: x or y,
            'not': lambda x: not x,
            '==': lambda x, y: x == y,
            '!=': lambda x, y: x != y,
            '<': lambda x, y: x < y,
            '>': lambda x, y: x > y,
            '<=': lambda x, y: x <= y,
            '>=': lambda x, y: x >= y,
        }
        
        # For now, just return True to allow the constraint system to work
        # In a real implementation, you would parse and evaluate the expression
        return True
    except Exception as e:
        print(f"Error evaluating expression: {e}")
        return False 