from enum import Enum

class Primitive(str, Enum):
    RULE_HEAVY = "rule_heavy"       
    ADDRESS_HEAVY = "address_heavy" 
    HYBRID = "hybrid"               

TASK_PRIMITIVE: dict[str, Primitive] = {
    "addition":    Primitive.RULE_HEAVY,
    "sorting":     Primitive.RULE_HEAVY,
    "dyck":        Primitive.RULE_HEAVY,
    "kv":          Primitive.ADDRESS_HEAVY,
    "indexing":    Primitive.ADDRESS_HEAVY,
    "func":        Primitive.HYBRID,
}