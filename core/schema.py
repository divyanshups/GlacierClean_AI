from dataclasses import dataclass, field
from typing import Any

@dataclass
class CleaningAction:
    #WHERE
    column: str
    
    #WHAT
    operation: str
    
    #HOW
    params: dict[str, Any] = field(default_factory=dict)
    
    #WHO & WHY
    source: str = "manual"
    rationale: str = ""


@dataclass
class ExecutionLog:
    column: str
    operation: str
    status: str
    message: str