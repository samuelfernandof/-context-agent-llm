from dataclasses import dataclass
from typing import List, Literal

@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str

@dataclass(frozen=True)
class Event:
    type: str
    data: dict
