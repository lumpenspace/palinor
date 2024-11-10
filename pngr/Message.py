from dataclasses import dataclass, asdict
from typing import List, Optional, Any


@dataclass
class Message:
    """
    A single message in a chat conversation.

    Attributes:
        role: The role of the speaker ("system", "user", or "assistant")
        content: The content of the message
    """

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        """Convert message to dictionary."""
        return asdict(self)


@dataclass
class DatasetEntry:
    """
    A pair of contrastive examples.

    Attributes:
        a: First sequence of messages
        b: Second sequence of messages
        a_trait: Trait for the first sequence
        b_trait: Trait for the second sequence
    """

    a: List[Message]
    b: List[Message]
    a_trait: Optional[str] = None
    b_trait: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert entry to dictionary."""
        return {
            "a": [m.to_dict() for m in self.a],
            "b": [m.to_dict() for m in self.b],
            "a_trait": self.a_trait,
            "b_trait": self.b_trait,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetEntry":
        """Create a DatasetEntry instance from a dictionary."""
        return cls(
            a=[Message(**m) for m in data["a"]],
            b=[Message(**m) for m in data["b"]],
            a_trait=data.get("a_trait"),
            b_trait=data.get("b_trait"),
        )
