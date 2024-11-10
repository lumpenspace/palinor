from dataclasses import dataclass
from typing import Literal, Sequence


@dataclass
class Message:
    """
    A single message in a chat conversation.

    Attributes:
        role: The role of the speaker ("system", "user", or "assistant")
        content: The content of the message
    """

    role: Literal["system", "user", "assistant"]
    content: str

    def to_llama_string(self) -> str:
        """Convert message to Llama chat format."""
        if self.role == "system":
            return f"<s>[INST] <<SYS>>\n{self.content}\n<</SYS>>\n\n"
        elif self.role == "user":
            return f"[INST] {self.content} [/INST]"
        else:  # assistant
            return f"{self.content} </s>"


@dataclass
class DatasetEntry:
    """
    A pair of contrastive examples.

    Attributes:
        a: First sequence of messages
        b: Second sequence of messages
    """

    a: Sequence[Message]
    b: Sequence[Message]

    def to_llama_strings(self) -> tuple[str, str]:
        """Convert both message sequences to Llama chat format."""
        a_string = "".join(msg.to_llama_string() for msg in self.a)
        b_string = "".join(msg.to_llama_string() for msg in self.b)
        return a_string, b_string
