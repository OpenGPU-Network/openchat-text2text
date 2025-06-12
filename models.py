from typing import List

from pydantic import BaseModel


class Message(BaseModel):
    role: str = "user"
    content: str = "what is the capital of France?"

class InputData(BaseModel):
    messages: List[Message]