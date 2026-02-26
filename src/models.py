from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from typing import List, Union, Tuple, Optional, Type
from fastapi import Query, Form

# @dataclass
class Message(BaseModel):
    message: str = ""
    sender_id: str
    role: str
    message_id: str
    