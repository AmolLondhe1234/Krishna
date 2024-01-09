from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

class ModelChat(BaseModel):
    prompt : str
    created_at : Optional[datetime] = Field( default_factory=lambda: datetime.now())