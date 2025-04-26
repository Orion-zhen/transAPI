import time
from pydantic import BaseModel, Field


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "transAPI"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard] = []
