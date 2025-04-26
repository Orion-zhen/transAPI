from pydantic import BaseModel


class TokenizeRequest(BaseModel):
    model: str | None = None
    prompt: str


class TokenizeResponse(BaseModel):
    tokens: list[int]
    count: int


class DetokenizeRequest(BaseModel):
    model: str | None = None
    tokens: list[int]


class DetokenizeResponse(BaseModel):
    prompt: str
