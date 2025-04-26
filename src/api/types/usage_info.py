from pydantic import BaseModel


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int | None = None  # Optional for streaming chunks
    total_tokens: int
