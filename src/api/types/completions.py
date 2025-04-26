import time
from pydantic import BaseModel, Field
from src.api.types.samplers import Samplers
from src.api.types.usage_info import UsageInfo
from transformers.generation.configuration_utils import GenerationConfig


class CompletionRequest(Samplers):
    model: str | None = None
    prompt: str
    stream: bool = False

    def gen_config(self):
        exclude = set(["stream", "model", "prompt"])
        return GenerationConfig(**self.model_dump(exclude=exclude, exclude_none=True))


class CompletionChoice(BaseModel):
    text: str
    index: int = 0
    logprobs: str | None = None
    finish_reason: str | None = None  # None for stream chunks until final


class CompletionResponse(BaseModel):
    id: str = "transAPI"
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str | None = None
    choices: list[CompletionChoice]
    usage: UsageInfo | None = None  # None for streaming chunks until final
