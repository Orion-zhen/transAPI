import time
from pydantic import BaseModel, Field
from src.api.types.samplers import Samplers
from src.api.types.usage_info import UsageInfo
from transformers.generation.configuration_utils import GenerationConfig


class Message(BaseModel):
    role: str
    content: str | list[dict[str, str]]


class GeneratedMessage(BaseModel):
    role: str = "assistant"
    content: str
    reasoning_content: str | None = None


class ChatCompletionRequest(Samplers):
    model: str | None = None
    messages: list[Message]
    stream: bool = False

    def gen_config(self):
        exclude = set(["stream", "model", "messages"])
        return GenerationConfig(**self.model_dump(exclude=exclude, exclude_none=True))


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: GeneratedMessage | None = None
    delta: GeneratedMessage | None = None
    logprobs: str | None = None
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str = "transAPI"
    object: str = "chat_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str | None = None
    choices: list[ChatCompletionChoice]
    usage: UsageInfo | None = None
