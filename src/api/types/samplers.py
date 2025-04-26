from pydantic import BaseModel, Field
from transformers.generation.configuration_utils import GenerationConfig


class Samplers(BaseModel):
    do_sample: bool | None = True
    max_length: int = Field(2048, alias="max_tokens")
    max_new_tokens: int | None = Field(None, alias="max_completion_tokens")
    stop_strings: str | list[str] | None = Field(None, alias="stop")
    num_beams: int | None = None
    num_beam_groups: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None

    def gen_config(self):
        exclude = set(["stream", "model", "messages", "prompt"])
        return GenerationConfig(**self.model_dump(exclude=exclude, exclude_none=True))
