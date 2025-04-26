import torch
import logging
from config.settings import AppSettings
from src.core.loader import load_model_with_settings
from transformers.generation.streamers import BaseStreamer
from transformers.tokenization_utils_base import BatchEncoding
from transformers.generation.configuration_utils import GenerationConfig


class InferenceEngine:
    def __init__(self, settings: AppSettings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger
        self.model, self.tokenizer, self.processor = load_model_with_settings(
            self.settings, self.logger
        )

    def _resilient_generate(
        self,
        generation_config: GenerationConfig | None = None,
        streamer: BaseStreamer | None = None,
        *args,
        **kwargs,
    ):
        oom = False
        try:
            return self.model.generate(
                use_model_defaults=True,
                generation_config=generation_config,
                streamer=streamer,
                tokenizer=self.tokenizer,
                *args,
                **kwargs,
            )
        except torch.cuda.OutOfMemoryError as e:
            self.logger.warning(f"OOM error during generation: {e}")
            oom = True
        if oom:
            torch.cuda.empty_cache()
            kwargs["cache_implementation"] = "offloaded"
            return self.model.generate(use_model_defaults=True, *args, **kwargs)

    def is_multimodal(
        self, conversation: list[dict[str, str]] | list[list[dict[str, str]]]
    ) -> bool:
        return any(
            isinstance(
                msg.get("content"), list
            )  # Check if 'content' exists and is a list
            and any(
                isinstance(item, dict)
                and item.get("type")
                in [
                    "image",
                    "video",
                ]  # Check if item is dict and has type 'image' or 'video'
                for item in msg["content"]  # Iterate through content list if it exists
            )
            # This part assumes msg is a dictionary in the outer list
            for msg in conversation
            if isinstance(msg, dict) and "content" in msg
        )

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]] | list[list[dict[str, str]]],
        add_generation_prompt: bool = True,
        tokenize: bool = True,
        add_special_tokens: bool = True,
        return_dict: bool = False,
        **kwargs,
    ) -> str | list[int] | dict | list[str] | list[list[int]] | BatchEncoding:
        if not self.is_multimodal(conversation=conversation):
            return self.tokenizer.apply_chat_template(
                conversation=conversation,
                add_generation_prompt=add_generation_prompt,
                tokenize=tokenize,
                add_special_tokens=add_special_tokens,
                return_dict=return_dict,
                **kwargs,
            )

        # for multimodal inputs
        try:
            return self.processor.apply_chat_template(
                conversation=conversation,
                add_generation_prompt=add_generation_prompt,
                tokenize=tokenize,
                add_special_tokens=add_special_tokens,
                return_dict=return_dict,
                **kwargs,
            )
        # keep text-only tokenizer as a fallback
        except Exception:
            return self.tokenizer.apply_chat_template(
                conversation=conversation,
                add_generation_prompt=add_generation_prompt,
                tokenize=tokenize,
                add_special_tokens=add_special_tokens,
                return_dict=return_dict,
                **kwargs,
            )

    def generate_completions(
        self,
        prompt: str,
        generation_config: GenerationConfig | None = None,
        streamer: BaseStreamer | None = None,
        **kwargs,
    ) -> tuple[str, int, int]:
        if self.settings.log.prompt:
            self.logger.info(f"Prompt: {prompt}")
        if self.settings.log.params:
            self.logger.info(f"Generation config: {generation_config}")

        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_ids = model_inputs.get("input_ids")
        assert input_ids is not None, "Input IDs are missing"
        num_prompt_tokens = len(input_ids[0])

        all_outputs = self._resilient_generate(
            **model_inputs,
            generation_config=generation_config,
            streamer=streamer,
            **kwargs,
        )
        assert all_outputs is not None, "Model generation failed"
        num_completion_tokens = len(all_outputs[0]) - num_prompt_tokens

        generated_tokens = all_outputs[0][num_prompt_tokens:]
        generated_texts = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        if self.settings.log.completion:
            self.logger.info(f"Completion: {generated_texts}")

        return generated_texts, num_prompt_tokens, num_completion_tokens

    def generate_chat_completions(
        self,
        conversation: list[dict[str, str]] | list[list[dict[str, str]]],
        generation_config: GenerationConfig | None = None,
        streamer: BaseStreamer | None = None,
        **kwargs,
    ):
        if self.settings.log.prompt:
            self.logger.info(f"Conversation: {conversation}")
        if self.settings.log.params:
            self.logger.info(f"Generation Config: {generation_config}")

        processed_chat = self.apply_chat_template(
            conversation=conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        processed_chat.to(self.model.device)  # type: ignore
        input_ids = processed_chat.get("input_ids")  # type: ignore
        assert input_ids is not None, "processed_chat produced None input_ids"
        num_prompt_tokens = len(input_ids[0])
        # print(processed_chat)

        all_outputs = self._resilient_generate(
            **processed_chat,  # type: ignore
            generation_config=generation_config,
            streamer=streamer,
            **kwargs,
        )
        assert all_outputs is not None, "Model generation failed"
        num_completion_tokens = len(all_outputs[0]) - num_prompt_tokens

        generated_tokens = all_outputs[0][num_prompt_tokens:]

        generated_text = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )

        if self.settings.log.completion:
            self.logger.info(f"Completion: {generated_text}")

        return generated_text, num_prompt_tokens, num_completion_tokens

    def tokenize(self, prompt: str) -> list[int]:
        return self.tokenizer.encode(prompt)

    def detokenize(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)
