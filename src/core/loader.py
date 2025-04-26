import torch
import logging
from config.settings import AppSettings
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import ProcessorMixin
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def load_model_with_settings(
    settings: AppSettings,
    logger: logging.Logger,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin]:
    model_path = settings.model.model_path
    precision = settings.model.precision
    device = settings.model.device

    if not model_path:
        raise ValueError("Model name must be provided in the configuration")

    logger.info(f"Loading model {model_path}")

    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "auto": "auto",
    }.get(precision, "auto")

    quant_cfg = None
    if settings.model.quantization.bnb_4bit or settings.model.quantization.bnb_8bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_8bit=(
                settings.model.quantization.bnb_8bit
                and not settings.model.quantization.bnb_4bit
            ),
            llm_int8_enable_fp32_cpu_offload=True,
            load_in_4bit=settings.model.quantization.bnb_4bit,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )

    try:
        if quant_cfg:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                quantization_config=quant_cfg,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        assert isinstance(model, PreTrainedModel)

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if tokenizer.pad_token is None:
            if model.config.eos_token is not None:
                tokenizer.pad_token = model.config.eos_token
                logger.info(f"Set pad_token to eos_token {tokenizer.pad_token_id}")
            elif model.config.unk_token is not None:
                tokenizer.pad_token = model.config.unk_token
                logger.info(f"Set pad_token to unk_token {tokenizer.pad_token_id}")
            else:
                logger.warning("No pad token found, using default")

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        model = torch.compile(
            model, dynamic=True, fullgraph=True, mode="reduce-overhead"
        )

        logger.info("Model loaded and compiled successfully")

        return model, tokenizer, processor  # type: ignore

    except Exception as e:
        raise RuntimeError(f"Error initializing model and tokenizer: {e}")
