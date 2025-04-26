# TransAPI

An OpenAI compatible API **purely** based on Transformers.

> **Purely** means the inference engine backend only relies on Transformers. The project uses fastAPI for API server hosting.

Existing LLM inference engines like [vLLM](https://docs.vllm.ai/en/latest/) and [HuggingFace TGI](https://huggingface.co/docs/text-generation-inference/index) support many popular models, but some models still lack support. These unsupported models might also not be available through [GGUF](https://github.com/ggml-org/llama.cpp). To solve this, I built this project. It offers an OpenAI compatible API based entirely on Transformers, allowing you to run and test any model compatible with Transformers.

> [!NOTE]
> This project is NOT production ready. It's build for testing and development purposes only.

## Quick Start

**Clone the repository**:

```shell
git clone https://github.com/Orion-zhen/transAPI.git && cd transAPI
```

**Install packages**:

```shell
pip install -e .
```

Or use `pip install -e ".[extra]"` for extra functions, including vision support.

**Copy default config and modify it**:

```shell
cp config/config.yaml.sample config/config.yaml
```

**Run the server**:

```shell
python main.py
```

## Features

- [x] OpenAI compatible API
- [x] Native integration with Transformers
- [x] Native multimodal support (Text-to-Text and Image-Text-to-Text)
- [x] Reverse proxy support (e.g. Nginx)
- [x] Multiple sampling strategies including [beam search](https://huggingface.co/docs/transformers/main/en/generation_strategies#beam-search)
- [x] `torch.compile` for faster inference
- [x] `bitsandbytes` in-flight quantization
- [x] Various quantization formats supported by Transformers
- [x] Concurrent inference with asyncio
- [x] Streaming and non-streaming responses
- [x] Smart KV Cache offloading

## Limitations

- OpenAI style tool/function calling. I'm not familiar with tool call mechnism. If you have any suggestions or insights, please feel free to issue or PR.
- Continuous batching and paged attention. Integerating it with Transformers is literally impossible. Given that paged attention requires model-specific optimizations, we can't do it in a generic way.
- KV Cache quantization. Same reason as above.
