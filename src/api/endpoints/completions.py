from threading import Thread
from src.core.engine import InferenceEngine
from src.api.types.usage_info import UsageInfo
from fastapi.responses import StreamingResponse
from src.api.utils.format_sse import format_sse
from fastapi import APIRouter, HTTPException, Depends
from src.api.utils.dependencies import get_inference_engine
from transformers.generation.streamers import TextIteratorStreamer
from src.api.types.completions import (
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
)

router = APIRouter()


async def _stream_completion(request: CompletionRequest, engine: InferenceEngine):
    # init_choice = CompletionChoice(text="\n\n")
    # init_response = CompletionResponse(model=request.model, choices=[init_choice])
    # yield format_sse(init_response.model_dump())

    streamer = TextIteratorStreamer(
        engine.tokenizer,  # type: ignore
        skip_prompt=True,
        skip_special_tokens=True,
    )

    gen_cfg = request.gen_config()

    thread = Thread(
        target=engine.generate_completions,
        kwargs=dict(
            prompt=request.prompt, generation_config=gen_cfg, streamer=streamer
        ),
    )
    thread.start()

    finish_reason = None

    try:
        for text in streamer:
            if not text:
                continue

            # Construct Response Body
            choice = CompletionChoice(text=text, finish_reason=finish_reason)
            response = CompletionResponse(model=request.model, choices=[choice])
            yield format_sse(response.model_dump())
        finish_reason = "stop"
    except Exception:
        finish_reason = "error"
    finally:
        thread.join()
        final_choice = CompletionChoice(text="", finish_reason=finish_reason)
        final_response = CompletionResponse(model=request.model, choices=[final_choice])
        yield format_sse(final_response.model_dump())

        yield "data: [DONE]\n\n"  # finish sse


# Update main endpoint to inject engine
@router.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(
    request: CompletionRequest,
    engine: InferenceEngine = Depends(get_inference_engine),  # Inject engine
):
    """
    Endpoint for text completions. Handles both streaming and non-streaming.
    """
    if request.stream:
        return StreamingResponse(
            _stream_completion(request, engine), media_type="text/event-stream"
        )
    try:
        # Use the injected engine instance
        gen_cfg = request.gen_config()
        generated_response, num_prompt_tokens, num_completion_tokens = (
            engine.generate_completions(
                prompt=request.prompt, generation_config=gen_cfg
            )
        )

        # Construct Response Body
        usage_info = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_completion_tokens,
            total_tokens=num_prompt_tokens + num_completion_tokens,
        )
        choice = CompletionChoice(text=generated_response, finish_reason="stop")
        response = CompletionResponse(
            model=request.model, choices=[choice], usage=usage_info
        )
        return response

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Internal Server Error during generation. {e}"
        )
