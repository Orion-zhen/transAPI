from threading import Thread
from src.core.engine import InferenceEngine
from src.api.types.usage_info import UsageInfo
from fastapi.responses import StreamingResponse
from src.api.utils.format_sse import format_sse
from fastapi import APIRouter, HTTPException, Depends
from src.api.utils.dependencies import get_inference_engine
from transformers.generation.streamers import TextIteratorStreamer
from src.api.types.chat_completions import (
    GeneratedMessage,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
)

router = APIRouter()


async def _stream_completion(request: ChatCompletionRequest, engine: InferenceEngine):
    messages = [msg.model_dump(exclude_none=True) for msg in request.messages]
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
        target=engine.generate_chat_completions,
        kwargs=dict(
            conversation=messages, generation_config=gen_cfg, streamer=streamer
        ),
    )
    thread.start()

    finish_reason = None
    all_text = ""

    try:
        for text in streamer:
            if not text:
                continue

            all_text += text
            # Construct Response Body
            assistant_message = GeneratedMessage(content=text)
            choice = ChatCompletionChoice(
                delta=assistant_message, finish_reason=finish_reason
            )
            response = ChatCompletionResponse(model=request.model, choices=[choice])
            yield format_sse(response.model_dump())
        finish_reason = "stop"
    except Exception as e:
        print(e)
        finish_reason = "error"
    finally:
        thread.join()

        # Calculate usage
        num_prompt_tokens = len(engine.apply_chat_template(conversation=messages))
        num_completion_tokens = len(engine.tokenize(all_text))
        usage_info = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_completion_tokens,
            total_tokens=num_prompt_tokens + num_completion_tokens,
        )

        final_choice = ChatCompletionChoice(
            delta=GeneratedMessage(content=""), finish_reason=finish_reason
        )
        final_response = ChatCompletionResponse(
            model=request.model, choices=[final_choice], usage=usage_info
        )
        yield format_sse(final_response.model_dump())

        yield "data: [DONE]\n\n"  # finish sse


# Update main endpoint to inject engine
@router.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
)
async def create_chat_completion(
    request: ChatCompletionRequest,
    engine: InferenceEngine = Depends(get_inference_engine),  # Inject engine
):
    """
    Endpoint for chat completions. Handles both streaming and non-streaming.
    """
    if request.stream:
        return StreamingResponse(
            _stream_completion(request, engine), media_type="text/event-stream"
        )

    try:
        gen_cfg = request.gen_config()
        messages = [msg.model_dump(exclude_none=True) for msg in request.messages]
        generated_response, num_prompt_tokens, num_completion_tokens = (
            engine.generate_chat_completions(
                conversation=messages, generation_config=gen_cfg
            )
        )

        # Construct Response Body
        usage_info = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_completion_tokens,
            total_tokens=num_prompt_tokens + num_completion_tokens,
        )
        assistant_message = GeneratedMessage(content=generated_response)
        choice = ChatCompletionChoice(message=assistant_message, finish_reason="stop")
        response = ChatCompletionResponse(
            model=request.model, choices=[choice], usage=usage_info
        )
        return response

    except StopIteration:
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error: Generator yielded no response.",
        )
    except HTTPException as e:
        # Re-raise HTTPExceptions directly
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal Server Error during chat generation: {str(e)}",
        )
