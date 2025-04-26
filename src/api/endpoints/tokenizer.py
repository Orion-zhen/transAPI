from src.core.engine import InferenceEngine
from fastapi import APIRouter, HTTPException, Depends
from src.api.utils.dependencies import get_inference_engine
from src.api.types.tokenize import (
    TokenizeRequest,
    TokenizeResponse,
    DetokenizeRequest,
    DetokenizeResponse,
)

router = APIRouter()


@router.post("/tokenize", response_model=TokenizeResponse)
@router.post("/v1/tokenize", response_model=TokenizeResponse)
async def tokenize(
    request: TokenizeRequest,
    engine: InferenceEngine = Depends(get_inference_engine),  # Inject engine
):
    """Tokenizes the given text using the loaded model's tokenizer."""
    try:
        # Use the injected engine instance
        result = engine.tokenize(request.prompt)
        return TokenizeResponse(tokens=result, count=len(result))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Internal Server Error during tokenization. {e}"
        )


@router.post("/detokenize", response_model=DetokenizeResponse)
@router.post("/v1/detokenize", response_model=DetokenizeResponse)
async def detokenize(
    request: DetokenizeRequest,
    engine: InferenceEngine = Depends(get_inference_engine),  # Inject engine
):
    """Detokenizes the given list of token IDs."""
    try:
        # Use the injected engine instance
        result = engine.detokenize(request.tokens)
        return DetokenizeResponse(prompt=result)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Internal Server Error during detokenization. {e}"
        )
