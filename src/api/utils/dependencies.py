import logging
from src.core.engine import InferenceEngine
from fastapi import Request, HTTPException, status

logger = logging.getLogger(" TransAPI ")


def get_inference_engine(request: Request) -> InferenceEngine:
    """
    Dependency function to get the InferenceEngine instance.

    Relies on the engine being initialized during application startup
    and stored in request.app.state.engine.
    """
    if not hasattr(request.app.state, "engine") or request.app.state.engine is None:
        # This should ideally not happen if lifespan management is correct
        logger.error(
            "Inference engine dependency requested but not found in app state."
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine is not available or not yet initialized.",
        )
    return request.app.state.engine
