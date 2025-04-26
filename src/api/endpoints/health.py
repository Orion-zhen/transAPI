from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def health_check(request: Request):
    """Basic health check endpoint, checks engine status."""
    engine_status = (
        "available"
        if hasattr(request.app.state, "engine") and request.app.state.engine is not None
        else "unavailable"
    )
    return {"status": "ok", "engine_status": engine_status}
