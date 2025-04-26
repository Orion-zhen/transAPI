import os
import uvicorn
import logging
from fastapi import FastAPI, Request
from config.settings import load_config
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from src.core.engine import InferenceEngine
from fastapi.middleware.cors import CORSMiddleware
from src.api.endpoints import (
    health,
    tokenizer,
    list_models,
    completions,
    chat_completions,
)

if not os.path.exists("config/config.yaml"):
    settings = load_config("config/config.yaml.sample")
    print("Warning: config.yaml not found, using default settings.")
else:
    settings = load_config("config/config.yaml")
logging.basicConfig(level=settings.log.level.upper())
logger = logging.getLogger(" TransAPI ")


# --- Application Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the inference engine
    logger.info("Application startup: Loading inference engine...")
    try:
        engine = InferenceEngine(settings, logger)
        app.state.engine = engine  # Store the engine instance in app state
        logger.info("Inference engine loaded successfully.")
    except Exception as e:
        logger.error(
            f"FATAL: Failed to initialize InferenceEngine during startup: {e}",
            exc_info=True,
        )
        app.state.engine = None
        raise RuntimeError("Failed to initialize model engine during startup.") from e

    yield  # Application runs here

    # Shutdown: Clean up resources (optional)
    logger.info("Application shutdown: Cleaning up resources...")
    if hasattr(app.state, "engine") and app.state.engine is not None:
        del app.state.engine  # Remove from state
        logger.info("Inference engine resources released.")
    logger.info("Application shutdown complete.")


app = FastAPI(
    title="TransAPI - OpenAI Compatible API with Transformers",
    version="0.1.1",
    root_path=settings.server.root_path,
    lifespan=lifespan,  # Register the lifespan context manager
)
logger.info(
    f"FastAPI application initialized with root_path: '{settings.server.root_path}'"
)


# --- Global Exception Handler (Optional but recommended) ---
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal Server Error with {exc}"},
    )


def startup():
    if settings.cors.enabled:
        logger.info(
            f"Adding CORS middleware with origins: {settings.cors.allow_origins}"
        )
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors.allow_origins,
            allow_credentials=settings.cors.allow_credentials,
            allow_methods=settings.cors.allow_methods,
            allow_headers=settings.cors.allow_headers,
        )
    app.include_router(list_models.router, tags=["OpenAI - Models"])
    app.include_router(completions.router, tags=["OpenAI - Completions"])
    app.include_router(chat_completions.router, tags=["OpenAI - Chat Completions"])
    app.include_router(tokenizer.router, tags=["OpenAI - Tokenizer"])
    app.include_router(health.router, tags=["OpenAI - Health"])
    logger.info("Starting Uvicorn server...")

    uvicorn.run(
        app=app,
        # app="main:app", # Use this to enable reload
        host=settings.server.host,
        port=settings.server.port,
        log_level=settings.log.level.lower(),
        reload=False,  # Set to True for development (watches for code changes)
    )


if __name__ == "__main__":
    startup()
