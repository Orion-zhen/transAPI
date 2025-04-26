from fastapi import APIRouter, Depends
from src.core.engine import InferenceEngine
from src.api.types.model_list import ModelCard, ModelList
from src.api.utils.dependencies import get_inference_engine

router = APIRouter()


@router.get("/models", response_model=ModelList)
@router.get("/v1/models", response_model=ModelList)
async def list_models(
    engine: InferenceEngine = Depends(get_inference_engine),
):  # Inject engine
    """
    Endpoint to list available models.
    Mimics https://platform.openai.com/docs/api-reference/models/list
    """
    assert isinstance(
        engine.settings.server.served_model_names, list
    ), "Model names must be a list"
    model_cards = [
        ModelCard(id=model_name)
        # Use the injected engine instance
        for model_name in engine.settings.server.served_model_names
    ]
    return ModelList(data=model_cards)
