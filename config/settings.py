import yaml
from pydantic import BaseModel, Field


class ServerSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    root_path: str = ""
    served_model_names: list[str] | None = Field(default_factory=list)


class LogSettings(BaseModel):
    level: str = "info"
    prompt: bool = False
    params: bool = False
    completion: bool = False


class QuantizationSettings(BaseModel):
    bnb_4bit: bool = False
    bnb_8bit: bool = False


class ModelSettings(BaseModel):
    model_path: str
    quantization: QuantizationSettings = Field(default_factory=QuantizationSettings)
    device: str = "auto"
    precision: str = "bfloat16"


class CorsSettings(BaseModel):
    enabled: bool = False
    allow_origins: list[str] = ["*"]
    allow_credentials: bool = True
    allow_methods: list[str] = ["*"]
    allow_headers: list[str] = ["*"]


class AppSettings(BaseModel):
    server: ServerSettings = Field(default_factory=ServerSettings)
    model: ModelSettings
    log: LogSettings = Field(default_factory=LogSettings)
    cors: CorsSettings = Field(default_factory=CorsSettings)


def load_config(config_file: str = "config/config.yaml") -> AppSettings:
    """Loads configuration from a YAML file."""
    with open(config_file, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    # Basic validation: ensure required sections exist
    if "model" not in config_data or "model_path" not in config_data.get("model", {}):
        raise ValueError("Configuration error: 'model.model_path' is required.")
    original_model_name = config_data.get("model").get("model_path").split("/")[-1]
    if "server" not in config_data:
        config_data["server"] = ServerSettings(served_model_names=[original_model_name])
    if "served_model_names" not in config_data["server"]:
        config_data["server"]["served_model_names"] = [original_model_name]
    if config_data["server"]["served_model_names"] is None:
        config_data["server"]["served_model_names"] = list()
    if isinstance(config_data["server"]["served_model_names"], str):
        one_model_name = config_data["server"]["served_model_names"]
        config_data["server"]["served_model_names"] = [one_model_name]

    config_data["server"]["served_model_names"].append(original_model_name)

    return AppSettings(**config_data)
