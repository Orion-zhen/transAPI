from pydantic import BaseModel, Field


class Function(BaseModel):
    name: str = Field(default_factory=str)
    arguments: dict = Field(default_factory=dict)


class ToolCallRequest(BaseModel):
    type: str = Field("function")
    function: Function = Field(default_factory=Function)


class ToolResponse(BaseModel):
    role: str = Field("tool")
    name: str = Field(default_factory=str)
    content: str = Field(default_factory=str)
