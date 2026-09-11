from typing import Any

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------
# Request schema
# --------------------------------------------------

class ChatRequest(BaseModel):
    """
    Data sent by the client to POST /chat.
    """

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
    )

    @field_validator("message")
    @classmethod
    def validate_message(
        cls,
        value: str,
    ) -> str:

        value = value.strip()

        if not value:
            raise ValueError(
                "Message cannot be empty."
            )

        return value


# --------------------------------------------------
# Tool-call schema
# --------------------------------------------------

class ToolCallResponse(BaseModel):

    tool: str

    arguments: dict[str, Any]


# --------------------------------------------------
# Final API response
# --------------------------------------------------

class ChatResponse(BaseModel):

    user_message: str

    tool_call: ToolCallResponse

    tool_result: dict[str, Any]